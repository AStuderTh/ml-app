"""Onglet 'Prochains matchs': calendrier ATP en direct (source: API publique
ESPN, site.api.espn.com — pas de clé requise) pour les tournois en cours ou
à venir, filtré sur les matchs simples pas encore joués. Un match peut être
sélectionné dans la liste pour afficher ses stats et une prédiction d'un
modèle ML sauvegardé."""
import datetime as dt
import json

import pandas as pd
import requests
import streamlit as st

from app import bet_log, odds_log, store
from app.backtest import DEFAULT_STRATEGY
from app.live_match import (
    FEATURE_LABELS, build_live_features, compute_current_player_state,
    player_snapshot, resolve_player,
)
from app.odds import attach_odds, sport_key_for_tournament
from app.oddsapiio import FREE_BOOKMAKERS as ODDS_API_IO_BOOKMAKERS
from app.oddsapiio import attach_oddsapiio_odds
from app.polymarket import attach_polymarket_odds

ODDS_PROVIDERS = {
    "Aucune": None,
    "💰 The Odds API": "the_odds_api",
    "🎾 odds-api.io": "odds_api_io",
}

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
DISPLAY_TZ = "Europe/Paris"

# Mise en valeur d'une opportunité de pari dans le tableau: seules les propriétés
# CSS de couleur d'un Styler pandas sont interprétées par st.dataframe (les autres,
# comme font-weight, sont ignorées sans erreur). Vert semi-transparent pour rester
# lisible en thème clair comme en thème sombre.
SIGNAL_CSS = "background-color: rgba(33, 195, 84, 0.28); font-weight: 600;"


def _fetch_week(date: dt.date) -> dict:
    resp = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": date.strftime("%Y%m%d")},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.espn.com/",
            "Origin": "https://www.espn.com",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner="Récupération du calendrier ATP (ESPN)...")
def get_upcoming_matches(days_ahead: int = 21) -> pd.DataFrame:
    """Interroge le scoreboard ESPN une fois par semaine sur la fenêtre demandée.

    Chaque réponse renvoie les événements actifs autour de la date demandée;
    un pas de 7 jours couvre les tournois hebdomadaires tout en évitant de
    déclencher les limites de requêtes d'ESPN. Les matchs simples pas encore
    joués (status 'pre') sont dédupliqués par id ESPN.
    """
    today = dt.date.today()
    rows = {}
    errors = []
    successful_fetches = 0
    for offset in range(0, days_ahead + 1, 7):
        date = today + dt.timedelta(days=offset)
        try:
            data = _fetch_week(date)
        except Exception as exc:
            errors.append(f"{date.isoformat()}: {exc}")
            continue
        successful_fetches += 1
        for event in data.get("events", []):
            tourney_name = event.get("name")
            for grouping in event.get("groupings", []):
                if grouping.get("grouping", {}).get("slug") != "mens-singles":
                    continue
                for comp in grouping.get("competitions", []):
                    if comp["status"]["type"]["state"] != "pre":
                        continue
                    comp_id = comp["id"]
                    if comp_id in rows:
                        continue
                    competitors = sorted(comp.get("competitors", []), key=lambda c: c.get("order", 0))
                    names = [c.get("athlete", {}).get("displayName", "TBD") for c in competitors]
                    names += ["TBD"] * (2 - len(names))
                    rows[comp_id] = {
                        "tournoi": tourney_name,
                        "date_utc": comp.get("date"),
                        "round": (comp.get("round") or {}).get("displayName", ""),
                        "joueur_1": names[0],
                        "joueur_2": names[1],
                        "lieu": (comp.get("venue") or {}).get("fullName", ""),
                    }

    cols = ["tournoi", "date_utc", "round", "joueur_1", "joueur_2", "lieu"]
    if not rows:
        empty = pd.DataFrame(columns=cols)
        empty.attrs["fetch_errors"] = errors
        empty.attrs["successful_fetches"] = successful_fetches
        return empty

    df = pd.DataFrame(rows.values())
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    df["date_locale"] = df["date_utc"].dt.tz_convert(DISPLAY_TZ)
    df = df.sort_values("date_utc").reset_index(drop=True)
    df.attrs["fetch_errors"] = errors
    df.attrs["successful_fetches"] = successful_fetches
    return df


@st.cache_data(show_spinner=False)
def _guess_tourney_context(tourney_name: str, dataset: pd.DataFrame) -> dict:
    """Devine surface/best_of/indoor d'un tournoi à venir en cherchant les
    valeurs du match le plus récent joué sous un nom de tournoi équivalent
    dans l'historique (gère les changements de sponsor via
    canonicalize_tourney). best_of/indoor alimentent les features
    best_of_5/is_indoor de app.features en direct (cf. live_match.build_live_features)."""
    from scripts.consolidate.normalize import canonicalize_tourney, norm_tourney_name

    target = canonicalize_tourney(norm_tourney_name(tourney_name))
    hist_names = dataset["tourney_name"].dropna().unique()
    matching = [n for n in hist_names if canonicalize_tourney(norm_tourney_name(n)) == target]
    if not matching:
        return {"surface": "Hard", "best_of": None, "indoor": None}
    sub = dataset[dataset["tourney_name"].isin(matching)].sort_values("tourney_date")

    def _last_known(col):
        s = sub[col].dropna()
        return s.iloc[-1] if not s.empty else None

    return {
        "surface": _last_known("surface") or "Hard",
        "best_of": _last_known("best_of"),
        "indoor": _last_known("indoor"),
    }


def _implied_prob_from_row(row) -> float | None:
    """Probabilité implicite p1 à utiliser comme feature d'entrée du modèle
    (si le modèle l'utilise): priorité au bookmaker actuellement attaché au
    tableau (The Odds API ou odds-api.io), Polymarket en repli — même
    logique que dans _render_match_detail."""
    for o1_col, o2_col in (("cote_j1", "cote_j2"), ("oio_j1", "oio_j2")):
        o1, o2 = row.get(o1_col), row.get(o2_col)
        if pd.notna(o1) and pd.notna(o2) and o1 > 1 and o2 > 1:
            i1, i2 = 1.0 / float(o1), 1.0 / float(o2)
            return i1 / (i1 + i2)
    p1, p2 = row.get("pm_prob_j1"), row.get("pm_prob_j2")
    if pd.notna(p1) and pd.notna(p2) and p1 > 0 and p2 > 0:
        return float(p1)
    return None


def _signal_odds_from_row(row) -> tuple[float | None, float | None, str | None]:
    """Cotes servant à DÉCIDER de parier: bookmaker uniquement (The Odds API,
    puis odds-api.io). Volontairement PAS de repli Polymarket ici — le seuil
    d'edge de la règle de sélection a été calibré en backtest contre des cotes
    bookmaker, qui incluent une marge (surround ~5%); les probabilités
    Polymarket sont, elles, quasi normalisées (somme ~1, pas de marge), donc
    le même seuil appliqué à Polymarket ne sélectionne pas la même population
    de paris que celle mesurée en backtest. Cotes BRUTES (non dévigorisées),
    comme dans app.backtest.run_backtest."""
    for o1_col, o2_col, label in (
        ("cote_j1", "cote_j2", "The Odds API"),
        ("oio_j1", "oio_j2", "odds-api.io"),
    ):
        o1, o2 = row.get(o1_col), row.get(o2_col)
        if pd.notna(o1) and pd.notna(o2) and o1 > 1 and o2 > 1:
            return float(o1), float(o2), label
    return None, None, None


def _polymarket_odds_from_row(row) -> tuple[float | None, float | None]:
    """Cotes décimales équivalentes Polymarket (1/prix du marché), pour
    l'EXÉCUTION du pari une fois qu'un signal bookmaker existe. `outcomePrices`
    de la Gamma API est un prix de marché (mid), pas un carnet d'ordres: la
    cote réellement obtenue à l'achat sera légèrement moins bonne (spread /
    slippage), ces cotes sont donc indicatives."""
    p1, p2 = row.get("pm_prob_j1"), row.get("pm_prob_j2")
    if pd.notna(p1) and pd.notna(p2) and 0 < float(p1) < 1 and 0 < float(p2) < 1:
        return 1.0 / float(p1), 1.0 / float(p2)
    return None, None


def _attach_bet_signals(sub: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """Applique la règle de sélection de la stratégie choisie (edge_threshold,
    stake_mode, kelly_fraction) aux probabilités du modèle pour marquer les
    opportunités de pari sur les matchs à venir — transposition exacte de la
    logique de sélection de app.backtest.run_backtest, pour que le signal
    affiché ici corresponde à ce qui a été mesuré en backtest.

    La DÉCISION (parier ou non, et sur qui) se prend exclusivement sur les
    cotes bookmaker — seule source contre laquelle le seuil d'edge a été
    validé. L'EXÉCUTION est ensuite routée vers le marché qui offre la
    meilleure cote pour ce même joueur (bookmaker ou Polymarket): une cote
    plus haute sur le pari déjà décidé ne fait qu'augmenter le gain, elle ne
    change pas la population de paris sélectionnée, donc le ROI du backtest
    reste un plancher.

    Ajoute bet_side ('j1'/'j2'/None), bet_edge/bet_odds/bet_source (le signal,
    côté bookmaker), exec_odds/exec_venue/exec_gain (où miser réellement) et
    bet_stake. Reste vide si la probabilité du modèle est indisponible ou si
    aucun bookmaker ne couvre le match."""
    sub = sub.copy()
    for col in ("bet_side", "bet_source", "exec_venue"):
        sub[col] = None
    for col in ("bet_edge", "bet_odds", "exec_odds", "exec_gain", "bet_stake"):
        sub[col] = pd.NA
    if not strategy:
        return sub

    strat = {**DEFAULT_STRATEGY, **strategy}
    for i, row in sub.iterrows():
        model_p1 = row["model_prob_j1"]
        if pd.isna(model_p1):
            continue
        o1, o2, source = _signal_odds_from_row(row)
        if o1 is None:
            continue

        model_p1 = float(model_p1)
        model_p2 = 1.0 - model_p1
        edge_1, edge_2 = model_p1 - 1.0 / o1, model_p2 - 1.0 / o2
        if edge_1 >= strat["edge_threshold"] and edge_1 >= edge_2:
            side, edge, odds, p = "j1", edge_1, o1, model_p1
        elif edge_2 >= strat["edge_threshold"]:
            side, edge, odds, p = "j2", edge_2, o2, model_p2
        else:
            continue

        # Routage de l'exécution: on ne compare que le côté effectivement joué
        # (l'autre côté ne sera pas misé, sa cote n'a aucune importance).
        exec_odds, exec_venue = odds, source
        pm_o1, pm_o2 = _polymarket_odds_from_row(row)
        if pm_o1 is not None:
            pm_odds = pm_o1 if side == "j1" else pm_o2
            if pm_odds > exec_odds:
                exec_odds, exec_venue = pm_odds, "Polymarket"

        if strat["stake_mode"] == "kelly":
            # Kelly calculé sur la cote réellement obtenue (exec_odds): c'est ce
            # payout-là qui détermine la mise optimale. Bankroll = celle de la
            # stratégie (valeur de départ du backtest), aucune bankroll réelle
            # n'étant suivie ici — la mise reste donc indicative.
            b = exec_odds - 1.0
            f = ((p * b - (1 - p)) / b) if b > 0 else 0.0
            stake = strat["bankroll"] * max(0.0, min(f * (strat["kelly_fraction"] or 0.0), 1.0))
        else:
            stake = strat["flat_stake"]
        if stake <= 0:
            continue

        sub.loc[i, "bet_side"] = side
        sub.loc[i, "bet_edge"] = edge
        sub.loc[i, "bet_odds"] = odds
        sub.loc[i, "bet_source"] = source
        sub.loc[i, "exec_odds"] = exec_odds
        sub.loc[i, "exec_venue"] = exec_venue
        sub.loc[i, "exec_gain"] = exec_odds / odds - 1.0
        sub.loc[i, "bet_stake"] = stake
    return sub


def _style_signals(display: pd.DataFrame, sides: pd.Series):
    """Colore en vert la case du joueur sur lequel la stratégie envoie un signal
    de pari (et la case '🎯 Pari' de la même ligne). `sides` doit partager
    l'index de `display` (les deux viennent du même `sub` réindexé)."""
    css = pd.DataFrame("", index=display.index, columns=display.columns)
    for i, side in sides.items():
        if side not in ("j1", "j2"):
            continue
        for col in ("Joueur 1" if side == "j1" else "Joueur 2", "🎯 Pari"):
            if col in css.columns:
                css.loc[i, col] = SIGNAL_CSS
    return display.style.apply(lambda _: css, axis=None)


def _attach_model_predictions(sub: pd.DataFrame, model_row, state: dict, tourney_ctx: dict = None) -> pd.DataFrame:
    """Ajoute model_prob_j1/model_prob_j2 (probabilité de victoire prédite
    par le modèle sauvegardé sélectionné au-dessus du tableau) à chaque
    ligne de `sub`, en réutilisant l'état courant des joueurs déjà calculé
    (`state`, cf. compute_current_player_state). Reste à NaN si un joueur
    est TBD, introuvable dans l'historique, ou si une feature requise par le
    modèle est manquante pour ce match."""
    sub = sub.copy()
    sub["model_prob_j1"] = pd.NA
    sub["model_prob_j2"] = pd.NA
    if model_row is None or state is None:
        return sub

    model = store.load_model_object(model_row["id"])
    if model is None:
        return sub
    features = json.loads(model_row["features_json"])
    tourney_ctx = tourney_ctx or {}

    for i, row in sub.iterrows():
        if row["joueur_1"] == "TBD" or row["joueur_2"] == "TBD":
            continue
        canon1 = resolve_player(row["joueur_1"], state)
        canon2 = resolve_player(row["joueur_2"], state)
        if canon1 is None or canon2 is None:
            continue
        surface = row.get("surface") or "Hard"
        snap1 = player_snapshot(state, canon1, surface)
        snap2 = player_snapshot(state, canon2, surface)
        implied_prob_p1 = _implied_prob_from_row(row)
        values, missing = build_live_features(
            snap1, snap2, state, canon1, canon2, features, surface, implied_prob_p1,
            best_of=tourney_ctx.get("best_of"), indoor=tourney_ctx.get("indoor"),
        )
        if missing:
            continue
        X = pd.DataFrame([values])[features].values
        prob_p1 = float(model.predict_proba(X)[0, 1])
        sub.loc[i, "model_prob_j1"] = prob_p1
        sub.loc[i, "model_prob_j2"] = 1 - prob_p1
    return sub


def _render_bet_signal(match: dict, strategy: dict, strategy_name: str = None):
    """Bandeau de recommandation en tête du détail d'un match, à partir du
    signal déjà calculé par _attach_bet_signals sur la ligne sélectionnée."""
    if not strategy:
        return
    side = match.get("bet_side")
    if side not in ("j1", "j2"):
        st.info(
            f"Aucun signal de pari sur ce match pour la stratégie **{strategy_name}** "
            f"(edge insuffisant, probabilité du modèle non calculable, ou aucune cote "
            f"bookmaker disponible — le signal ne se prend jamais sur Polymarket seul)."
        )
        return

    player = match["joueur_1"] if side == "j1" else match["joueur_2"]
    strat = {**DEFAULT_STRATEGY, **strategy}
    if strat["stake_mode"] == "kelly":
        stake_txt = (f"{match['bet_stake']:.2f} € — Kelly {strat['kelly_fraction']:.0%} "
                     f"sur une bankroll de {strat['bankroll']:.0f} €")
    else:
        stake_txt = f"{match['bet_stake']:.2f} € (mise fixe)"

    exec_line = f"**Où miser :** {match['exec_venue']} @ **{match['exec_odds']:.2f}**"
    if match["exec_venue"] == "Polymarket":
        gain = match["exec_gain"]
        exec_line += (f" — soit **+{gain*100:.1f}% de gain** par rapport à la cote "
                      f"{match['bet_source']} ({match['bet_odds']:.2f}) qui a déclenché le signal "
                      f"(cote Polymarket indicative : prix de marché, hors spread)")
    st.success(
        f"🎯 **Opportunité — miser sur {player}**  \n"
        f"**Signal :** edge **+{match['bet_edge']*100:.1f} pts** vs {match['bet_source']} "
        f"@ {match['bet_odds']:.2f} (seuil de la stratégie : {strat['edge_threshold']*100:.1f} pts)  \n"
        f"{exec_line}  \n"
        f"**Mise conseillée :** {stake_txt}"
    )


def _render_match_detail(match: dict, dataset: pd.DataFrame, model_row=None,
                          strategy: dict = None, strategy_name: str = None):
    st.divider()
    st.markdown(f"### 🔍 {match['joueur_1']} vs {match['joueur_2']}")
    st.caption(f"{match['tournoi']} — {match['round']} — {match['date_locale'].strftime('%a %d/%m %H:%M')}")

    _render_bet_signal(match, strategy, strategy_name)

    if match["joueur_1"] == "TBD" or match["joueur_2"] == "TBD":
        st.info("Adversaires pas encore connus (tirage au sort pas encore sorti) — pas de stats possibles.")
        return

    if dataset is None:
        st.warning("Base de données indisponible — impossible de calculer les stats.")
        return

    state = compute_current_player_state(dataset)
    canon1 = resolve_player(match["joueur_1"], state)
    canon2 = resolve_player(match["joueur_2"], state)
    if canon1 is None or canon2 is None:
        missing = match["joueur_1"] if canon1 is None else match["joueur_2"]
        st.warning(f"'{missing}' introuvable dans l'historique de data/tennis.db — pas de stats possibles.")
        return

    tourney_ctx = _guess_tourney_context(match["tournoi"], dataset)
    surface = match.get("surface") or tourney_ctx["surface"]
    snap1 = player_snapshot(state, canon1, surface)
    snap2 = player_snapshot(state, canon2, surface)

    st.caption(
        f"Surface supposée : **{surface}** (déduite de l'historique du tournoi). "
        f"Noms retrouvés dans la base : **{canon1}** vs **{canon2}**."
    )

    def _fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%d/%m/%Y")
        if isinstance(v, float):
            return f"{v:.0f}"
        return str(v)

    stat_rows = [
        ("Elo global", f"{snap1['elo']:.0f}", f"{snap2['elo']:.0f}"),
        ("Elo global (pondéré récence)", f"{snap1['elo_recent']:.0f}", f"{snap2['elo_recent']:.0f}"),
        (f"Elo {surface}", f"{snap1['elo_surface']:.0f}", f"{snap2['elo_surface']:.0f}"),
        (f"Elo {surface} (pondéré récence)", f"{snap1['elo_surface_recent']:.0f}", f"{snap2['elo_surface_recent']:.0f}"),
        ("Classement ATP", _fmt(snap1["rank"]), _fmt(snap2["rank"])),
        ("Points ATP", _fmt(snap1["rank_points"]), _fmt(snap2["rank_points"])),
        ("Âge (dernier connu)", _fmt(snap1["age"]), _fmt(snap2["age"])),
        ("Taille (cm)", _fmt(snap1["ht"]), _fmt(snap2["ht"])),
        ("Main", _fmt(snap1["hand"]), _fmt(snap2["hand"])),
        ("Forme (10 derniers)", f"{snap1['form']*100:.0f}%", f"{snap2['form']*100:.0f}%"),
        ("Matchs joués (historique)", _fmt(snap1["played"]), _fmt(snap2["played"])),
        ("Dernier match connu", _fmt(snap1["last_match_date"]), _fmt(snap2["last_match_date"])),
    ]
    stats_df = pd.DataFrame(stat_rows, columns=["Stat", match["joueur_1"], match["joueur_2"]])
    st.dataframe(stats_df, width="stretch", hide_index=True)

    # --- Comparaison des sources (bookmaker / Polymarket / modèle ML) -----
    def _implied_probs(o1, o2):
        if pd.isna(o1) or pd.isna(o2) or o1 <= 1 or o2 <= 1:
            return None, None
        i1, i2 = 1.0 / float(o1), 1.0 / float(o2)
        total = i1 + i2
        return i1 / total, i2 / total

    def _add_source_row(rows, label, p1, p2, o1, o2):
        rows.append({"source": label, "p1": p1, "p2": p2, "o1": o1, "o2": o2})

    comparison = []
    bookmaker_label = None
    if pd.notna(match.get("cote_j1")) and pd.notna(match.get("cote_j2")):
        bookmaker_label, bo1, bo2 = "💰 The Odds API", float(match["cote_j1"]), float(match["cote_j2"])
    elif pd.notna(match.get("oio_j1")) and pd.notna(match.get("oio_j2")):
        bookmaker_label, bo1, bo2 = "💰 odds-api.io", float(match["oio_j1"]), float(match["oio_j2"])
    if bookmaker_label:
        p1, p2 = _implied_probs(bo1, bo2)
        _add_source_row(comparison, bookmaker_label, p1, p2, bo1, bo2)

    pm1, pm2 = match.get("pm_prob_j1"), match.get("pm_prob_j2")
    if pd.notna(pm1) and pd.notna(pm2) and pm1 > 0 and pm2 > 0:
        pm1, pm2 = float(pm1), float(pm2)
        _add_source_row(comparison, "📊 Polymarket", pm1, pm2, 1 / pm1, 1 / pm2)

    st.markdown("#### 🤖 Prédiction d'un modèle sauvegardé")
    if model_row is None:
        st.info(
            "Aucun modèle sauvegardé, ou aucun sélectionné — choisis-en un dans le menu déroulant "
            "'🤖 Modèle pour les colonnes de prédiction' au-dessus du tableau."
        )
    else:
        # implied_prob_p1 utilisé comme FEATURE d'entrée du modèle (si le modèle
        # l'utilise): priorité au bookmaker (cohérent avec les cotes utilisées à
        # l'entraînement), Polymarket en repli.
        implied_prob_p1 = None
        if comparison:
            implied_prob_p1 = comparison[0]["p1"]

        features = json.loads(model_row["features_json"])
        values, missing = build_live_features(
            snap1, snap2, state, canon1, canon2, features, surface, implied_prob_p1,
            best_of=tourney_ctx["best_of"], indoor=tourney_ctx["indoor"],
        )

        with st.expander("Détail des features utilisées par ce modèle"):
            feat_df = pd.DataFrame([
                {"Feature": FEATURE_LABELS.get(f, f), "Valeur": ("—" if v is None else round(v, 3))}
                for f, v in values.items()
            ])
            st.dataframe(feat_df, width="stretch", hide_index=True)

        if missing:
            st.warning(
                f"Prédiction impossible : données manquantes pour "
                f"{', '.join(FEATURE_LABELS.get(f, f) for f in missing)}."
            )
        else:
            model = store.load_model_object(model_row["id"])
            if model is None:
                st.error("Impossible de charger ce modèle sauvegardé (fichier introuvable).")
            else:
                X = pd.DataFrame([values])[features].values
                prob_p1 = float(model.predict_proba(X)[0, 1])
                prob_p2 = 1 - prob_p1
                _add_source_row(
                    comparison, f"🤖 {model_row['name']}", prob_p1, prob_p2,
                    (1 / prob_p1 if prob_p1 > 0 else None), (1 / prob_p2 if prob_p2 > 0 else None),
                )

    if not comparison:
        st.info("Aucune source de cotes/probabilités disponible pour ce match pour l'instant.")
        return

    st.markdown("#### ⚖️ Comparaison des sources")
    comp_df = pd.DataFrame([
        {
            "Source": r["source"],
            f"Proba {match['joueur_1']}": f"{r['p1']*100:.1f}%" if r["p1"] is not None else "—",
            f"Cote {match['joueur_1']}": f"{r['o1']:.2f}" if r["o1"] is not None else "—",
            f"Proba {match['joueur_2']}": f"{r['p2']*100:.1f}%" if r["p2"] is not None else "—",
            f"Cote {match['joueur_2']}": f"{r['o2']:.2f}" if r["o2"] is not None else "—",
        }
        for r in comparison
    ])
    st.dataframe(comp_df, width="stretch", hide_index=True)

    # écarts du modèle (s'il a été calculé) vs chaque source de marché
    model_rows = [r for r in comparison if r["source"].startswith("🤖")]
    market_rows = [r for r in comparison if not r["source"].startswith("🤖")]
    if model_rows and market_rows:
        model_p1 = model_rows[0]["p1"]
        edges = "  ·  ".join(
            f"vs {r['source']}: {(model_p1 - r['p1'])*100:+.1f} pts" for r in market_rows if r["p1"] is not None
        )
        st.caption(f"Écart du modèle sur la probabilité de victoire de {match['joueur_1']} — {edges}")


def _update_all_strategies_signals(df: pd.DataFrame, dataset: pd.DataFrame, tournois: list) -> tuple[int, int]:
    """Rejoue le pipeline de signal (prédiction modèle + règle de sélection) pour
    CHAQUE stratégie enregistrée — pas seulement celle choisie dans le menu
    déroulant au-dessus du tableau — et journalise les nouveaux paris (cf.
    app.bet_log.record_signals). Permet de tenir à jour l'historique de toutes
    les stratégies en un clic plutôt que de les sélectionner une par une.

    `df` doit déjà porter les colonnes de cotes (Polymarket + bookmaker choisi),
    comme le `df` de render_upcoming au moment de l'affichage. Les prédictions
    du modèle sont mise en cache par (model_id, tournoi) pour éviter de les
    recalculer plusieurs fois quand deux stratégies partagent le même modèle.

    Retourne (nombre de stratégies exploitables, nombre de paris ajoutés)."""
    strategies_df = store.list_strategies()
    if strategies_df.empty or df.empty or dataset is None:
        return 0, 0
    saved_models = store.list_models()
    state = compute_current_player_state(dataset)

    predictions_cache = {}  # (model_id, tourney) -> sub avec model_prob_j1/j2
    n_strategies, n_bets_logged = 0, 0
    for _, strat_row in strategies_df.iterrows():
        model_matches = saved_models[saved_models["id"] == strat_row["model_id"]]
        if model_matches.empty:
            continue
        model_row = model_matches.iloc[0]
        if not store.artifact_exists(model_row):
            continue
        strategy = json.loads(strat_row["strategy_json"])
        n_strategies += 1

        for tourney in tournois:
            sub = df[df["tournoi"] == tourney].reset_index(drop=True)
            if sub.empty:
                continue
            cache_key = (model_row["id"], tourney)
            if cache_key not in predictions_cache:
                tourney_ctx = _guess_tourney_context(tourney, dataset)
                sub_ctx = sub.copy()
                sub_ctx["surface"] = tourney_ctx.get("surface")
                predictions_cache[cache_key] = _attach_model_predictions(sub_ctx, model_row, state, tourney_ctx)
            sub_signals = _attach_bet_signals(predictions_cache[cache_key], strategy)
            n_bets_logged += bet_log.record_signals(sub_signals, strat_row["id"], strat_row["name"], strategy)

    return n_strategies, n_bets_logged


def render_upcoming(dataset: pd.DataFrame = None):
    st.subheader("📅 Prochains matchs (ATP)")
    st.caption(
        "Calendrier en direct via l'API publique ESPN — tournois ATP en cours ou à venir. "
        "Heures affichées en Europe/Paris. Le tableau d'un tournoi n'est publié par l'ATP que "
        "quelques jours avant son début : au-delà, les matchs existent déjà (bon nombre de "
        "'Round 1', 'Round 2'...) mais les joueurs restent en 'TBD' tant que le tirage au sort "
        "n'est pas sorti. Sélectionne une ligne dans un tableau pour voir le détail du match."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    days_ahead = c1.slider("Fenêtre (jours)", 3, 45, 21, key="upc_days")
    hide_tbd = c2.checkbox("Masquer les 'TBD'", value=False, key="upc_hide_tbd")
    if c3.button("🔄 Rafraîchir le calendrier"):
        get_upcoming_matches.clear()
        fresh_df = get_upcoming_matches(days_ahead)
        # assignation explicite (pas juste un pop) puis rerun: c'est ce qui force le
        # widget multiselect à se resynchroniser visuellement avec "tout sélectionné"
        # (un simple pop + default laisse parfois le widget affiché vide malgré une
        # valeur interne correcte)
        st.session_state["upc_tournois"] = list(dict.fromkeys(fresh_df["tournoi"])) if not fresh_df.empty else []
        st.rerun()

    df = get_upcoming_matches(days_ahead)
    if df.empty:
        errors = df.attrs.get("fetch_errors", [])
        successful_fetches = df.attrs.get("successful_fetches", 0)
        if errors and not successful_fetches:
            st.warning(
                "Impossible de récupérer le calendrier ESPN pour la fenêtre demandée. "
                f"Dernière erreur : {errors[-1]}"
            )
        else:
            st.info("Aucun match à venir trouvé (tournois ATP en pause, ou API indisponible).")
        return

    if hide_tbd:
        df = df[(df["joueur_1"] != "TBD") & (df["joueur_2"] != "TBD")]

    tournois = list(dict.fromkeys(df["tournoi"]))  # ordre d'apparition (par date), sans doublons
    selected = st.multiselect("Tournois", tournois, default=tournois, key="upc_tournois")
    selected = selected or tournois  # sélection vide -> on affiche quand même tout
    df = df[df["tournoi"].isin(selected)]

    covered = {t for t in tournois if sport_key_for_tournament(t)}

    # Polymarket: toujours récupéré et affiché (gratuit, sans clé, large couverture)
    df, pm_info = attach_polymarket_odds(df)
    st.caption(
        "📊 Polymarket (marché prédictif, sans bookmaker) — toujours affiché, couvre tour "
        "principal ET Challengers, mais un marché n'existe que si le tirage au sort du tour "
        "est déjà sorti. Probabilité implicite (%) + cote décimale équivalente (1/proba). "
        + (pm_info or "")
    )

    c4, c5 = st.columns([1, 3])
    provider_labels = list(ODDS_PROVIDERS.keys())
    default_provider_idx = provider_labels.index("🎾 odds-api.io")
    provider_label = c4.selectbox(
        "Fournisseur de cotes bookmaker (en plus de Polymarket)",
        provider_labels, index=default_provider_idx, key="upc_odds_provider",
    )
    provider = ODDS_PROVIDERS[provider_label]

    PROVIDER_CAPTIONS = {
        "the_odds_api": (
            "The Odds API — couverture limitée aux Grand Chelems / Masters / ATP 500, et seulement "
            "une fois le marché ouvert (quelques jours avant le tournoi). "
            f"Tournoi(s) sélectionné(s) couvert(s) : {', '.join(sorted(covered)) or 'aucun'}."
        ),
        "odds_api_io": (
            "odds-api.io — couvre aussi les ATP 250 et Challengers. Plan gratuit limité à 2 "
            f"bookmakers imposés ({', '.join(ODDS_API_IO_BOOKMAKERS)}), meilleure cote entre les deux affichée."
        ),
    }
    if provider:
        c5.caption(PROVIDER_CAPTIONS[provider])

    info = None
    if provider == "the_odds_api":
        df, info = attach_odds(df)
        if info == "no_key":
            st.warning("Aucune clé ODDS_API_KEY configurée dans .streamlit/secrets.toml.")
        elif info is not None and not str(info).isdigit():
            st.warning(f"Erreur The Odds API : {info}")
        elif info is not None:
            st.caption(f"Requêtes The Odds API restantes ce mois-ci : {info}")
    elif provider == "odds_api_io":
        df, info = attach_oddsapiio_odds(df)
        if info == "no_key":
            st.warning("Aucune clé ODDS_API_IO_KEY configurée dans .streamlit/secrets.toml.")
        elif info:
            st.caption(info)

    st.markdown("#### 🎯 Stratégie / modèle appliqué aux matchs à venir")
    saved_models = store.list_models()
    strategies_df = store.list_strategies()
    model_row, strategy, strategy_name, strategy_id = None, None, None, None

    if saved_models.empty:
        st.caption(
            "Aucun modèle sauvegardé — entraîne et sauvegarde un modèle dans l'onglet "
            "'🛠️ Construire un modèle' pour afficher ses prédictions ici."
        )
    else:
        # Une stratégie (modèle + règle de sélection) permet en plus de signaler les
        # opportunités de pari; un modèle seul n'affiche que ses probabilités.
        choices = ([("strategy", r) for _, r in strategies_df.iterrows()]
                   + [("model", r) for _, r in saved_models.iterrows()])

        def _choice_label(idx):
            kind, r = choices[idx]
            if kind == "strategy":
                strat = json.loads(r["strategy_json"])
                return (f"🎯 {r['name']} — edge ≥ {strat.get('edge_threshold', 0)*100:.1f} pts, "
                        f"{strat.get('stake_mode')} (ROI backtest {r['roi']*100:+.1f}%, {int(r['n_bets'])} paris)")
            return f"🤖 {r['name']} ({r['algo']}, AUC {r['auc']:.3f}) — probabilités seules, sans signal"

        choice_idx = st.selectbox(
            "Stratégie (🎯 = signaux de pari mis en valeur) ou modèle seul (🤖)",
            range(len(choices)), format_func=_choice_label, key="upc_strategy_select",
        )
        kind, chosen = choices[choice_idx]
        if kind == "strategy":
            model_row = saved_models[saved_models["id"] == chosen["model_id"]].iloc[0]
            strategy = json.loads(chosen["strategy_json"])
            strategy_name = chosen["name"]
            strategy_id = chosen["id"]
        else:
            model_row = chosen

        if strategies_df.empty:
            st.caption(
                "Aucune stratégie créée — associe un modèle à une règle de sélection dans l'onglet "
                "'🎯 Stratégies' pour faire apparaître ici les opportunités de pari."
            )
        elif strategy is not None:
            st.caption(
                f"Modèle **{model_row['name']}** + règle **{chosen['rule_name']}**. "
                "**Décision** : un pari est signalé (case du joueur en vert) quand la probabilité du "
                "modèle dépasse d'au moins le seuil d'edge celle impliquée par la **cote bookmaker** "
                "— seule source contre laquelle la règle a été validée en backtest, Polymarket n'est "
                "jamais utilisé pour décider. **Exécution** : la colonne '🛒 Où miser' route ensuite le "
                "pari vers le marché offrant la meilleure cote pour ce joueur, souvent Polymarket "
                "(pas de marge bookmaker). Miser plus haut sur un pari déjà décidé ne peut qu'améliorer "
                "le résultat : le ROI du backtest reste un plancher."
            )

        if not strategies_df.empty:
            if st.button(
                "🔄 Mettre à jour les paris (signaux) pour toutes les stratégies",
                key="upc_update_all_strategies",
                help="Rejoue le pipeline de signal pour CHAQUE stratégie enregistrée (pas "
                     "seulement celle sélectionnée ci-dessus) sur les tournois affichés, et "
                     "journalise les nouveaux paris — sans avoir à les sélectionner une par une.",
            ):
                with st.spinner("Recalcul des signaux pour toutes les stratégies..."):
                    n_strat, n_added = _update_all_strategies_signals(df, dataset, tournois)
                if n_strat == 0:
                    st.warning(
                        "Aucune stratégie exploitable : soit aucun modèle associé n'est "
                        "retrouvable sur le disque, soit `data/tennis.db` est indisponible."
                    )
                else:
                    st.success(
                        f"✅ {n_strat} stratégie(s) rejouée(s) sur {len(tournois)} tournoi(s) — "
                        f"{n_added} nouveau(x) pari(s) enregistré(s) au journal."
                    )

    # Un modèle dont le .joblib a disparu ne produirait aucune prédiction, en
    # silence: le tableau s'afficherait avec des probabilités vides, sans que
    # rien n'indique pourquoi. On le signale explicitement plutôt que de laisser
    # l'utilisateur croire à un modèle sans opinion.
    if model_row is not None and not store.artifact_exists(model_row):
        st.warning(
            f"Le modèle **{model_row['name']}** est enregistré mais son fichier "
            f"`data/models/{model_row['id']}.joblib` est introuvable : aucune prédiction "
            "ne peut être calculée. Le dossier `data/` n'étant pas versionné, les modèles "
            "entraînés sur une autre machine ou effacés depuis doivent être ré-entraînés "
            "(onglet 🛠️ Construire un modèle)."
        )
        model_row, strategy, strategy_id = None, None, None

    state = compute_current_player_state(dataset) if (model_row is not None and dataset is not None) else None

    st.caption(f"{len(df)} matchs à venir sur la période sélectionnée")

    def _fmt_decimal(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    def _fmt_pm(v):
        if pd.isna(v) or v <= 0:
            return "—"
        return f"{v*100:.1f}% ({1/v:.2f})"

    def _fmt_book(o1, o2):
        """Cote bookmaker affichée avec la probabilité de victoire qu'elle
        implique. La probabilité est DÉVIGORISÉE (répartie au prorata sur les
        deux joueurs, donc somme = 100%): l'inverse brut de la cote inclut la
        marge du bookmaker et les deux côtés sommeraient à ~105%, ce qui
        surestimerait les deux joueurs à la fois et ne serait pas comparable à
        la colonne Polymarket, qui est elle nativement sans marge. La cote
        brute reste affichée entre parenthèses: c'est elle qui détermine le
        gain, et c'est elle qui sert au calcul du signal (cf. _signal_odds_from_row)."""
        if pd.isna(o1) or pd.isna(o2) or o1 <= 1 or o2 <= 1:
            return _fmt_decimal(o1)
        i1, i2 = 1.0 / float(o1), 1.0 / float(o2)
        return f"{i1/(i1+i2)*100:.1f}% ({float(o1):.2f})"

    selected_match = None
    signal_banner = st.empty()  # rempli après la boucle: total d'opportunités tous tournois confondus
    total_signals = 0
    total_pm_exec = 0  # parmi elles, celles à exécuter sur Polymarket (meilleure cote)

    # Journal des prix: lu ICI (avant la boucle) depuis session_state car le
    # widget qui le pilote est rendu en bas de page, après les tableaux.
    log_prices = st.session_state.get("upc_log_prices", True)
    n_logged = 0

    # Journal des paris (app.bet_log): même mécanique, piloté par un widget rendu
    # en bas de page. Chaque signal de la stratégie sélectionnée est enregistré
    # comme un pari flat, aux cotes du moment — c'est ce journal qui alimente
    # l'onglet 💰 Bankroll.
    log_bets = st.session_state.get("upc_log_bets", True)
    n_bets_logged = 0

    for tourney in tournois:
        sub = df[df["tournoi"] == tourney].reset_index(drop=True)
        if sub.empty:
            continue
        tourney_ctx = _guess_tourney_context(tourney, dataset) if dataset is not None else {}
        sub["surface"] = tourney_ctx.get("surface")
        sub = _attach_model_predictions(sub, model_row, state, tourney_ctx)
        sub = _attach_bet_signals(sub, strategy)
        n_signals = int(sub["bet_side"].notna().sum())
        total_signals += n_signals
        total_pm_exec += int((sub["exec_venue"] == "Polymarket").sum())

        if log_prices:
            n_logged += odds_log.log_snapshots(
                sub, _signal_odds_from_row,
                model_id=model_row["id"] if model_row is not None else None,
            )
        if log_bets and strategy_id and n_signals:
            n_bets_logged += bet_log.record_signals(sub, strategy_id, strategy_name, strategy)

        title = f"🏆 {tourney} ({len(sub)} matchs)"
        if n_signals:
            title += f" — 🎯 {n_signals} opportunité{'s' if n_signals > 1 else ''}"
        with st.expander(title, expanded=len(sub) <= 20 or n_signals > 0):
            display = sub.copy()
            display["Date"] = display["date_locale"].dt.strftime("%a %d/%m %H:%M")
            display["surface"] = display["surface"].fillna("?")
            cols = ["Date", "round", "joueur_1", "joueur_2"]
            rename = {"round": "Round", "joueur_1": "Joueur 1", "joueur_2": "Joueur 2",
                      "surface": "Surface", "lieu": "Lieu"}

            # Colonnes du signal placées juste après les joueurs: c'est l'info la
            # plus actionnable du tableau, elle ne doit pas demander de scroller.
            if strategy is not None:
                display["signal_player"] = sub.apply(
                    lambda r: "—" if r["bet_side"] not in ("j1", "j2")
                    else f"✅ {r['joueur_1'] if r['bet_side'] == 'j1' else r['joueur_2']}", axis=1,
                )
                display["signal_edge"] = sub["bet_edge"].apply(
                    lambda e: f"+{e*100:.1f} pts" if pd.notna(e) else "—")
                display["signal_odds"] = sub["bet_odds"].apply(_fmt_decimal)
                display["exec_venue"] = sub.apply(
                    lambda r: "—" if pd.isna(r["exec_odds"])
                    else (f"📊 Polymarket {r['exec_odds']:.2f} (+{r['exec_gain']*100:.1f}%)"
                          if r["exec_venue"] == "Polymarket" else f"💰 {r['exec_venue']} {r['exec_odds']:.2f}"),
                    axis=1,
                )
                display["signal_stake"] = sub["bet_stake"].apply(
                    lambda s: f"{s:.2f}" if pd.notna(s) else "—")
                cols += ["signal_player", "signal_edge", "signal_odds", "exec_venue", "signal_stake"]
                rename.update({"signal_player": "🎯 Pari", "signal_edge": "Edge (signal)",
                               "signal_odds": "Cote signal", "exec_venue": "🛒 Où miser",
                               "signal_stake": "Mise"})

            cols += ["surface", "lieu"]

            # Polymarket: toujours affiché
            display["pm_j1"] = display["pm_prob_j1"].apply(_fmt_pm)
            display["pm_j2"] = display["pm_prob_j2"].apply(_fmt_pm)
            cols += ["pm_j1", "pm_j2"]
            rename.update({"pm_j1": "Polymarket J1", "pm_j2": "Polymarket J2"})

            # Cotes bookmaker: même format que Polymarket (proba % + cote), pour
            # que les deux marchés se comparent d'un coup d'œil sur la ligne.
            if provider == "the_odds_api":
                book_j1, book_j2 = "cote_j1", "cote_j2"
                book_label = "TOA"
            elif provider == "odds_api_io":
                book_j1, book_j2 = "oio_j1", "oio_j2"
                book_label = "odds.io"
            else:
                book_j1 = None

            if book_j1:
                display["book_j1"] = sub.apply(lambda r: _fmt_book(r[book_j1], r[book_j2]), axis=1)
                display["book_j2"] = sub.apply(lambda r: _fmt_book(r[book_j2], r[book_j1]), axis=1)
                cols += ["book_j1", "book_j2"]
                rename.update({"book_j1": f"💰 {book_label} J1", "book_j2": f"💰 {book_label} J2"})

            if model_row is not None:
                display["model_j1"] = display["model_prob_j1"].apply(_fmt_pm)
                display["model_j2"] = display["model_prob_j2"].apply(_fmt_pm)
                cols += ["model_j1", "model_j2"]
                rename.update({"model_j1": f"🤖 {model_row['name']} J1", "model_j2": f"🤖 {model_row['name']} J2"})

            display = display[cols].rename(columns=rename)
            table = _style_signals(display, sub["bet_side"]) if n_signals else display

            event = st.dataframe(
                table, width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row", key=f"upc_table_{tourney}",
            )
            rows = event.selection.rows if event and event.selection else []
            if rows:
                selected_match = sub.iloc[rows[0]].to_dict()

    if strategy is not None:
        if total_signals:
            plural = "s" if total_signals > 1 else ""
            routing = ""
            if total_pm_exec:
                routing = (f" Dont **{total_pm_exec}** à exécuter sur **Polymarket** plutôt que chez le "
                           f"bookmaker (cote plus élevée pour le même pari).")
            signal_banner.success(
                f"🎯 **{total_signals} opportunité{plural} de pari** détectée{plural} par la stratégie "
                f"**{strategy_name}** sur la période — joueur à jouer surligné en vert dans les tableaux "
                f"ci-dessous.{routing}"
            )
        elif provider is None:
            signal_banner.warning(
                "Aucun fournisseur de cotes bookmaker sélectionné : la décision de pari se prend "
                "uniquement sur les cotes bookmaker, aucun signal ne peut donc être calculé. "
                "Choisis 'The Odds API' ou 'odds-api.io' dans le menu 'Fournisseur de cotes' ci-dessus."
            )
        else:
            signal_banner.info(
                f"Aucune opportunité détectée par la stratégie **{strategy_name}** sur la période : "
                "soit aucun match n'atteint le seuil d'edge, soit les cotes bookmaker/probabilités du "
                "modèle manquent encore (le marché n'ouvre que quelques jours avant le tournoi)."
            )

    if selected_match is not None:
        _render_match_detail(selected_match, dataset, model_row, strategy, strategy_name)

    _render_bet_log_panel(n_bets_logged, strategy_id, strategy_name)
    _render_price_log_panel(n_logged)


def _render_bet_log_panel(n_logged: int, strategy_id: str, strategy_name: str):
    """Pilotage et état du journal des paris (cf. app.bet_log): c'est lui qui
    transforme les signaux affichés ci-dessus en un relevé de performance réelle,
    exploité par l'onglet 💰 Bankroll."""
    with st.expander("🧾 Journal des paris de la stratégie (onglet 💰 Bankroll)", expanded=bool(n_logged)):
        st.checkbox(
            "Enregistrer les signaux comme paris flat", value=True, key="upc_log_bets",
            help="Un pari par match et par stratégie, au premier signal — les consultations "
                 "suivantes ne réécrivent pas la cote enregistrée.",
        )
        st.caption(
            "Chaque opportunité signalée ci-dessus est enregistrée comme un pari à mise flat de "
            "1 unité, à la cote disponible au moment du signal, sur un match pas encore joué. "
            "Une fois le match dans `data/tennis.db`, le pari est réglé automatiquement et vient "
            "alimenter la courbe de bankroll de l'onglet 💰 Bankroll. C'est la seule mesure de "
            "performance **hors échantillon** : le ROI du backtest, lui, vient d'un jeu de test "
            "déjà exploré des centaines de fois par la recherche de règle."
        )

        if not strategy_id:
            st.info(
                "Aucune stratégie sélectionnée ci-dessus (un modèle seul 🤖 ne produit pas de "
                "signal) — rien n'est enregistré."
            )
            return

        bets = bet_log.list_bets(strategy_id)
        if n_logged:
            st.success(f"➕ {n_logged} nouveau(x) pari(s) enregistré(s) à l'instant pour **{strategy_name}**.")
        if bets.empty:
            st.info("Aucun pari enregistré pour cette stratégie pour l'instant.")
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Paris enregistrés", len(bets))
        m2.metric("En attente", int((bets["status"] == bet_log.PENDING).sum()))
        m3.metric("Réglés", int((bets["status"] != bet_log.PENDING).sum()))


def _render_price_log_panel(n_logged: int):
    """Pilotage et état du journal des prix (cf. app.odds_log): c'est lui qui
    construit, jour après jour, l'historique Polymarket dont on ne dispose pas
    et sans lequel aucun signal Polymarket-only ne peut être validé."""
    with st.expander("📝 Journal des prix de marché (constitution de l'historique Polymarket)"):
        st.checkbox(
            "Enregistrer les prix des matchs affichés", value=True, key="upc_log_prices",
            help="Un instantané par match et par heure de consultation, au maximum.",
        )
        st.caption(
            "Aucun historique Polymarket n'est disponible publiquement, alors que l'historique "
            "bookmaker couvre des années (data/tennis.db). C'est ce qui empêche aujourd'hui de "
            "valider un signal sur les matchs que **seul** Polymarket couvre : ils sont absents "
            "du backtest, qui ne retient que les matchs cotés par un bookmaker. Ce journal "
            "enregistre à chaque consultation le prix Polymarket **et** la cote bookmaker du "
            "moment, pour reconstituer cet historique. Une fois les résultats connus, les matchs "
            "couverts par les deux sources permettent de mesurer directement l'écart entre "
            "Polymarket et le consensus bookmaker dévigorisé, et les matchs Polymarket-only "
            "deviennent enfin backtestables."
        )

        s = odds_log.stats()
        if not s["n_snapshots"]:
            st.info("Journal vide pour l'instant — il se remplira à chaque consultation de cet onglet.")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Instantanés", f"{s['n_snapshots']:,}".replace(",", " "))
        m2.metric("Matchs suivis", s["n_matches"])
        m3.metric("Polymarket + bookmaker", s["n_both"], help="Comparables entre eux — calibration du biais.")
        m4.metric("Polymarket seul", s["n_pm_only"], help="La population aujourd'hui non backtestable.")
        if n_logged:
            st.caption(f"➕ {n_logged} instantané(s) ajouté(s) à l'instant.")
        if s["since"]:
            st.caption(f"Collecte démarrée le {s['since'][:10]}.")

        snaps = odds_log.load_snapshots()
        st.download_button(
            "⬇️ Exporter le journal (CSV)", snaps.to_csv(index=False).encode("utf-8"),
            file_name="odds_log.csv", mime="text/csv",
        )
