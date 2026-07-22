"""Onglet 'Prochains matchs': calendrier ATP en direct (source: API publique
ESPN, site.api.espn.com — pas de clé requise) pour les tournois en cours ou
à venir, filtré sur les matchs simples pas encore joués. Un match peut être
sélectionné dans la liste pour afficher ses stats et une prédiction d'un
modèle ML sauvegardé."""
import datetime as dt

import pandas as pd
import requests
import streamlit as st

from app import store
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


def _fetch_week(date: dt.date) -> dict:
    resp = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": date.strftime("%Y%m%d")},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=1800, show_spinner="Récupération du calendrier ATP (ESPN)...")
def get_upcoming_matches(days_ahead: int = 21) -> pd.DataFrame:
    """Interroge le scoreboard ESPN tous les 3 jours sur la fenêtre demandée
    (chaque requête renvoie toute la semaine de tournoi active à cette date;
    un pas de 3 jours garantit qu'aucune semaine de tournoi, qui dure ~7-9
    jours, ne soit sautée entre deux requêtes) et retourne les matchs simples
    pas encore joués (status 'pre'), dédupliqués par id de match ESPN."""
    today = dt.date.today()
    rows = {}
    for offset in range(0, days_ahead + 1, 3):
        try:
            data = _fetch_week(today + dt.timedelta(days=offset))
        except Exception:
            continue
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
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows.values())
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    df["date_locale"] = df["date_utc"].dt.tz_convert(DISPLAY_TZ)
    return df.sort_values("date_utc").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _guess_surface(tourney_name: str, dataset: pd.DataFrame) -> str:
    """Devine la surface d'un tournoi à venir en cherchant la surface du
    match le plus récent joué sous un nom de tournoi équivalent dans
    l'historique (gère les changements de sponsor via canonicalize_tourney)."""
    from scripts.consolidate.normalize import canonicalize_tourney, norm_tourney_name

    target = canonicalize_tourney(norm_tourney_name(tourney_name))
    hist_names = dataset["tourney_name"].dropna().unique()
    matching = [n for n in hist_names if canonicalize_tourney(norm_tourney_name(n)) == target]
    if not matching:
        return "Hard"
    sub = dataset[dataset["tourney_name"].isin(matching)].sort_values("tourney_date")
    surf = sub["surface"].dropna()
    return surf.iloc[-1] if not surf.empty else "Hard"


def _render_match_detail(match: dict, dataset: pd.DataFrame):
    st.divider()
    st.markdown(f"### 🔍 {match['joueur_1']} vs {match['joueur_2']}")
    st.caption(f"{match['tournoi']} — {match['round']} — {match['date_locale'].strftime('%a %d/%m %H:%M')}")

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

    surface = match.get("surface") or _guess_surface(match["tournoi"], dataset)
    snap1 = player_snapshot(state, canon1, surface)
    snap2 = player_snapshot(state, canon2, surface)

    st.caption(
        f"Surface supposée : **{surface}** (déduite de l'historique du tournoi). "
        f"Noms retrouvés dans la base : **{canon1}** vs **{canon2}**."
    )

    stat_rows = [
        ("Elo global", f"{snap1['elo']:.0f}", f"{snap2['elo']:.0f}"),
        ("Elo global (pondéré récence)", f"{snap1['elo_recent']:.0f}", f"{snap2['elo_recent']:.0f}"),
        (f"Elo {surface}", f"{snap1['elo_surface']:.0f}", f"{snap2['elo_surface']:.0f}"),
        (f"Elo {surface} (pondéré récence)", f"{snap1['elo_surface_recent']:.0f}", f"{snap2['elo_surface_recent']:.0f}"),
        ("Classement ATP", snap1["rank"], snap2["rank"]),
        ("Points ATP", snap1["rank_points"], snap2["rank_points"]),
        ("Âge (dernier connu)", snap1["age"], snap2["age"]),
        ("Taille (cm)", snap1["ht"], snap2["ht"]),
        ("Main", snap1["hand"], snap2["hand"]),
        ("Forme (10 derniers)", f"{snap1['form']*100:.0f}%", f"{snap2['form']*100:.0f}%"),
        ("Matchs joués (historique)", snap1["played"], snap2["played"]),
        ("Dernier match connu", snap1["last_match_date"], snap2["last_match_date"]),
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
    saved = store.list_models()
    if saved.empty:
        st.info("Aucun modèle sauvegardé. Entraîne et sauvegarde un modèle dans l'onglet 'Construire un modèle'.")
    else:
        labels = [f"{row['name']} ({row['algo']}, AUC {row['auc']:.3f})" for _, row in saved.iterrows()]
        idx = st.selectbox(
            "Modèle", range(len(labels)), format_func=lambda i: labels[i], key="upc_model_select",
        )
        model_row = saved.iloc[idx]

        # implied_prob_p1 utilisé comme FEATURE d'entrée du modèle (si le modèle
        # l'utilise): priorité au bookmaker (cohérent avec les cotes utilisées à
        # l'entraînement), Polymarket en repli.
        implied_prob_p1 = None
        if comparison:
            implied_prob_p1 = comparison[0]["p1"]

        import json as _json
        features = _json.loads(model_row["features_json"])
        values, missing = build_live_features(snap1, snap2, state, canon1, canon2, features, surface, implied_prob_p1)

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
    provider_label = c4.selectbox(
        "Fournisseur de cotes bookmaker (en plus de Polymarket)",
        list(ODDS_PROVIDERS.keys()), key="upc_odds_provider",
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

    st.caption(f"{len(df)} matchs à venir sur la période sélectionnée")

    def _fmt_decimal(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    def _fmt_pm(v):
        if pd.isna(v) or v <= 0:
            return "—"
        return f"{v*100:.1f}% ({1/v:.2f})"

    selected_match = None

    for tourney in tournois:
        sub = df[df["tournoi"] == tourney].reset_index(drop=True)
        if sub.empty:
            continue
        sub["surface"] = _guess_surface(tourney, dataset) if dataset is not None else None
        with st.expander(f"🏆 {tourney} ({len(sub)} matchs)", expanded=len(sub) <= 20):
            display = sub.copy()
            display["Date"] = display["date_locale"].dt.strftime("%a %d/%m %H:%M")
            display["surface"] = display["surface"].fillna("?")
            cols = ["Date", "round", "joueur_1", "joueur_2", "surface", "lieu"]
            rename = {"round": "Round", "joueur_1": "Joueur 1", "joueur_2": "Joueur 2",
                      "surface": "Surface", "lieu": "Lieu"}

            # Polymarket: toujours affiché
            display["pm_j1"] = display["pm_prob_j1"].apply(_fmt_pm)
            display["pm_j2"] = display["pm_prob_j2"].apply(_fmt_pm)
            cols += ["pm_j1", "pm_j2"]
            rename.update({"pm_j1": "Polymarket J1", "pm_j2": "Polymarket J2"})

            if provider == "the_odds_api":
                display["cote_j1"] = display["cote_j1"].apply(_fmt_decimal)
                display["cote_j2"] = display["cote_j2"].apply(_fmt_decimal)
                cols += ["cote_j1", "cote_j2"]
                rename.update({"cote_j1": "Cote J1 (TOA)", "cote_j2": "Cote J2 (TOA)"})
            elif provider == "odds_api_io":
                display["oio_j1"] = display["oio_j1"].apply(_fmt_decimal)
                display["oio_j2"] = display["oio_j2"].apply(_fmt_decimal)
                cols += ["oio_j1", "oio_j2"]
                rename.update({"oio_j1": "Cote J1 (odds.io)", "oio_j2": "Cote J2 (odds.io)"})
            display = display[cols].rename(columns=rename)

            event = st.dataframe(
                display, width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row", key=f"upc_table_{tourney}",
            )
            rows = event.selection.rows if event and event.selection else []
            if rows:
                selected_match = sub.iloc[rows[0]].to_dict()

    if selected_match is not None:
        _render_match_detail(selected_match, dataset)
