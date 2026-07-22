"""Onglet 'Prochains matchs': calendrier ATP en direct (source: API publique
ESPN, site.api.espn.com — pas de clé requise) pour les tournois en cours ou
à venir, filtré sur les matchs simples pas encore joués."""
import datetime as dt

import pandas as pd
import requests
import streamlit as st

from app.odds import attach_odds, sport_key_for_tournament

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


def render_upcoming():
    st.subheader("📅 Prochains matchs (ATP)")
    st.caption(
        "Calendrier en direct via l'API publique ESPN — tournois ATP en cours ou à venir. "
        "Heures affichées en Europe/Paris. Le tableau d'un tournoi n'est publié par l'ATP que "
        "quelques jours avant son début : au-delà, les matchs existent déjà (bon nombre de "
        "'Round 1', 'Round 2'...) mais les joueurs restent en 'TBD' tant que le tirage au sort "
        "n'est pas sorti."
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
    c4, c5 = st.columns([1, 3])
    show_odds = c4.checkbox("💰 Récupérer les cotes", value=False, key="upc_show_odds")
    c5.caption(
        "Cotes via The Odds API — couverture limitée aux Grand Chelems / Masters / ATP 500, "
        "et seulement une fois le marché ouvert (quelques jours avant le tournoi). "
        f"Tournoi(s) sélectionné(s) couvert(s) par ce fournisseur : {', '.join(sorted(covered)) or 'aucun'}."
    )

    quota_info = None
    if show_odds:
        df, quota_info = attach_odds(df)
        if quota_info == "no_key":
            st.warning("Aucune clé ODDS_API_KEY configurée dans .streamlit/secrets.toml.")
        elif quota_info is not None and not str(quota_info).isdigit():
            st.warning(f"Erreur The Odds API : {quota_info}")
        elif quota_info is not None:
            st.caption(f"Requêtes The Odds API restantes ce mois-ci : {quota_info}")

    st.caption(f"{len(df)} matchs à venir sur la période sélectionnée")

    for tourney in tournois:
        sub = df[df["tournoi"] == tourney]
        if sub.empty:
            continue
        with st.expander(f"🏆 {tourney} ({len(sub)} matchs)", expanded=len(sub) <= 20):
            display = sub.copy()
            display["Date"] = display["date_locale"].dt.strftime("%a %d/%m %H:%M")
            cols = ["Date", "round", "joueur_1", "joueur_2", "lieu"]
            rename = {"round": "Round", "joueur_1": "Joueur 1", "joueur_2": "Joueur 2", "lieu": "Lieu"}
            if show_odds and tourney in covered:
                display["cote_j1"] = display["cote_j1"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
                display["cote_j2"] = display["cote_j2"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
                cols += ["cote_j1", "cote_j2"]
                rename.update({"cote_j1": "Cote J1", "cote_j2": "Cote J2"})
            display = display[cols].rename(columns=rename)
            st.dataframe(display, width="stretch", hide_index=True)
