"""Cotes des matchs à venir via The Odds API (the-odds-api.com, clé requise,
stockée dans .streamlit/secrets.toml sous ODDS_API_KEY — jamais committée).

Couverture limitée par le fournisseur: seulement les Grand Chelems et
tournois Masters/500 principaux, et seulement dans les jours qui précèdent
le tournoi (le marché n'existe pas encore avant). Les ATP 250 et tournois
plus loin dans le calendrier n'auront donc pas de cotes — c'est une
limite du fournisseur de données, pas un bug de l'app."""
import os

import pandas as pd
import requests
import streamlit as st

from app.name_utils import normalize_name as _normalize_name

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sous-chaîne (normalisée, minuscule) trouvée dans le nom de tournoi ESPN ->
# clé de sport The Odds API. Couverture du fournisseur: Grand Chelems +
# principaux Masters/500 ATP uniquement.
TOURNAMENT_KEY_MAP = {
    "australian open": "tennis_atp_aus_open_singles",
    "french open": "tennis_atp_french_open",
    "roland garros": "tennis_atp_french_open",
    "wimbledon": "tennis_atp_wimbledon",
    "us open": "tennis_atp_us_open",
    "indian wells": "tennis_atp_indian_wells",
    "miami open": "tennis_atp_miami_open",
    "monte-carlo masters": "tennis_atp_monte_carlo_masters",
    "monte carlo masters": "tennis_atp_monte_carlo_masters",
    "madrid open": "tennis_atp_madrid_open",
    "italian open": "tennis_atp_italian_open",
    "internazionali": "tennis_atp_italian_open",
    "national bank open": "tennis_atp_canadian_open",
    "canadian open": "tennis_atp_canadian_open",
    "cincinnati open": "tennis_atp_cincinnati_open",
    "shanghai masters": "tennis_atp_shanghai_masters",
    "paris masters": "tennis_atp_paris_masters",
    "dubai": "tennis_atp_dubai",
    "barcelona open": "tennis_atp_barcelona_open",
    "halle open": "tennis_atp_halle_open",
    "hamburg open": "tennis_atp_hamburg_open",
    "munich": "tennis_atp_munich",
    "qatar open": "tennis_atp_qatar_open",
    "queen's club": "tennis_atp_queens_club_champ",
}


def _get_api_key():
    try:
        key = st.secrets.get("ODDS_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("ODDS_API_KEY")


def sport_key_for_tournament(tourney_name: str) -> str | None:
    name = (tourney_name or "").lower()
    for substr, key in TOURNAMENT_KEY_MAP.items():
        if substr in name:
            return key
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_odds(sport_key: str):
    api_key = _get_api_key()
    if not api_key:
        return None, "no_key"
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
            params={"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json(), resp.headers.get("x-requests-remaining")
    except Exception as e:
        return None, str(e)


def _best_odds_for_player(bookmakers, player_norm):
    """Meilleure cote (max toutes bookmakers 'eu' confondues) pour un joueur,
    identifié par correspondance exacte du nom normalisé, ou à défaut par
    son nom de famille (dernier mot)."""
    player_surname = player_norm.split()[-1] if player_norm else ""
    best_exact, best_surname = None, None
    for bk in bookmakers or []:
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                oname = _normalize_name(outcome.get("name", ""))
                price = outcome.get("price")
                if price is None:
                    continue
                if oname == player_norm:
                    best_exact = price if best_exact is None else max(best_exact, price)
                elif oname.split()[-1] == player_surname and player_surname:
                    best_surname = price if best_surname is None else max(best_surname, price)
    return best_exact if best_exact is not None else best_surname


def attach_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Ajoute des colonnes cote_j1/cote_j2 (meilleure cote dispo, marché eu)
    au dataframe de matchs à venir, en matchant tournoi + horaire + noms.
    Retourne (df enrichi, quota restant ou message d'erreur/absence de clé)."""
    df = df.copy()
    df["cote_j1"] = pd.NA
    df["cote_j2"] = pd.NA

    if df.empty:
        return df, None

    quota_info = None
    for tourney in df["tournoi"].unique():
        sport_key = sport_key_for_tournament(tourney)
        if sport_key is None:
            continue
        events, info = _fetch_odds(sport_key)
        if info is not None:
            quota_info = info  # quota restant, ou message d'erreur / "no_key"
        if not events:
            continue

        idx = df.index[df["tournoi"] == tourney]
        for i in idx:
            row = df.loc[i]
            n1, n2 = _normalize_name(row["joueur_1"]), _normalize_name(row["joueur_2"])
            if n1 == "tbd" or n2 == "tbd":
                continue
            match_dt = row["date_utc"]
            best_event, best_diff = None, None
            for ev in events:
                ev_time = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
                if pd.isna(ev_time):
                    continue
                diff = abs((ev_time - match_dt).total_seconds())
                if diff > 6 * 3600:  # tolérance: 6h autour de l'horaire ESPN
                    continue
                ev_names = {_normalize_name(ev.get("home_team", "")), _normalize_name(ev.get("away_team", ""))}
                if not ({n1, n2} & ev_names):
                    # match approché sur le nom de famille si pas de correspondance exacte
                    surnames_row = {n1.split()[-1] if n1 else "", n2.split()[-1] if n2 else ""}
                    surnames_ev = {n.split()[-1] if n else "" for n in ev_names}
                    if not (surnames_row & surnames_ev):
                        continue
                if best_diff is None or diff < best_diff:
                    best_event, best_diff = ev, diff
            if best_event is None:
                continue
            bookmakers = best_event.get("bookmakers")
            df.loc[i, "cote_j1"] = _best_odds_for_player(bookmakers, n1)
            df.loc[i, "cote_j2"] = _best_odds_for_player(bookmakers, n2)

    return df, quota_info
