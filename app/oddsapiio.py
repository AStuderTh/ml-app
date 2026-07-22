"""Cotes des matchs à venir via odds-api.io (api.odds-api.io, clé requise,
stockée dans .streamlit/secrets.toml sous ODDS_API_IO_KEY — jamais committée).

Couverture bien plus large que The Odds API: inclut les ATP 250 et même les
Challengers/ITF. Limite du plan gratuit: accès à seulement 2 bookmakers,
imposés par le fournisseur (pas de choix libre sans upgrade)."""
import pandas as pd
import requests
import streamlit as st

from app.name_utils import normalize_name

BASE = "https://api.odds-api.io/v3"
FREE_BOOKMAKERS = ["Bwin FR", "Winamax FR"]  # imposés par le plan gratuit


def _get_api_key():
    try:
        return st.secrets.get("ODDS_API_IO_KEY")
    except Exception:
        return None


def _surname(name_raw: str, name_norm: str) -> str:
    """odds-api.io donne les noms au format 'Nom, Prenom' (contrairement à
    ESPN qui donne 'Prenom Nom') — on extrait le nom de famille avant la
    virgule si présente, sinon on retombe sur le dernier mot normalisé."""
    if "," in str(name_raw):
        return normalize_name(str(name_raw).split(",")[0])
    return name_norm.split()[-1] if name_norm else ""


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_pending_events(limit: int = 300):
    api_key = _get_api_key()
    if not api_key:
        return None, "no_key"
    try:
        resp = requests.get(
            f"{BASE}/events",
            params={"apiKey": api_key, "sport": "tennis", "status": "pending", "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_event_odds(event_id):
    api_key = _get_api_key()
    if not api_key:
        return None, "no_key"
    try:
        resp = requests.get(
            f"{BASE}/odds",
            params={"apiKey": api_key, "eventId": event_id, "bookmakers": ",".join(FREE_BOOKMAKERS)},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return None, data["error"]
        return data, None
    except Exception as e:
        return None, str(e)


def attach_oddsapiio_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Ajoute oio_j1/oio_j2 (meilleure cote décimale parmi Bwin FR/Winamax FR)
    au dataframe de matchs à venir, en matchant sur les noms de famille des
    2 joueurs (+ tolérance 24h sur l'heure). Retourne (df enrichi, info)."""
    df = df.copy()
    df["oio_j1"] = pd.NA
    df["oio_j2"] = pd.NA

    if df.empty:
        return df, None

    events, error = _fetch_pending_events()
    if error == "no_key":
        return df, "no_key"
    if error:
        return df, f"Erreur odds-api.io : {error}"
    if not events:
        return df, "Aucun événement tennis en attente trouvé sur odds-api.io."

    candidates = []
    for ev in events:
        home_raw, away_raw = ev.get("home", ""), ev.get("away", "")
        home_norm, away_norm = normalize_name(home_raw), normalize_name(away_raw)
        candidates.append({
            "id": ev.get("id"),
            "home_raw": home_raw, "away_raw": away_raw,
            "home_surname": _surname(home_raw, home_norm),
            "away_surname": _surname(away_raw, away_norm),
            "date": pd.to_datetime(ev.get("date"), utc=True, errors="coerce"),
        })

    n_matched = 0
    for i, row in df.iterrows():
        n1, n2 = normalize_name(row["joueur_1"]), normalize_name(row["joueur_2"])
        if n1 == "tbd" or n2 == "tbd":
            continue
        target_surnames = {n1.split()[-1], n2.split()[-1]}

        best, best_diff = None, None
        for c in candidates:
            c_surnames = {c["home_surname"], c["away_surname"]}
            if c_surnames != target_surnames:
                continue
            if pd.notna(c["date"]) and pd.notna(row["date_utc"]):
                diff = abs((c["date"] - row["date_utc"]).total_seconds())
                if diff > 24 * 3600:
                    continue
            else:
                diff = 0
            if best_diff is None or diff < best_diff:
                best, best_diff = c, diff

        if best is None:
            continue
        odds_data, _ = _fetch_event_odds(best["id"])
        if not odds_data:
            continue

        j1_is_home = best["home_surname"] == n1.split()[-1]
        home_price, away_price = None, None
        for markets in (odds_data.get("bookmakers") or {}).values():
            for m in markets:
                if m.get("name") != "ML":
                    continue
                for o in m.get("odds", []):
                    try:
                        h, a = float(o["home"]), float(o["away"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    home_price = h if home_price is None else max(home_price, h)
                    away_price = a if away_price is None else max(away_price, a)
        if home_price is None:
            continue
        p1, p2 = (home_price, away_price) if j1_is_home else (away_price, home_price)
        df.loc[i, "oio_j1"] = p1
        df.loc[i, "oio_j2"] = p2
        n_matched += 1

    return df, f"{n_matched} match(s) avec cote(s) trouvé(s) (bookmakers: {', '.join(FREE_BOOKMAKERS)})."
