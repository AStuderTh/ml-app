"""Cotes des matchs à venir via Polymarket (Gamma API, publique, sans clé:
https://gamma-api.polymarket.com). Couvre tout le circuit ATP (tour
principal + Challengers), on ne garde donc que les correspondances trouvées
pour les matchs déjà présents dans notre calendrier (tour principal only).

Contrairement à un bookmaker, un marché Polymarket ne donne pas une "cote"
mais un prix de 0 à 1 = probabilité implicite que le joueur gagne. On
affiche cette probabilité directement (plus honnête qu'une fausse cote
décimale), et on la convertit aussi en cote décimale équivalente (1/proba)
pour comparaison visuelle avec les autres sources."""
import json

import pandas as pd
import requests
import streamlit as st

from app.name_utils import normalize_name

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
ATP_SERIES_ID = "10365"  # série "atp" chez Polymarket (tour principal + Challengers)
MAX_TIME_DIFF_SECONDS = 24 * 3600  # tolérance entre l'heure ESPN et l'heure Polymarket


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_atp_events(limit: int = 300):
    try:
        resp = requests.get(
            f"{GAMMA_API_BASE}/events",
            params={"series_id": ATP_SERIES_ID, "limit": limit, "closed": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def _match_winner_candidates(events):
    """Extrait, pour chaque événement, le marché 'qui gagne le match' (les 2
    outcomes = les 2 noms de joueurs, toujours le 1er marché de l'événement)."""
    candidates = []
    for ev in events or []:
        markets = ev.get("markets") or []
        if not markets:
            continue
        m0 = markets[0]
        # outcomes/outcomePrices sont des chaînes JSON imbriquées côté Gamma API
        # (ex: '["Joueur A", "Joueur B"]'), pas des listes natives — il faut les
        # re-décoder explicitement.
        try:
            outcomes = json.loads(m0.get("outcomes") or "[]")
            prices = [float(p) for p in json.loads(m0.get("outcomePrices") or "[]")]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(outcomes) != 2 or len(prices) != 2:
            continue
        candidates.append({
            "names_norm": [normalize_name(outcomes[0]), normalize_name(outcomes[1])],
            "prices": prices,
            "game_start": pd.to_datetime(m0.get("gameStartTime"), utc=True, errors="coerce"),
            "title": ev.get("title"),
        })
    return candidates


def attach_polymarket_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Ajoute pm_prob_j1/pm_prob_j2 (probabilité implicite 0-1) au dataframe
    de matchs à venir, en matchant sur les noms des 2 joueurs (+ tolérance
    de 24h sur l'heure) parmi tous les événements ATP actifs sur Polymarket.
    Retourne (df enrichi, message d'info/erreur)."""
    df = df.copy()
    df["pm_prob_j1"] = pd.NA
    df["pm_prob_j2"] = pd.NA

    if df.empty:
        return df, None

    events, error = _fetch_atp_events()
    if error:
        return df, f"Erreur Polymarket : {error}"
    candidates = _match_winner_candidates(events)
    if not candidates:
        return df, "Aucun événement ATP actif trouvé sur Polymarket."

    for i, row in df.iterrows():
        n1, n2 = normalize_name(row["joueur_1"]), normalize_name(row["joueur_2"])
        if n1 == "tbd" or n2 == "tbd":
            continue
        target = {n1, n2}
        target_surnames = {n.split()[-1] for n in target if n}

        best, best_diff = None, None
        for c in candidates:
            c_names = set(c["names_norm"])
            if c_names != target:
                c_surnames = {n.split()[-1] for n in c_names if n}
                if c_surnames != target_surnames:
                    continue
            if pd.notna(c["game_start"]) and pd.notna(row["date_utc"]):
                diff = abs((c["game_start"] - row["date_utc"]).total_seconds())
                if diff > MAX_TIME_DIFF_SECONDS:
                    continue
            else:
                diff = 0
            if best_diff is None or diff < best_diff:
                best, best_diff = c, diff

        if best is None:
            continue
        p1_first_outcome_is_j1 = best["names_norm"][0].split()[-1] == n1.split()[-1]
        p1, p2 = best["prices"] if p1_first_outcome_is_j1 else best["prices"][::-1]
        df.loc[i, "pm_prob_j1"] = p1
        df.loc[i, "pm_prob_j2"] = p2

    n_matched = int(df["pm_prob_j1"].notna().sum())
    return df, f"{len(candidates)} événements ATP récupérés sur Polymarket, {n_matched} match(s) trouvé(s)."
