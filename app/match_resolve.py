"""Rapprochement d'un match annoncé (noms ESPN + date prévue) avec son
résultat dans l'historique une fois joué.

Deux modules en ont besoin et faisaient le même rapprochement chacun de leur
côté: app.odds_log (résolution du journal des prix) et app.bet_log (règlement
des paris enregistrés). Les deux partent d'une ligne ESPN et doivent retrouver
la rencontre dans data/tennis.db, où ni la date ni l'orthographe des noms ne
correspondent exactement:
  - `tourney_date` est la date de DÉBUT DU TOURNOI, pas celle du match, d'où
    la fenêtre de tolérance asymétrique (large en arrière, étroite en avant);
  - les noms viennent d'une autre source (accents, prénom complet vs initiale),
    d'où le rapprochement sur le seul nom de famille normalisé.
"""
import pandas as pd

from app.name_utils import normalize_name


def last_name(name: str) -> str:
    """Nom de famille normalisé (dernier mot), ou "" si le nom est inexploitable."""
    n = normalize_name(name)
    return n.split()[-1] if n else ""


def prepare_history(dataset: pd.DataFrame) -> pd.DataFrame:
    """Pré-calcule les colonnes de rapprochement sur l'historique complet.

    À faire UNE fois pour tout un lot de matchs à résoudre: normaliser les noms
    de dizaines de milliers de lignes à chaque appel de `winner_of` rendrait la
    résolution quadratique."""
    if dataset is None or dataset.empty:
        return pd.DataFrame(columns=["_w", "_l", "_date"])
    hist = dataset.dropna(subset=["winner_name", "loser_name", "tourney_date"]).copy()
    hist["_w"] = hist["winner_name"].map(last_name)
    hist["_l"] = hist["loser_name"].map(last_name)
    hist["_date"] = pd.to_datetime(hist["tourney_date"], errors="coerce")
    return hist


def winner_of(hist: pd.DataFrame, day, name_1: str, name_2: str,
              max_day_gap: int = 3) -> int | None:
    """1 si `name_1` a gagné, 0 si `name_2`, None si le match reste introuvable.

    `hist` doit venir de `prepare_history`. `day` est la date annoncée du match
    (tz-aware acceptée). Un None signifie "pas encore résolu": soit le match n'a
    pas encore été joué, soit l'historique n'a pas encore été mis à jour — dans
    les deux cas l'appelant doit réessayer plus tard, pas conclure à un échec."""
    if hist.empty:
        return None
    n1, n2 = last_name(name_1), last_name(name_2)
    day = pd.to_datetime(day, errors="coerce")
    if pd.isna(day) or not n1 or not n2 or n1 == n2:
        return None
    if day.tzinfo is not None:
        day = day.tz_localize(None)  # heure murale UTC — `hist["_date"]` est naïve

    window = hist[(hist["_date"] >= day - pd.Timedelta(days=max_day_gap + 14))
                  & (hist["_date"] <= day + pd.Timedelta(days=max_day_gap))]
    hit = window[((window["_w"] == n1) & (window["_l"] == n2))
                 | ((window["_w"] == n2) & (window["_l"] == n1))]
    if hit.empty:
        return None
    return int(hit.iloc[0]["_w"] == n1)
