"""Chargement des matchs depuis data/tennis.db."""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "tennis.db")

MATCH_COLUMNS = [
    "match_id", "tourney_date", "match_date", "tourney_name", "surface", "indoor",
    "tourney_level", "round", "best_of",
    "winner_name", "winner_player_id", "winner_hand", "winner_ht", "winner_age",
    "winner_rank", "winner_rank_points",
    "loser_name", "loser_player_id", "loser_hand", "loser_ht", "loser_age",
    "loser_rank", "loser_rank_points",
    # classements officiels ATP repris du référentiel hebdomadaire Sackmann:
    # meilleure couverture que les colonnes embarquées ci-dessus, et la valeur
    # à 52 semaines permet une trajectoire (cf. scripts/consolidate/rankings.py)
    "winner_atp_rank", "winner_atp_points", "winner_atp_rank_prev", "winner_atp_points_prev",
    "loser_atp_rank", "loser_atp_points", "loser_atp_rank_prev", "loser_atp_points_prev",
    # Points de service: seule base permettant de mesurer la compétence sur
    # ~150 points par match plutôt que sur un unique résultat binaire (dont
    # dérivent Elo et forme). Les stats de retour s'en déduisent: les points de
    # retour gagnés par un joueur sont les points de service perdus par l'autre.
    "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon",
    "score", "minutes", "match_confidence", "match_comment",
    "odds_w_b365", "odds_l_b365", "odds_w_ps", "odds_l_ps",
    "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg",
]


def load_matches() -> pd.DataFrame:
    """Charge tout l'historique (utilisé pour calculer Elo/forme/H2H sur la
    profondeur maximale), trié chronologiquement."""
    con = sqlite3.connect(DB_PATH)
    try:
        cols = ", ".join(MATCH_COLUMNS)
        df = pd.read_sql(f"SELECT {cols} FROM matches WHERE winner_name IS NOT NULL AND loser_name IS NOT NULL", con)
    finally:
        con.close()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    numeric_cols = [
        "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
        "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
        "winner_player_id", "loser_player_id",
        "winner_atp_rank", "winner_atp_points", "winner_atp_rank_prev", "winner_atp_points_prev",
        "loser_atp_rank", "loser_atp_points", "loser_atp_rank_prev", "loser_atp_points_prev",
        "best_of", "minutes",
        "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon",
        "odds_w_b365", "odds_l_b365", "odds_w_ps", "odds_l_ps",
        "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # kind="stable" (le quicksort par défaut ne l'est pas): tous les matchs
    # d'un même tournoi partagent la date de début du tournoi, seul l'ordre
    # des lignes de la base les départage — et il y est déjà chronologique
    # (tri par tour effectué à la consolidation). Un tri non stable le
    # détruirait, replaçant potentiellement une finale avant un 1er tour.
    df = df.sort_values("tourney_date", kind="stable").reset_index(drop=True)
    return df


ODDS_CHOICES = {
    "Moyenne du marché (Avg)": ("odds_w_avg", "odds_l_avg"),
    "Bet365": ("odds_w_b365", "odds_l_b365"),
    "Pinnacle": ("odds_w_ps", "odds_l_ps"),
    "Meilleure cote (Max)": ("odds_w_max", "odds_l_max"),
}
