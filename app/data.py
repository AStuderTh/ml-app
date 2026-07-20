"""Chargement des matchs depuis data/tennis.db."""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "tennis.db")

MATCH_COLUMNS = [
    "match_id", "tourney_date", "tourney_name", "surface", "tourney_level", "round", "best_of",
    "winner_name", "winner_hand", "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
    "loser_name", "loser_hand", "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
    "score", "minutes", "match_confidence",
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
    numeric_cols = [
        "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
        "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
        "best_of", "minutes",
        "odds_w_b365", "odds_l_b365", "odds_w_ps", "odds_l_ps",
        "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df


ODDS_CHOICES = {
    "Moyenne du marché (Avg)": ("odds_w_avg", "odds_l_avg"),
    "Bet365": ("odds_w_b365", "odds_l_b365"),
    "Pinnacle": ("odds_w_ps", "odds_l_ps"),
    "Meilleure cote (Max)": ("odds_w_max", "odds_l_max"),
}
