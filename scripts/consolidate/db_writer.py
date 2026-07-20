"""Écrit le résultat consolidé dans un fichier SQLite (recréé à chaque
exécution) et mentionne aussi la file de validation manuelle dans une table
`match_review` (miroir informatif du CSV, qui reste la source éditable)."""
import os
import sqlite3

import pandas as pd

MATCHES_SCHEMA_HINTS = {
    "tourney_date": "TEXT",
    "has_pending_review": "INTEGER",
}

NUMERIC_COLUMNS = [
    "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
    "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
    "best_of", "minutes", "draw_size",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "odds_w_b365", "odds_l_b365", "odds_w_ps", "odds_l_ps", "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg",
]


def write_sqlite(db_path: str, matches_df: pd.DataFrame, review_df: pd.DataFrame):
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    try:
        df = matches_df.copy()
        df["tourney_date"] = pd.to_datetime(df["tourney_date"]).dt.strftime("%Y-%m-%d")
        # certaines colonnes numériques finissent en dtype 'object' après la
        # fusion des 3 sources (mélange int/float/None selon la ligne), ce
        # qui les ferait écrire en TEXT dans sqlite: on force le typage ici.
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_sql("matches", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX idx_matches_match_id ON matches(match_id)")
        con.execute("CREATE INDEX idx_matches_date ON matches(tourney_date)")
        con.execute("CREATE INDEX idx_matches_winner ON matches(winner_name)")
        con.execute("CREATE INDEX idx_matches_loser ON matches(loser_name)")
        con.execute("CREATE INDEX idx_matches_confidence ON matches(match_confidence)")

        review_df.to_sql("match_review", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
