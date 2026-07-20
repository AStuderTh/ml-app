"""Persistance des modèles sauvegardés: métadonnées + métriques dans une
base SQLite dédiée (séparée de data/tennis.db), modèle entraîné en joblib,
journal des paris du backtest en parquet (pour ré-afficher les graphs sans
ré-entraîner)."""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "data", "models")
DB_PATH = os.path.join(ROOT, "data", "ml_models.db")

os.makedirs(MODELS_DIR, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS saved_models (
    id TEXT PRIMARY KEY,
    name TEXT,
    algo TEXT,
    params_json TEXT,
    features_json TEXT,
    strategy_json TEXT,
    train_start TEXT, train_end TEXT, test_start TEXT, test_end TEXT,
    n_train INTEGER, n_test INTEGER,
    auc REAL, accuracy REAL, logloss REAL, brier REAL,
    n_bets INTEGER, roi REAL, total_profit REAL, win_rate REAL,
    final_bankroll REAL, max_drawdown REAL,
    created_at TEXT,
    model_path TEXT, bets_path TEXT
)
"""


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    return con


def save_model(result: dict, name: str, train_start, train_end, test_start, test_end) -> str:
    model_id = uuid.uuid4().hex[:12]
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    bets_path = os.path.join(MODELS_DIR, f"{model_id}_bets.parquet")

    joblib.dump(result["model"], model_path)
    bets_df = result["bets_df"]
    if bets_df is not None and not bets_df.empty:
        bets_df.to_parquet(bets_path, index=False)
    else:
        bets_path = ""

    m, bt = result["metrics"], result["backtest_metrics"]
    con = _connect()
    try:
        con.execute(
            """INSERT INTO saved_models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, name, result["algo"],
                json.dumps(result["params"]), json.dumps(result["features"]), json.dumps(result["strategy"]),
                str(train_start), str(train_end), str(test_start), str(test_end),
                int(m["n_train"]), int(m["n_test"]),
                m["auc"], m["accuracy"], m["logloss"], m["brier"],
                int(bt["n_bets"]), bt["roi"], bt["total_profit"], bt["win_rate"],
                bt["final_bankroll"], bt["max_drawdown"],
                datetime.now(timezone.utc).isoformat(),
                model_path, bets_path,
            ),
        )
        con.commit()
    finally:
        con.close()
    return model_id


def list_models() -> pd.DataFrame:
    con = _connect()
    try:
        df = pd.read_sql("SELECT * FROM saved_models ORDER BY created_at DESC", con)
    finally:
        con.close()
    return df


def load_model_bets(model_id: str) -> pd.DataFrame:
    row = list_models()
    row = row[row.id == model_id]
    if row.empty or not row.iloc[0]["bets_path"]:
        return pd.DataFrame()
    return pd.read_parquet(row.iloc[0]["bets_path"])


def load_model_object(model_id: str):
    row = list_models()
    row = row[row.id == model_id]
    if row.empty:
        return None
    return joblib.load(row.iloc[0]["model_path"])


def delete_model(model_id: str):
    con = _connect()
    try:
        row = con.execute("SELECT model_path, bets_path FROM saved_models WHERE id=?", (model_id,)).fetchone()
        con.execute("DELETE FROM saved_models WHERE id=?", (model_id,))
        con.commit()
    finally:
        con.close()
    if row:
        for p in row:
            if p and os.path.exists(p):
                os.remove(p)
