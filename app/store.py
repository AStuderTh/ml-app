"""Persistance des modèles sauvegardés: métadonnées + métriques dans une
base SQLite dédiée (séparée de data/tennis.db), modèle entraîné en joblib,
journal des paris du backtest en parquet (pour ré-afficher les graphs sans
ré-entraîner)."""
import json
import os
import shutil
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

# Colonnes ajoutées après la création initiale de la table: nécessaires pour
# reconstruire le jeu de test complet d'un modèle sauvegardé (toutes les
# lignes, pas seulement les paris placés à l'époque) et rejouer son backtest
# avec d'autres stratégies sans réentraîner (cf. app.roi_bruteforce). Ajout
# via migration idempotente: les modèles sauvegardés avant cet ajout auront
# ces colonnes à NULL, l'UI leur demande alors la source de cotes à la main.
_MIGRATIONS = [
    "ALTER TABLE saved_models ADD COLUMN odds_w_col TEXT",
    "ALTER TABLE saved_models ADD COLUMN odds_l_col TEXT",
    # ece calculé dans train.py après la conception initiale de cette table — nécessaire
    # pour que le score de qualité (app.scoring.compute_quality_score) soit complet une
    # fois le modèle rechargé depuis la base plutôt que depuis un résultat frais.
    "ALTER TABLE saved_models ADD COLUMN ece REAL",
]

# Règles de sélection (seuil d'edge, mode de mise) sauvegardées, associées à un
# modèle (cf. app.roi_bruteforce et la simulation ROI manuelle) — table séparée
# de saved_models: un même modèle peut avoir plusieurs règles testées/gardées.
# Table nommée selection_rules (anciennement roi_strategies — renommée pour
# rester cohérente avec le terme "règle de sélection" utilisé dans l'UI, cf.
# _rename_legacy_tables ci-dessous pour la migration des bases existantes).
SELECTION_RULE_SCHEMA = """
CREATE TABLE IF NOT EXISTS selection_rules (
    id TEXT PRIMARY KEY,
    model_id TEXT,
    name TEXT,
    strategy_json TEXT,
    n_bets INTEGER, roi REAL, total_profit REAL, win_rate REAL,
    final_bankroll REAL, max_drawdown REAL,
    roi_score REAL,
    created_at TEXT,
    bets_path TEXT
)
"""

# "Stratégies" (onglet 🎯 Stratégies): association explicite modèle + règle de
# sélection, créée à la main par l'utilisateur (contrairement à selection_rules,
# cette table est vide tant que l'utilisateur n'a rien ajouté lui-même).
STRATEGY_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    name TEXT,
    model_id TEXT,
    rule_id TEXT,
    created_at TEXT
)
"""


def _rename_legacy_tables(con):
    """Renomme roi_strategies -> selection_rules sur une base créée avant ce
    renommage — fait AVANT la création des tables pour ne pas dupliquer les
    anciennes données sous un nom vide. Les id des lignes ne changent pas, donc
    `strategies.rule_id` (qui référence ces id) reste valide sans autre migration."""
    existing = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "roi_strategies" in existing and "selection_rules" not in existing:
        con.execute("ALTER TABLE roi_strategies RENAME TO selection_rules")
        # sans commit explicite ici, le rename reste dans une transaction non validée
        # et disparaît silencieusement à la fermeture de la connexion
        con.commit()


def _connect():
    con = sqlite3.connect(DB_PATH)
    _rename_legacy_tables(con)
    con.execute(SCHEMA)
    con.execute(SELECTION_RULE_SCHEMA)
    con.execute(STRATEGY_SCHEMA)
    for migration in _MIGRATIONS:
        try:
            con.execute(migration)
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    return con


def save_model(result: dict, name: str, train_start, train_end, test_start, test_end,
               odds_w_col: str = None, odds_l_col: str = None) -> str:
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
            """INSERT INTO saved_models (
                id, name, algo, params_json, features_json, strategy_json,
                train_start, train_end, test_start, test_end,
                n_train, n_test, auc, accuracy, logloss, brier,
                n_bets, roi, total_profit, win_rate, final_bankroll, max_drawdown,
                created_at, model_path, bets_path, odds_w_col, odds_l_col, ece
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, name, result["algo"],
                json.dumps(result["params"]), json.dumps(result["features"]), json.dumps(result["strategy"]),
                str(train_start), str(train_end), str(test_start), str(test_end),
                int(m["n_train"]), int(m["n_test"]),
                m["auc"], m["accuracy"], m["logloss"], m["brier"],
                int(bt["n_bets"]), bt["roi"], bt["total_profit"], bt["win_rate"],
                bt["final_bankroll"], bt["max_drawdown"],
                datetime.now(timezone.utc).isoformat(),
                model_path, bets_path, odds_w_col, odds_l_col, m.get("ece"),
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
    model_path = row.iloc[0]["model_path"]
    if not model_path or not os.path.isfile(model_path):
        return None
    return joblib.load(model_path)


def load_test_frame_with_probs(model_id: str, dataset: pd.DataFrame, odds_w_col: str = None,
                                odds_l_col: str = None):
    """Recharge un modèle sauvegardé, reconstruit son jeu de test COMPLET
    (toutes les lignes de la période de test, pas seulement les paris placés
    à l'époque) et recalcule les probabilités prédites — pour pouvoir rejouer
    le backtest avec d'autres stratégies (cf. app.roi_bruteforce) sans
    réentraîner. `dataset` doit être le DataFrame déjà enrichi des features
    chronologiques (app.features.compute_chronological_features), fourni par
    l'appelant pour éviter de le recalculer ici (coûteux).

    `odds_w_col`/`odds_l_col` permettent de forcer la source de cotes utilisée
    à l'entraînement (nécessaire pour les modèles sauvegardés avant l'ajout de
    ces colonnes, où elles valent NULL en base). Renvoie None si la source de
    cotes est introuvable/non fournie, ou si le modèle/les features sont
    incompatibles avec `dataset`."""
    from .features import build_training_frame  # import tardif: évite un cycle avec features -> ... -> store

    row = list_models()
    row = row[row.id == model_id]
    if row.empty:
        return None
    row = row.iloc[0]

    odds_w_col = odds_w_col or row.get("odds_w_col")
    odds_l_col = odds_l_col or row.get("odds_l_col")
    if not odds_w_col or not odds_l_col or pd.isna(odds_w_col) or pd.isna(odds_l_col):
        return None

    model = load_model_object(model_id)
    if model is None:
        return None
    features = json.loads(row["features_json"])

    full = build_training_frame(dataset, odds_w_col, odds_l_col,
                                 min_date=row["train_start"], max_date=row["test_end"])
    test_frame = full[full["tourney_date"] >= pd.Timestamp(row["test_start"])].dropna(subset=features)
    if test_frame.empty:
        return None

    probs = model.predict_proba(test_frame[features].values)[:, 1]
    return test_frame.assign(model_prob_p1=probs), features


def rename_model(model_id: str, new_name: str):
    con = _connect()
    try:
        con.execute("UPDATE saved_models SET name=? WHERE id=?", (new_name, model_id))
        con.commit()
    finally:
        con.close()


def duplicate_model(model_id: str, new_name: str) -> str:
    """Copie un modèle sauvegardé (fichiers joblib/parquet inclus, pas
    seulement la ligne en base) sous un nouvel id — un point de départ pour
    l'éditer sans toucher à l'original. Les règles de sélection associées à
    l'original ne sont PAS dupliquées (elles resteront liées à l'original)."""
    row = list_models()
    row = row[row.id == model_id]
    if row.empty:
        return None
    row = row.iloc[0]

    new_id = uuid.uuid4().hex[:12]
    new_model_path = os.path.join(MODELS_DIR, f"{new_id}.joblib")
    shutil.copyfile(row["model_path"], new_model_path)
    new_bets_path = ""
    if row["bets_path"] and os.path.exists(row["bets_path"]):
        new_bets_path = os.path.join(MODELS_DIR, f"{new_id}_bets.parquet")
        shutil.copyfile(row["bets_path"], new_bets_path)

    con = _connect()
    try:
        con.execute(
            """INSERT INTO saved_models (
                id, name, algo, params_json, features_json, strategy_json,
                train_start, train_end, test_start, test_end,
                n_train, n_test, auc, accuracy, logloss, brier,
                n_bets, roi, total_profit, win_rate, final_bankroll, max_drawdown,
                created_at, model_path, bets_path, odds_w_col, odds_l_col, ece
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                new_id, new_name, row["algo"], row["params_json"], row["features_json"], row["strategy_json"],
                row["train_start"], row["train_end"], row["test_start"], row["test_end"],
                int(row["n_train"]), int(row["n_test"]), row["auc"], row["accuracy"], row["logloss"], row["brier"],
                int(row["n_bets"]), row["roi"], row["total_profit"], row["win_rate"],
                row["final_bankroll"], row["max_drawdown"],
                datetime.now(timezone.utc).isoformat(), new_model_path, new_bets_path,
                row["odds_w_col"], row["odds_l_col"], row.get("ece"),
            ),
        )
        con.commit()
    finally:
        con.close()
    return new_id


def delete_model(model_id: str):
    """Supprime le modèle ET, en cascade, ses règles de sélection sauvegardées
    ainsi que les stratégies (onglet 🎯 Stratégies) qui le référencent (une
    règle/stratégie n'a aucun sens sans le modèle sur lequel elle a été
    mesurée)."""
    con = _connect()
    try:
        row = con.execute("SELECT model_path, bets_path FROM saved_models WHERE id=?", (model_id,)).fetchone()
        strat_bets_paths = con.execute(
            "SELECT bets_path FROM selection_rules WHERE model_id=?", (model_id,)
        ).fetchall()
        con.execute("DELETE FROM saved_models WHERE id=?", (model_id,))
        con.execute("DELETE FROM selection_rules WHERE model_id=?", (model_id,))
        con.execute("DELETE FROM strategies WHERE model_id=?", (model_id,))
        con.commit()
    finally:
        con.close()
    if row:
        for p in row:
            if p and os.path.exists(p):
                os.remove(p)
    for (bp,) in strat_bets_paths:
        if bp and os.path.exists(bp):
            os.remove(bp)


def save_selection_rule(model_id: str, name: str, strategy: dict, bt: dict, roi_score: float,
                         bets_df: pd.DataFrame = None) -> str:
    """Sauvegarde une règle de sélection (issue du bruteforce ROI ou de la
    simulation manuelle) associée à un modèle déjà sauvegardé."""
    rule_id = uuid.uuid4().hex[:12]
    bets_path = ""
    if bets_df is not None and not bets_df.empty:
        bets_path = os.path.join(MODELS_DIR, f"rule_{rule_id}_bets.parquet")
        bets_df.to_parquet(bets_path, index=False)

    con = _connect()
    try:
        con.execute(
            """INSERT INTO selection_rules (
                id, model_id, name, strategy_json, n_bets, roi, total_profit, win_rate,
                final_bankroll, max_drawdown, roi_score, created_at, bets_path
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rule_id, model_id, name, json.dumps(strategy),
                int(bt["n_bets"]), bt["roi"], bt["total_profit"], bt["win_rate"],
                bt["final_bankroll"], bt["max_drawdown"], roi_score,
                datetime.now(timezone.utc).isoformat(), bets_path,
            ),
        )
        con.commit()
    finally:
        con.close()
    return rule_id


def list_selection_rules(model_id: str = None) -> pd.DataFrame:
    con = _connect()
    try:
        if model_id:
            df = pd.read_sql(
                "SELECT * FROM selection_rules WHERE model_id=? ORDER BY created_at DESC", con, params=(model_id,)
            )
        else:
            df = pd.read_sql("SELECT * FROM selection_rules ORDER BY created_at DESC", con)
    finally:
        con.close()
    return df


def load_selection_rule_bets(rule_id: str) -> pd.DataFrame:
    df = list_selection_rules()
    row = df[df.id == rule_id]
    if row.empty or not row.iloc[0]["bets_path"]:
        return pd.DataFrame()
    return pd.read_parquet(row.iloc[0]["bets_path"])


def delete_selection_rule(rule_id: str):
    """Supprime la règle ET, en cascade, les stratégies (onglet 🎯 Stratégies)
    qui la référencent (une stratégie n'a aucun sens sans sa règle)."""
    con = _connect()
    try:
        row = con.execute("SELECT bets_path FROM selection_rules WHERE id=?", (rule_id,)).fetchone()
        con.execute("DELETE FROM selection_rules WHERE id=?", (rule_id,))
        con.execute("DELETE FROM strategies WHERE rule_id=?", (rule_id,))
        con.commit()
    finally:
        con.close()
    if row and row[0] and os.path.exists(row[0]):
        os.remove(row[0])


def rename_selection_rule(rule_id: str, new_name: str):
    con = _connect()
    try:
        con.execute("UPDATE selection_rules SET name=? WHERE id=?", (new_name, rule_id))
        con.commit()
    finally:
        con.close()


def update_selection_rule(rule_id: str, strategy: dict, bt: dict, roi_score: float,
                           bets_df: pd.DataFrame = None):
    """Remplace la config de mise et les métriques d'une règle de sélection déjà
    sauvegardée (même id, donc conserve les stratégies qui la référencent) —
    utilisé par le bouton 'Éditer' de l'onglet 🎯 Stratégies."""
    existing = list_selection_rules()
    existing = existing[existing.id == rule_id]
    bets_path = existing.iloc[0]["bets_path"] if not existing.empty else ""
    if bets_df is not None and not bets_df.empty:
        bets_path = bets_path or os.path.join(MODELS_DIR, f"rule_{rule_id}_bets.parquet")
        bets_df.to_parquet(bets_path, index=False)

    con = _connect()
    try:
        con.execute(
            """UPDATE selection_rules SET
                strategy_json=?, n_bets=?, roi=?, total_profit=?, win_rate=?,
                final_bankroll=?, max_drawdown=?, roi_score=?, bets_path=?
               WHERE id=?""",
            (
                json.dumps(strategy), int(bt["n_bets"]), bt["roi"], bt["total_profit"], bt["win_rate"],
                bt["final_bankroll"], bt["max_drawdown"], roi_score, bets_path, rule_id,
            ),
        )
        con.commit()
    finally:
        con.close()


def create_strategy(name: str, model_id: str, rule_id: str) -> str:
    """Crée une 'stratégie' (onglet 🎯 Stratégies): association explicite entre
    un modèle sauvegardé et une de ses règles de sélection sauvegardées.
    Contrairement aux règles elles-mêmes (bruteforce/simulation manuelle,
    table selection_rules), cette table est vide par défaut — c'est
    l'utilisateur qui crée chaque association à la main."""
    strategy_id = uuid.uuid4().hex[:12]
    con = _connect()
    try:
        con.execute(
            "INSERT INTO strategies (id, name, model_id, rule_id, created_at) VALUES (?,?,?,?,?)",
            (strategy_id, name, model_id, rule_id, datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
    finally:
        con.close()
    return strategy_id


def list_strategies() -> pd.DataFrame:
    """Liste les stratégies créées par l'utilisateur, avec les caractéristiques
    du modèle et de la règle de sélection associés jointes — pour l'onglet
    🎯 Stratégies."""
    con = _connect()
    try:
        df = pd.read_sql(
            """
            SELECT
                st.id AS id, st.name AS name, st.created_at,
                m.id AS model_id, m.name AS model_name, m.algo,
                m.auc, m.logloss, m.brier, m.ece,
                r.id AS rule_id, r.name AS rule_name, r.strategy_json,
                r.n_bets, r.roi, r.total_profit, r.win_rate, r.final_bankroll, r.max_drawdown, r.roi_score
            FROM strategies st
            JOIN saved_models m ON m.id = st.model_id
            JOIN selection_rules r ON r.id = st.rule_id
            ORDER BY st.created_at DESC
            """,
            con,
        )
    finally:
        con.close()
    return df


def rename_strategy(strategy_id: str, new_name: str):
    con = _connect()
    try:
        con.execute("UPDATE strategies SET name=? WHERE id=?", (new_name, strategy_id))
        con.commit()
    finally:
        con.close()


def update_strategy(strategy_id: str, model_id: str, rule_id: str):
    """Change le modèle et/ou la règle de sélection pointés par cette
    stratégie (bouton 'Éditer' de l'onglet 🎯 Stratégies)."""
    con = _connect()
    try:
        con.execute("UPDATE strategies SET model_id=?, rule_id=? WHERE id=?", (model_id, rule_id, strategy_id))
        con.commit()
    finally:
        con.close()


def delete_strategy(strategy_id: str):
    con = _connect()
    try:
        con.execute("DELETE FROM strategies WHERE id=?", (strategy_id,))
        con.commit()
    finally:
        con.close()
