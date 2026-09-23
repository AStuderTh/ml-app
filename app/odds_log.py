"""Journal des prix de marché observés sur les matchs à venir.

Raison d'être: on ne dispose d'aucun historique Polymarket, alors qu'on a des
années de cotes bookmaker (data/tennis.db). Impossible, donc, de calibrer ou
de valider une règle de sélection contre les prix Polymarket — en particulier
sur les matchs que SEUL Polymarket couvre (Challengers, petits tournois), qui
sont justement absents du jeu de backtest: app.features.build_training_frame
fait un dropna sur les colonnes de cotes bookmaker, donc 100% des matchs
validés sont des matchs couverts par un bookmaker.

Ce module enregistre, à chaque consultation de l'onglet 'Prochains matchs',
le prix Polymarket ET la cote bookmaker du moment pour chaque match affiché.
Accumulé sur quelques mois puis recoupé avec les résultats (resolve_snapshots),
il produit exactement le jeu de données manquant:
  - paires (prix Polymarket, cote bookmaker) sur les matchs couverts par les
    deux -> mesure directe du biais/bruit de Polymarket vs consensus bookmaker
    dévigorisé, au lieu de l'extrapoler depuis l'overround;
  - matchs Polymarket-only avec leur résultat -> permet enfin de backtester la
    population de paris qu'on s'interdit aujourd'hui faute de données.

Base séparée (data/odds_log.db) de data/ml_models.db: donnée opérationnelle
accumulée en continu, sans rapport avec les métadonnées de modèles.
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

from app import match_resolve
from app.name_utils import normalize_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "odds_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT,
    bucket TEXT,
    match_key TEXT,
    tournoi TEXT,
    round TEXT,
    date_utc TEXT,
    surface TEXT,
    joueur_1 TEXT,
    joueur_2 TEXT,
    pm_prob_j1 REAL,
    pm_prob_j2 REAL,
    book_odds_j1 REAL,
    book_odds_j2 REAL,
    book_source TEXT,
    model_id TEXT,
    model_prob_j1 REAL
)
"""

# Un rerun Streamlit se produit à chaque interaction avec un widget: sans
# dédoublonnage, la table grossirait de plusieurs lignes par seconde de
# consultation. On ne garde donc qu'un instantané par match et par heure
# (INSERT OR IGNORE sur cet index) — borné à 24 lignes/match/jour, tout en
# conservant la trajectoire du prix à l'approche du match, qui est elle-même
# une information utile (le prix le plus proche du coup d'envoi est le plus
# informatif, et c'est celui auquel on miserait réellement).
INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_unique
ON odds_snapshots (match_key, bucket)
"""


def _connect():
    """À utiliser via `with closing(_connect()) as con:` — `with con:` seul gère
    la transaction mais NE ferme PAS la connexion (comportement sqlite3), et
    chaque rerun Streamlit en ouvrirait alors une de plus sans jamais la
    libérer. Ajouter `, con` derrière le closing pour valider en écriture."""
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    con.execute(INDEX)
    return con


def _match_key(row) -> str | None:
    """Identifiant stable d'un match, indépendant de l'ordre des joueurs (ESPN
    peut inverser j1/j2 d'une requête à l'autre) et de l'heure exacte, qui est
    régulièrement réajustée par ESPN au fil des jours. L'orientation j1/j2 reste
    récupérable via les colonnes joueur_1/joueur_2 stockées à côté."""
    n1, n2 = normalize_name(row["joueur_1"]), normalize_name(row["joueur_2"])
    if not n1 or not n2 or "tbd" in (n1, n2):
        return None
    day = pd.to_datetime(row["date_utc"], utc=True, errors="coerce")
    if pd.isna(day):
        return None
    return f"{day.strftime('%Y-%m-%d')}|" + "|".join(sorted([n1, n2]))


def _num(value):
    """Convertit en float natif, ou None — sqlite3 ne sait pas lier un pd.NA
    ni un numpy.float64 NaN de façon exploitable."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def log_snapshots(sub: pd.DataFrame, signal_odds_fn, model_id: str = None) -> int:
    """Enregistre un instantané des prix pour chaque match de `sub`.

    `signal_odds_fn(row) -> (odds_j1, odds_j2, source)` est injecté par
    l'appelant (app.upcoming._signal_odds_from_row) pour que l'ordre de
    priorité des bookmakers reste défini à un seul endroit.

    Les matchs sans aucun prix (ni Polymarket ni bookmaker) sont ignorés: une
    ligne vide n'apprendrait rien. Les matchs Polymarket-only sont en revanche
    bien enregistrés — ce sont eux qui manquent le plus.

    Retourne le nombre de lignes réellement insérées (0 si tous les matchs ont
    déjà un instantané dans l'heure en cours)."""
    if sub is None or sub.empty:
        return 0

    now = datetime.now(timezone.utc)
    bucket = now.strftime("%Y-%m-%dT%H")
    captured_at = now.isoformat(timespec="seconds")

    rows = []
    for _, row in sub.iterrows():
        key = _match_key(row)
        if key is None:
            continue
        pm1, pm2 = _num(row.get("pm_prob_j1")), _num(row.get("pm_prob_j2"))
        o1, o2, source = signal_odds_fn(row)
        if pm1 is None and o1 is None:
            continue
        date_utc = pd.to_datetime(row["date_utc"], utc=True, errors="coerce")
        rows.append((
            captured_at, bucket, key,
            row.get("tournoi"), row.get("round"),
            None if pd.isna(date_utc) else date_utc.isoformat(),
            row.get("surface"),
            row.get("joueur_1"), row.get("joueur_2"),
            pm1, pm2, _num(o1), _num(o2), source,
            model_id, _num(row.get("model_prob_j1")),
        ))

    if not rows:
        return 0

    with closing(_connect()) as con, con:
        cur = con.executemany(
            """INSERT OR IGNORE INTO odds_snapshots
               (captured_at, bucket, match_key, tournoi, round, date_utc, surface,
                joueur_1, joueur_2, pm_prob_j1, pm_prob_j2, book_odds_j1, book_odds_j2,
                book_source, model_id, model_prob_j1)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        return cur.rowcount


def load_snapshots() -> pd.DataFrame:
    with closing(_connect()) as con:
        return pd.read_sql("SELECT * FROM odds_snapshots ORDER BY captured_at", con)


def stats() -> dict:
    """Compteurs pour l'UI: volume accumulé et, surtout, répartition entre les
    matchs couverts par les deux sources (comparables entre eux) et les matchs
    Polymarket-only (la population qu'on ne sait pas encore backtester).

    La classification se fait sur TOUTE la durée de vie du match, pas sur un
    instantané isolé: un bookmaker ouvre souvent son marché plus tard que
    Polymarket, donc un match vu sans cote à J-10 en aura une à J-1. Le
    compter comme 'Polymarket seul' sur la foi du premier instantané
    surestimerait largement cette population — seuls comptent ici les matchs
    qui n'ont JAMAIS eu de cote bookmaker. Les deux catégories sont ainsi
    mutuellement exclusives."""
    with closing(_connect()) as con:
        row = con.execute(
            """SELECT COUNT(*), SUM(has_pm AND has_book), SUM(has_pm AND NOT has_book)
               FROM (SELECT MAX(pm_prob_j1 IS NOT NULL) AS has_pm,
                            MAX(book_odds_j1 IS NOT NULL) AS has_book
                     FROM odds_snapshots GROUP BY match_key)"""
        ).fetchone()
        totals = con.execute("SELECT COUNT(*), MIN(captured_at) FROM odds_snapshots").fetchone()
    return {
        "n_snapshots": totals[0] or 0,
        "n_matches": row[0] or 0,
        "n_both": row[1] or 0,
        "n_pm_only": row[2] or 0,
        "since": totals[1],
    }


def resolve_snapshots(dataset: pd.DataFrame, max_day_gap: int = 3) -> pd.DataFrame:
    """Recoupe les instantanés avec les résultats connus pour produire le jeu
    de calibration: une ligne par match résolu, avec le DERNIER prix observé
    avant le coup d'envoi (le plus informatif, et celui auquel on aurait misé)
    et l'issue réelle.

    Le rapprochement se fait sur les noms de famille normalisés des deux joueurs
    et une tolérance de quelques jours sur la date, `tourney_date` de
    l'historique étant la date de DÉBUT du tournoi et non celle du match.

    Colonnes ajoutées: j1_won (1 si joueur_1 a gagné), pm_fair_j1 (= pm_prob_j1,
    déjà sans marge), book_fair_j1 (cote bookmaker dévigorisée, comparable au
    prix Polymarket) et book_overround."""
    snaps = load_snapshots()
    if snaps.empty or dataset is None or dataset.empty:
        return pd.DataFrame()

    snaps["date_utc"] = pd.to_datetime(snaps["date_utc"], utc=True, errors="coerce")
    snaps["captured_at"] = pd.to_datetime(snaps["captured_at"], utc=True, errors="coerce")
    # Dernier instantané pris AVANT le début du match (on ignore ceux capturés
    # après: le prix y intègre déjà, en partie, le déroulement de la rencontre).
    before = snaps[snaps["captured_at"] <= snaps["date_utc"]]
    latest = (before.sort_values("captured_at")
                    .groupby("match_key", as_index=False).last())
    if latest.empty:
        return pd.DataFrame()

    hist = match_resolve.prepare_history(dataset)

    out = []
    for _, snap in latest.iterrows():
        j1_won = match_resolve.winner_of(hist, snap["date_utc"], snap["joueur_1"],
                                          snap["joueur_2"], max_day_gap)
        if j1_won is None:
            continue
        row = snap.to_dict()
        row["j1_won"] = j1_won
        out.append(row)

    resolved = pd.DataFrame(out)
    if resolved.empty:
        return resolved

    resolved["pm_fair_j1"] = resolved["pm_prob_j1"]
    inv1 = 1.0 / resolved["book_odds_j1"]
    inv2 = 1.0 / resolved["book_odds_j2"]
    resolved["book_overround"] = inv1 + inv2
    resolved["book_fair_j1"] = inv1 / resolved["book_overround"]
    return resolved
