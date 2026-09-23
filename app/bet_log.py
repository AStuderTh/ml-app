"""Journal des paris signalés par une stratégie sur les matchs à venir.

Chaque fois que l'onglet '📅 Prochains matchs' affiche un signal pour la
stratégie sélectionnée, ce signal est enregistré ici comme un pari à mise
FLAT (1 unité). C'est le pendant réel du backtest: le backtest mesure ce
qu'une stratégie AURAIT fait sur l'historique, ce journal enregistre ce
qu'elle a effectivement signalé, aux cotes réellement disponibles au moment du
signal, sur des matchs dont le résultat n'était pas encore connu. C'est la
seule mesure de performance non contaminée par le sur-ajustement de la
recherche de règle (le jeu de test, lui, a été vu des centaines de fois par le
bruteforce ROI).

Trois choix structurants:

- **Un seul pari par (stratégie, match), le premier signal fait foi.** Les
  cotes bougent, le modèle peut changer d'avis d'une consultation à l'autre;
  réécrire un pari déjà enregistré reviendrait à réécrire l'histoire et à
  choisir a posteriori le meilleur point d'entrée. La ligne enregistrée est
  donc immuable (INSERT OR IGNORE sur (strategy_id, match_key)).

- **Rien n'est enregistré sur un match déjà commencé.** Les cotes d'un match en
  cours intègrent déjà son déroulement: un pari pris dessus serait un pari
  gagné d'avance, pas un signal.

- **La mise stockée est la mise flat; la courbe Kelly est recalculée à
  l'affichage** à partir de la probabilité du modèle et de la cote (cf.
  `simulate`). Une mise Kelly dépend de la bankroll au moment du pari, donc de
  paramètres de simulation qu'on veut pouvoir changer après coup sans réécrire
  le journal.

Base séparée (data/bets.db) de data/ml_models.db, comme app.odds_log: donnée
opérationnelle qui s'accumule en continu. La stratégie est référencée par son
id ET par son nom dénormalisé — supprimer une stratégie du registre n'efface
pas son historique de paris, qui reste un relevé factuel.
"""
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import match_resolve
from app.backtest import DEFAULT_STRATEGY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "bets.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS placed_bets (
    id TEXT PRIMARY KEY,
    strategy_id TEXT,
    strategy_name TEXT,
    match_key TEXT,
    placed_at TEXT,
    match_date TEXT,
    tournoi TEXT,
    round TEXT,
    surface TEXT,
    joueur_1 TEXT,
    joueur_2 TEXT,
    bet_side TEXT,
    bet_player TEXT,
    opponent TEXT,
    model_prob REAL,
    edge REAL,
    odds REAL,
    odds_source TEXT,
    exec_odds REAL,
    exec_venue TEXT,
    flat_stake REAL,
    status TEXT,
    settled_at TEXT
)
"""

INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_bet_unique
ON placed_bets (strategy_id, match_key)
"""

PENDING, WON, LOST = "pending", "won", "lost"


def _connect():
    """À utiliser via `with closing(_connect()) as con:` — `with con:` seul gère
    la transaction mais NE ferme PAS la connexion (comportement sqlite3), et
    chaque rerun Streamlit en ouvrirait alors une de plus. Ajouter `, con`
    derrière le closing pour valider en écriture."""
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    con.execute(INDEX)
    return con


def _match_key(row) -> str | None:
    """Identifiant stable d'un match, indépendant de l'ordre des joueurs (ESPN
    peut inverser j1/j2 d'une requête à l'autre) et de l'heure exacte, souvent
    réajustée au fil des jours. Même clé que app.odds_log._match_key, pour que
    les deux journaux puissent être recoupés."""
    n1 = match_resolve.last_name(row["joueur_1"])
    n2 = match_resolve.last_name(row["joueur_2"])
    if not n1 or not n2 or "tbd" in (n1, n2):
        return None
    day = pd.to_datetime(row["date_utc"], utc=True, errors="coerce")
    if pd.isna(day):
        return None
    return f"{day.strftime('%Y-%m-%d')}|" + "|".join(sorted([n1, n2]))


def _num(value):
    """Convertit en float natif, ou None — sqlite3 ne sait lier ni un pd.NA ni
    un numpy.float64 NaN de façon exploitable."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def record_signals(sub: pd.DataFrame, strategy_id: str, strategy_name: str,
                   strategy: dict) -> int:
    """Enregistre comme paris flat tous les signaux de `sub` (DataFrame déjà
    passé par app.upcoming._attach_bet_signals) pour la stratégie donnée.

    Retourne le nombre de paris réellement ajoutés — 0 si tous les signaux
    affichés étaient déjà au journal, ce qui est le cas normal dès la deuxième
    consultation."""
    if sub is None or sub.empty or not strategy_id or "bet_side" not in sub.columns:
        return 0

    now = pd.Timestamp.now(tz="UTC")
    flat_stake = float({**DEFAULT_STRATEGY, **(strategy or {})}["flat_stake"])
    placed_at = now.isoformat(timespec="seconds")

    rows = []
    for _, row in sub.iterrows():
        side = row.get("bet_side")
        if side not in ("j1", "j2"):
            continue
        key = _match_key(row)
        if key is None:
            continue
        start = pd.to_datetime(row["date_utc"], utc=True, errors="coerce")
        if pd.isna(start) or start <= now:
            continue  # match déjà commencé: la cote n'est plus un prix d'avant-match

        model_p1 = _num(row.get("model_prob_j1"))
        if model_p1 is None:
            continue
        model_prob = model_p1 if side == "j1" else 1.0 - model_p1
        player = row["joueur_1"] if side == "j1" else row["joueur_2"]
        opponent = row["joueur_2"] if side == "j1" else row["joueur_1"]

        rows.append((
            uuid.uuid4().hex[:12], strategy_id, strategy_name, key, placed_at,
            start.isoformat(), row.get("tournoi"), row.get("round"), row.get("surface"),
            row.get("joueur_1"), row.get("joueur_2"),
            side, player, opponent,
            model_prob, _num(row.get("bet_edge")), _num(row.get("bet_odds")), row.get("bet_source"),
            _num(row.get("exec_odds")), row.get("exec_venue"),
            flat_stake, PENDING, None,
        ))

    if not rows:
        return 0

    with closing(_connect()) as con, con:
        cur = con.executemany(
            """INSERT OR IGNORE INTO placed_bets
               (id, strategy_id, strategy_name, match_key, placed_at, match_date,
                tournoi, round, surface, joueur_1, joueur_2, bet_side, bet_player,
                opponent, model_prob, edge, odds, odds_source, exec_odds, exec_venue,
                flat_stake, status, settled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        return cur.rowcount


def list_bets(strategy_id: str = None) -> pd.DataFrame:
    """Paris enregistrés, du plus ancien au plus récent (ordre de la simulation:
    la bankroll se construit dans l'ordre où les matchs se jouent)."""
    with closing(_connect()) as con:
        if strategy_id:
            df = pd.read_sql("SELECT * FROM placed_bets WHERE strategy_id=? ORDER BY match_date",
                             con, params=(strategy_id,))
        else:
            df = pd.read_sql("SELECT * FROM placed_bets ORDER BY match_date", con)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    return df


def journal_strategies() -> pd.DataFrame:
    """Stratégies présentes AU JOURNAL (id, nom, volume, dernier pari).

    Distincte de store.list_strategies(): une stratégie supprimée du registre
    garde son historique de paris ici, et doit rester consultable."""
    with closing(_connect()) as con:
        return pd.read_sql(
            """SELECT strategy_id, MAX(strategy_name) AS strategy_name,
                      COUNT(*) AS n_bets,
                      SUM(status != 'pending') AS n_settled,
                      MAX(match_date) AS last_bet
               FROM placed_bets GROUP BY strategy_id ORDER BY last_bet DESC""",
            con,
        )


def settle(dataset: pd.DataFrame, max_day_gap: int = 3) -> int:
    """Règle les paris en attente dont le match est retrouvé dans l'historique.

    Un pari reste 'pending' tant que le match n'est pas retrouvé: soit il n'a
    pas encore été joué, soit data/tennis.db n'a pas encore été mis à jour
    (onglet 🗄️ Données). Retourne le nombre de paris réglés à cet appel."""
    pending = list_bets()
    if pending.empty:
        return 0
    pending = pending[pending["status"] == PENDING]
    if pending.empty or dataset is None or dataset.empty:
        return 0

    hist = match_resolve.prepare_history(dataset)
    settled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    updates = []
    for _, bet in pending.iterrows():
        j1_won = match_resolve.winner_of(hist, bet["match_date"], bet["joueur_1"],
                                          bet["joueur_2"], max_day_gap)
        if j1_won is None:
            continue
        won = (j1_won == 1) if bet["bet_side"] == "j1" else (j1_won == 0)
        updates.append((WON if won else LOST, settled_at, bet["id"]))

    if not updates:
        return 0
    with closing(_connect()) as con, con:
        con.executemany("UPDATE placed_bets SET status=?, settled_at=? WHERE id=?", updates)
    return len(updates)


def delete_bet(bet_id: str):
    with closing(_connect()) as con, con:
        con.execute("DELETE FROM placed_bets WHERE id=?", (bet_id,))


def clear_strategy(strategy_id: str):
    """Efface tout l'historique de paris d'une stratégie — à ne proposer que
    derrière une confirmation: c'est le relevé de performance réelle."""
    with closing(_connect()) as con, con:
        con.execute("DELETE FROM placed_bets WHERE strategy_id=?", (strategy_id,))


def simulate(bets: pd.DataFrame, mode: str = "flat", bankroll: float = 100.0,
             flat_stake: float = 1.0, kelly_fraction: float = 0.25,
             odds_col: str = "exec_odds") -> tuple[pd.DataFrame, dict]:
    """Rejoue les paris RÉGLÉS dans l'ordre chronologique et renvoie la
    trajectoire de bankroll + les métriques, au même format que
    app.backtest.run_backtest (pour que l'UI réutilise render_backtest_metrics).

    Les paris en attente sont exclus: leur issue est inconnue, les inclure à
    profit nul ferait apparaître un palier plat trompeur en fin de courbe.

    `mode` vaut 'flat' (mise constante) ou 'kelly' (mise proportionnelle à
    l'avantage et à la bankroll courante). Le journal est enregistré en flat,
    mais Kelly reste simulable a posteriori: la décision de parier — la seule
    chose qui ne se rejoue pas — est identique dans les deux modes, seule la
    taille de mise change.

    `odds_col` choisit la cote de paiement: 'exec_odds' (meilleur marché trouvé,
    ce qu'on aurait réellement obtenu) ou 'odds' (cote bookmaker du signal,
    vue prudente et directement comparable au backtest)."""
    empty_metrics = dict(n_bets=0, total_staked=0.0, total_profit=0.0, roi=0.0,
                         win_rate=0.0, final_bankroll=float(bankroll), max_drawdown=0.0)
    if bets is None or bets.empty:
        return pd.DataFrame(), empty_metrics

    df = bets[bets["status"].isin([WON, LOST])].sort_values("match_date").copy()
    # Repli sur la cote du signal quand l'exécution n'a pas été enregistrée
    # (paris d'avant l'ajout de la colonne, ou match sans routage Polymarket).
    df["_odds"] = df[odds_col] if odds_col in df.columns else np.nan
    df["_odds"] = df["_odds"].fillna(df["odds"])
    df = df[df["_odds"] > 1]
    if df.empty:
        return pd.DataFrame(), empty_metrics

    current = float(bankroll)
    rows = []
    for _, bet in df.iterrows():
        odds = float(bet["_odds"])
        won = bet["status"] == WON
        if mode == "kelly":
            p = float(bet["model_prob"]) if pd.notna(bet["model_prob"]) else 0.0
            b = odds - 1.0
            f = ((p * b - (1 - p)) / b) if b > 0 else 0.0
            stake = current * max(0.0, min(f * (kelly_fraction or 0.0), 1.0))
        else:
            stake = float(flat_stake)
        if stake <= 0:
            continue
        profit = stake * (odds - 1.0) if won else -stake
        current += profit
        rows.append({
            "id": bet["id"],  # permet de rattacher la mise simulée à la ligne du journal
            "date": bet["match_date"], "tournoi": bet["tournoi"], "round": bet["round"],
            "surface": bet["surface"], "player": bet["bet_player"], "opponent": bet["opponent"],
            "model_prob": bet["model_prob"], "edge": bet["edge"],
            "odds": odds, "venue": bet["exec_venue"] or bet["odds_source"],
            "stake": stake, "won": won, "profit": profit, "bankroll_after": current,
        })

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve, empty_metrics

    total_staked = curve["stake"].sum()
    total_profit = curve["profit"].sum()
    running_max = curve["bankroll_after"].cummax()
    drawdown = (running_max - curve["bankroll_after"]) / running_max.replace(0, np.nan)
    metrics = dict(
        n_bets=len(curve),
        total_staked=float(total_staked),
        total_profit=float(total_profit),
        roi=float(total_profit / total_staked) if total_staked > 0 else 0.0,
        win_rate=float(curve["won"].mean()),
        final_bankroll=float(curve["bankroll_after"].iloc[-1]),
        max_drawdown=float(drawdown.max(skipna=True) or 0.0),
    )
    return curve, metrics
