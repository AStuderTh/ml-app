"""Simulation de stratégie de paris à partir de probabilités modèle vs cotes
du marché ("value betting"): on ne mise que quand le modèle estime une
probabilité de victoire supérieure à celle impliquée par la cote, au-delà
d'un seuil de marge (edge)."""
import numpy as np
import pandas as pd

DEFAULT_STRATEGY = {
    "edge_threshold": 0.03,
    "stake_mode": "flat",       # "flat" ou "kelly"
    "flat_stake": 1.0,
    "kelly_fraction": 0.25,     # fraction du Kelly plein (gestion du risque)
    "bankroll": 100.0,
    "min_odds": None,           # None = pas de borne
    "max_odds": None,
}


def run_backtest(probs_p1: np.ndarray, test_df: pd.DataFrame, strategy: dict) -> tuple[pd.DataFrame, dict]:
    strat = {**DEFAULT_STRATEGY, **(strategy or {})}
    odds_p1 = test_df["odds_p1"].values
    odds_p2 = test_df["odds_p2"].values
    label = test_df["label"].values
    dates = test_df["tourney_date"].values

    implied_p1 = 1.0 / odds_p1
    implied_p2 = 1.0 / odds_p2
    model_p1 = probs_p1
    model_p2 = 1.0 - probs_p1
    edge_p1 = model_p1 - implied_p1
    edge_p2 = model_p2 - implied_p2

    # Filtre optionnel de cote: exclut les grosses faveurs (edge mécaniquement
    # gonflé par une petite erreur de calibration près de p=1) et/ou les gros
    # outsiders (variance élevée pour un edge nominal identique), indépendamment
    # du seuil d'edge lui-même.
    min_odds, max_odds = strat["min_odds"], strat["max_odds"]

    def _odds_ok(odds):
        ok = np.ones(len(odds), dtype=bool)
        if min_odds is not None:
            ok &= odds >= min_odds
        if max_odds is not None:
            ok &= odds <= max_odds
        return ok

    bet_p1 = (edge_p1 >= strat["edge_threshold"]) & (edge_p1 >= edge_p2) & _odds_ok(odds_p1)
    bet_p2 = (~bet_p1) & (edge_p2 >= strat["edge_threshold"]) & _odds_ok(odds_p2)

    # Absent des jeux de test allégés enregistrés avant l'ajout du découpage par
    # segment (cf. BACKTEST_COLS): on dégrade au lieu de lever, sinon rejouer un
    # modèle sauvegardé de l'ancienne génération casse.
    levels = (test_df["tourney_level"].values if "tourney_level" in test_df.columns
              else np.full(len(test_df), None))

    rows = []
    bankroll = strat["bankroll"]
    for i in range(len(test_df)):
        if bet_p1[i]:
            side, odds, p, won = "p1", odds_p1[i], model_p1[i], label[i] == 1
        elif bet_p2[i]:
            side, odds, p, won = "p2", odds_p2[i], model_p2[i], label[i] == 0
        else:
            continue

        if strat["stake_mode"] == "kelly":
            b = odds - 1.0
            f = ((p * b - (1 - p)) / b) if b > 0 else 0.0
            f = max(0.0, min(f * strat["kelly_fraction"], 1.0))
            stake = bankroll * f
        else:
            stake = strat["flat_stake"]

        if stake <= 0:
            continue

        profit = stake * (odds - 1.0) if won else -stake
        bankroll += profit
        rows.append({
            "date": dates[i], "match_id": test_df["match_id"].values[i],
            "player": test_df["p1_name"].values[i] if side == "p1" else test_df["p2_name"].values[i],
            "opponent": test_df["p2_name"].values[i] if side == "p1" else test_df["p1_name"].values[i],
            "side": side, "odds": odds, "model_prob": p, "stake": stake,
            "won": bool(won), "profit": profit, "bankroll_after": bankroll,
            "surface": test_df["surface"].values[i],
            "tourney_level": levels[i],
        })

    bets_df = pd.DataFrame(rows)
    if bets_df.empty:
        metrics = dict(n_bets=0, total_staked=0.0, total_profit=0.0, roi=0.0,
                        win_rate=0.0, final_bankroll=strat["bankroll"], max_drawdown=0.0)
        return bets_df, metrics

    total_staked = bets_df["stake"].sum()
    total_profit = bets_df["profit"].sum()
    running_max = bets_df["bankroll_after"].cummax()
    drawdown = (running_max - bets_df["bankroll_after"]) / running_max.replace(0, np.nan)
    metrics = dict(
        n_bets=len(bets_df),
        total_staked=float(total_staked),
        total_profit=float(total_profit),
        roi=float(total_profit / total_staked) if total_staked > 0 else 0.0,
        win_rate=float(bets_df["won"].mean()),
        final_bankroll=float(bets_df["bankroll_after"].iloc[-1]),
        max_drawdown=float(drawdown.max(skipna=True) or 0.0),
    )
    return bets_df, metrics


# Niveaux servis par un fournisseur de cotes AVANT match (cf. app/odds.py:
# The Odds API ne couvre que les Grand Chelems et les Masters/500 principaux).
# Un ROI porté par les niveaux absents de cet ensemble n'est pas jouable en
# production, quelle que soit sa solidité statistique en backtest.
LIVE_ODDS_LEVELS = {"G", "M"}


def segment_metrics(bets_df: pd.DataFrame, by: str = "tourney_level") -> pd.DataFrame:
    """ROI décomposé par segment de marché.

    Le ROI global mélange des marchés d'efficience très différente: une finale
    de Grand Chelem absorbe des millions et une vingtaine de modèles
    professionnels, un 1er tour d'ATP 250 absorbe quelques milliers d'euros et
    une ligne largement automatisée. L'erreur du marché n'est pas répartie
    uniformément, donc un ROI global n'est pas une quantité interprétable: il
    faut savoir QUEL segment le porte, et si ce segment est pariable.
    """
    if bets_df is None or bets_df.empty or by not in bets_df.columns:
        return pd.DataFrame()

    g = bets_df.groupby(by, dropna=False)
    out = pd.DataFrame({
        "n_bets": g.size(),
        "staked": g["stake"].sum(),
        "profit": g["profit"].sum(),
        "win_rate": g["won"].mean(),
        "avg_odds": g["odds"].mean(),
    })
    out["roi"] = out["profit"] / out["staked"].replace(0, np.nan)

    # Erreur-type EMPIRIQUE du ROI, et non une approximation analytique: les
    # mises varient (Kelly) et les cotes sont très dispersées, donc l'écart-type
    # réalisé des profits est le seul estimateur honnête. Sans cette colonne, un
    # ROI de +8% sur 40 paris se lit comme un signal alors que son incertitude
    # est de l'ordre de ±25 points.
    avg_stake = out["staked"] / out["n_bets"]
    out["roi_se"] = g["profit"].std() / np.sqrt(out["n_bets"]) / avg_stake.replace(0, np.nan)

    # |ROI| > 2 erreurs-types: seuil grossier, mais il suffit à écarter les
    # segments où le signe du ROI est indiscernable du hasard.
    out["significatif"] = out["roi"].abs() > 2 * out["roi_se"]

    if by == "tourney_level":
        out["pariable_live"] = [lvl in LIVE_ODDS_LEVELS for lvl in out.index]

    return out.sort_values("n_bets", ascending=False)
