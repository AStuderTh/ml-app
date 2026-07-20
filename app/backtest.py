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

    bet_p1 = (edge_p1 >= strat["edge_threshold"]) & (edge_p1 >= edge_p2)
    bet_p2 = (~bet_p1) & (edge_p2 >= strat["edge_threshold"])

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
