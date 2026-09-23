"""Bruteforce dédié à la STRATÉGIE DE MISE: à partir d'un modèle déjà
entraîné et figé (ses probabilités prédites ne changent pas d'un essai à
l'autre), teste plein de combinaisons (edge_threshold, stake_mode,
kelly_fraction) pour trouver la meilleure exploitation possible — séparé du
bruteforce de modèles (app.bruteforce), qui ne compare que la qualité
prédictive indépendamment de toute stratégie."""
import numpy as np

from .backtest import run_backtest

STAKE_MODES = ["flat", "kelly"]


def sample_strategy(rng: np.random.Generator) -> dict:
    stake_mode = STAKE_MODES[rng.integers(0, len(STAKE_MODES))]

    # Bornes de cote tirées séparément, avec 50% de chance chacune d'être
    # désactivée (None): le bruteforce doit pouvoir retomber sur une stratégie
    # sans filtre de cote aussi bien que sur une stratégie bornée.
    min_odds = float(rng.uniform(1.01, 3.0)) if rng.random() < 0.5 else None
    max_odds = float(rng.uniform(3.0, 15.0)) if rng.random() < 0.5 else None
    if min_odds is not None and max_odds is not None and min_odds >= max_odds:
        min_odds, max_odds = max_odds, min_odds

    return {
        "edge_threshold": float(rng.uniform(0.0, 0.15)),
        "stake_mode": stake_mode,
        "flat_stake": 1.0,
        # None en mode flat: run_backtest ne lit kelly_fraction qu'en mode kelly, donc
        # tirer une valeur ici serait un bruit trompeur (affiché comme "None" dans le
        # tableau de résultats, mais présent dans le JSON sauvegardé si on le tirait quand même).
        "kelly_fraction": float(rng.uniform(0.05, 1.0)) if stake_mode == "kelly" else None,
        "bankroll": 100.0,
        "min_odds": min_odds,
        "max_odds": max_odds,
    }


def run_roi_bruteforce(probs, test_frame, n_strategies: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n_strategies):
        strategy = sample_strategy(rng)
        bets_df, bt_metrics = run_backtest(probs, test_frame, strategy)
        yield i, strategy, bt_metrics, bets_df
