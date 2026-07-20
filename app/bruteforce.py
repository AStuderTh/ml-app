"""Génère N modèles à hyperparamètres/paramètres de stratégie aléatoires,
tous entraînés/backtestés sur la même période (pour que le classement par
ROI reste comparable d'un modèle à l'autre)."""
import numpy as np

from .features import FEATURE_POOL
from .modeling import ALGOS, sample_random_params
from .train import train_and_evaluate

STAKE_MODES = ["flat", "kelly"]


def sample_config(rng: np.random.Generator, algos_allowed, feature_pool=None, min_features=3):
    feature_pool = list(feature_pool) if feature_pool else list(FEATURE_POOL)
    algo = algos_allowed[rng.integers(0, len(algos_allowed))]
    params = sample_random_params(algo, rng)

    min_features = min(min_features, len(feature_pool))
    n_feat = int(rng.integers(min_features, len(feature_pool) + 1))
    features = list(rng.choice(feature_pool, size=n_feat, replace=False))

    strategy = {
        "edge_threshold": float(rng.uniform(0.0, 0.10)),
        "stake_mode": STAKE_MODES[rng.integers(0, len(STAKE_MODES))],
        "flat_stake": 1.0,
        "kelly_fraction": float(rng.uniform(0.1, 1.0)),
        "bankroll": 100.0,
    }
    return algo, params, features, strategy


def run_bruteforce(train_frame, test_frame, n_models: int, algos_allowed, seed: int = 0, feature_pool=None):
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        algo, params, features, strategy = sample_config(rng, algos_allowed, feature_pool=feature_pool)
        try:
            result = train_and_evaluate(train_frame, test_frame, algo, params, features, strategy)
        except Exception as exc:  # config invalide (ex: trop peu de features utiles) -> on passe
            yield i, None, {"error": str(exc), "algo": algo, "params": params, "features": features}
            continue
        yield i, result, None
