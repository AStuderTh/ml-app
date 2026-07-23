"""Génère N modèles à hyperparamètres aléatoires, tous entraînés/backtestés
sur la même période, pour comparer leur QUALITÉ PRÉDICTIVE (cf. app.scoring.
compute_quality_score) — indépendamment de toute stratégie de mise.

La stratégie de backtest est FIXE (DEFAULT_STRATEGY) pour tous les modèles
générés ici, pas tirée au hasard: on ne cherche pas le meilleur ROI dans ce
module (cf. app.roi_bruteforce pour ça, une fois un bon modèle choisi), le
ROI/n_bets affichés ne servent qu'à titre informatif, sur une base de
comparaison commune."""
import numpy as np

from .backtest import DEFAULT_STRATEGY
from .features import FEATURE_POOL, expand_features
from .modeling import ALGOS, sample_random_params
from .train import train_and_evaluate


def sample_config(rng: np.random.Generator, algos_allowed, feature_pool=None, min_features=3):
    feature_pool = list(feature_pool) if feature_pool else list(FEATURE_POOL)
    algo = algos_allowed[rng.integers(0, len(algos_allowed))]
    params = sample_random_params(algo, rng)

    min_features = min(min_features, len(feature_pool))
    n_feat = int(rng.integers(min_features, len(feature_pool) + 1))
    features = expand_features(list(rng.choice(feature_pool, size=n_feat, replace=False)))

    return algo, params, features, dict(DEFAULT_STRATEGY)


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
