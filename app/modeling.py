"""Registre des algorithmes disponibles: classe sklearn, espace d'hyper-
paramètres (pour le tirage aléatoire du bruteforce) et widgets par défaut
(pour le constructeur manuel)."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _sample_int(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _sample_float(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _sample_choice(rng, choices):
    return choices[rng.integers(0, len(choices))]


ALGOS = {
    "logistic_regression": {
        "label": "Régression logistique",
        "cls": LogisticRegression,
        "needs_scaling": True,
        "fixed_params": {"max_iter": 2000},
        "sample_params": lambda rng: {
            "C": _sample_float(rng, 0.01, 10.0),
        },
        "param_widgets": {
            "C": ("float", 0.01, 10.0, 1.0),
        },
    },
    "decision_tree": {
        "label": "Arbre de décision",
        "cls": DecisionTreeClassifier,
        "needs_scaling": False,
        "fixed_params": {"random_state": 0},
        "sample_params": lambda rng: {
            "max_depth": _sample_int(rng, 2, 12),
            "min_samples_leaf": _sample_int(rng, 5, 200),
        },
        "param_widgets": {
            "max_depth": ("int", 2, 20, 5),
            "min_samples_leaf": ("int", 1, 500, 20),
        },
    },
    "random_forest": {
        "label": "Forêt aléatoire",
        "cls": RandomForestClassifier,
        "needs_scaling": False,
        "fixed_params": {"random_state": 0, "n_jobs": -1},
        "sample_params": lambda rng: {
            "n_estimators": _sample_int(rng, 50, 400),
            "max_depth": _sample_choice(rng, [None, 4, 6, 8, 12, 16]),
            "min_samples_leaf": _sample_int(rng, 5, 100),
        },
        "param_widgets": {
            "n_estimators": ("int", 10, 800, 200),
            "max_depth": ("int", 1, 30, 6),
            "min_samples_leaf": ("int", 1, 300, 20),
        },
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "cls": GradientBoostingClassifier,
        "needs_scaling": False,
        "fixed_params": {"random_state": 0},
        "sample_params": lambda rng: {
            "n_estimators": _sample_int(rng, 50, 300),
            "max_depth": _sample_int(rng, 1, 5),
            "learning_rate": _sample_float(rng, 0.01, 0.3),
            "min_samples_leaf": _sample_int(rng, 5, 100),
        },
        "param_widgets": {
            "n_estimators": ("int", 10, 600, 150),
            "max_depth": ("int", 1, 10, 3),
            "learning_rate": ("float", 0.001, 1.0, 0.1),
            "min_samples_leaf": ("int", 1, 300, 20),
        },
    },
    "knn": {
        "label": "K plus proches voisins",
        "cls": KNeighborsClassifier,
        "needs_scaling": True,
        "fixed_params": {},
        "sample_params": lambda rng: {
            "n_neighbors": _sample_int(rng, 5, 200),
        },
        "param_widgets": {
            "n_neighbors": ("int", 1, 500, 25),
        },
    },
}


def build_estimator(algo: str, params: dict):
    spec = ALGOS[algo]
    est = spec["cls"](**spec["fixed_params"], **params)
    if spec["needs_scaling"]:
        return make_pipeline(StandardScaler(), est)
    return est


def sample_random_params(algo: str, rng: np.random.Generator) -> dict:
    return ALGOS[algo]["sample_params"](rng)
