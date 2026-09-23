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

# LightGBM est une dépendance OPTIONNELLE: import gardé plutôt que sec, sinon
# une installation sans lightgbm ferait planter tout le module — donc l'app
# entière, via streamlit_app -> modeling — alors que les 5 autres algos
# fonctionneraient très bien. Absent: l'entrée correspondante n'est simplement
# pas enregistrée dans ALGOS, et elle disparaît de l'UI comme du bruteforce.
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def _sample_int(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _sample_float(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _sample_log(rng, lo, hi):
    """Tirage log-uniforme, pour les paramètres à échelle multiplicative.

    Sur un taux d'apprentissage, l'écart entre 0.01 et 0.02 pèse autant que
    celui entre 0.15 et 0.30: ce qui compte est le rapport, pas la différence.
    Un tirage uniforme sur [0.01, 0.3] place environ 5 tirages sur 6 au-dessus
    de 0.05 et ne visite donc quasiment jamais le bas de la plage — qui est
    précisément la zone utile en boosting, là où un grand nombre d'arbres peut
    affiner sans surajuster.
    """
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


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
            "learning_rate": _sample_log(rng, 0.01, 0.3),
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


# Enregistré après coup (et non dans le littéral ci-dessus) pour que l'entrée
# disparaisse proprement quand lightgbm n'est pas installé.
if LGBMClassifier is not None:
    ALGOS["lightgbm"] = {
        "label": "LightGBM",
        "cls": LGBMClassifier,
        "needs_scaling": False,
        "fixed_params": {
            "random_state": 0,
            "n_jobs": -1,
            # LightGBM écrit sur stdout à CHAQUE fit ("Number of positive...",
            # "No further splits with positive gain..."). Le bruteforce en
            # enchaîne des centaines: sans -1, la console est noyée et les
            # vraies erreurs y deviennent illisibles.
            "verbose": -1,
        },
        # Croissance leaf-wise: c'est `num_leaves` qui borne la complexité d'un
        # arbre, pas `max_depth` comme pour GradientBoosting. On ne tire donc pas
        # max_depth ici — il ferait doublon avec num_leaves et n'élargirait
        # l'espace de recherche que pour du bruit.
        "sample_params": lambda rng: {
            "n_estimators": _sample_int(rng, 50, 400),
            "learning_rate": _sample_log(rng, 0.01, 0.3),
            "num_leaves": _sample_int(rng, 8, 64),
            "min_child_samples": _sample_int(rng, 5, 200),
            "colsample_bytree": _sample_float(rng, 0.5, 1.0),
        },
        # Mêmes clés que sample_params: charger un résultat de bruteforce dans
        # le constructeur manuel ignore silencieusement tout paramètre absent
        # d'ici (cf. streamlit_app.py), qui repartirait alors à sa valeur par
        # défaut sans que rien ne le signale.
        "param_widgets": {
            "n_estimators": ("int", 10, 800, 200),
            "learning_rate": ("float", 0.001, 1.0, 0.05),
            "num_leaves": ("int", 2, 255, 31),
            "min_child_samples": ("int", 1, 500, 20),
            "colsample_bytree": ("float", 0.3, 1.0, 1.0),
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
