"""Validation glissante (walk-forward) + jeu de confirmation isolé.

Le bruteforce et le constructeur manuel évaluent un modèle sur UNE seule
coupure temporelle train/test — un modèle "gagnant" sur cette coupure
précise peut simplement avoir eu de la chance sur cette fenêtre (biais de
comparaisons multiples, en particulier avec le bruteforce qui teste des
dizaines de configurations). Ce module réentraîne une configuration donnée
(algo + hyperparamètres + features + stratégie) sur plusieurs fenêtres
temporelles successives (entraînement extensible, test glissant), puis sur
une tranche de confirmation finale jamais vue pendant la recherche — pour
juger si une configuration généralise vraiment avant de lui faire confiance."""
import pandas as pd

from .features import build_training_frame
from .train import train_and_evaluate


def generate_walk_forward_windows(min_date, max_date, n_windows: int = 4, test_months: int = 6,
                                   min_train_months: int = 24, holdout_months: int = 6):
    """Découpe [min_date, max_date] en `n_windows` fenêtres glissantes
    (entraînement extensible depuis min_date, test de `test_months` mois
    chacune, non chevauchantes) + une fenêtre de confirmation finale de
    `holdout_months` mois, réservée et jamais utilisée par les fenêtres.
    Retourne (liste de fenêtres en ordre chronologique, fenêtre de confirmation)."""
    min_date = pd.Timestamp(min_date)
    max_date = pd.Timestamp(max_date)
    holdout_start = max_date - pd.DateOffset(months=holdout_months)

    windows = []
    test_end = holdout_start
    for _ in range(n_windows):
        test_start = test_end - pd.DateOffset(months=test_months)
        train_span_months = (test_start.year - min_date.year) * 12 + (test_start.month - min_date.month)
        if test_start <= min_date or train_span_months < min_train_months:
            break
        windows.append({
            "train_start": min_date, "train_end": test_start,
            "test_start": test_start, "test_end": test_end,
        })
        test_end = test_start
    windows.reverse()

    holdout = {
        "train_start": min_date, "train_end": holdout_start,
        "test_start": holdout_start, "test_end": max_date,
    }
    return windows, holdout


def _evaluate_window(dataset, odds_w_col, odds_l_col, window, algo, params, features, strategy, seed):
    full = build_training_frame(dataset, odds_w_col, odds_l_col, seed=seed,
                                 min_date=window["train_start"], max_date=window["test_end"])
    if full.empty:
        return None
    train_f = full[full["tourney_date"] < window["test_start"]]
    test_f = full[full["tourney_date"] >= window["test_start"]]
    if train_f.empty or test_f.empty:
        return None
    try:
        return train_and_evaluate(train_f, test_f, algo, params, features, strategy)
    except ValueError:
        return None


def run_walk_forward(dataset, odds_w_col, odds_l_col, algo, params, features, strategy,
                      n_windows: int = 4, test_months: int = 6, min_train_months: int = 24,
                      holdout_months: int = 6, seed: int = 42):
    """Réentraîne/backteste la même configuration sur chaque fenêtre glissante
    puis sur la confirmation finale. Retourne (liste de résultats par fenêtre
    avec leurs bornes, résultat de confirmation ou None, bornes de la
    confirmation)."""
    min_date = dataset["tourney_date"].min()
    max_date = dataset["tourney_date"].max()
    windows, holdout = generate_walk_forward_windows(
        min_date, max_date, n_windows, test_months, min_train_months, holdout_months,
    )
    if not windows:
        raise ValueError(
            "Pas assez d'historique pour générer des fenêtres de validation glissante avec ces "
            "paramètres (réduis le nombre de fenêtres, leur durée, ou le minimum d'entraînement)."
        )

    window_results = []
    for w in windows:
        result = _evaluate_window(dataset, odds_w_col, odds_l_col, w, algo, params, features, strategy, seed)
        if result is not None:
            window_results.append({"window": w, "result": result})

    holdout_result = _evaluate_window(dataset, odds_w_col, odds_l_col, holdout, algo, params, features,
                                       strategy, seed)

    return window_results, holdout_result, holdout
