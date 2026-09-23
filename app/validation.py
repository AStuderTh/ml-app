"""Validation glissante (walk-forward) + jeu de confirmation isolé.

Le bruteforce et le constructeur manuel évaluent un modèle sur UNE seule
coupure temporelle train/test — un modèle "gagnant" sur cette coupure
précise peut simplement avoir eu de la chance sur cette fenêtre (biais de
comparaisons multiples, en particulier avec le bruteforce qui teste des
dizaines de configurations). Ce module réentraîne une configuration donnée
(algo + hyperparamètres + features + stratégie) sur plusieurs fenêtres
temporelles successives (entraînement extensible, test glissant), puis sur
une tranche de confirmation finale jamais vue pendant la recherche — pour
juger si une configuration généralise vraiment avant de lui faire confiance.

PORTÉE DE CE QUE CHAQUE CHIFFRE PROUVE. Les fenêtres glissantes recouvrent en
général la période sur laquelle la config a été CHOISIE: le bruteforce y a
trié ses modèles, l'optimisation ROI y a réglé son seuil d'edge et ses bornes
de cote. Réentraîner le modèle par fenêtre ne défait pas ce choix — les
fenêtres mesurent donc la STABILITÉ d'une config d'une période à l'autre, pas
sa capacité à généraliser, et leur ROI reste in-sample pour la stratégie de
mise. Seule la tranche de confirmation, invisible depuis l'exploration (cf. la
réserve de confirmation côté UI), est réellement hors échantillon."""
import pandas as pd

from .features import NOT_PLAYED_COMMENTS, build_training_frame
from .train import train_and_evaluate


def _with_odds(dataset, odds_w_col, odds_l_col):
    """Lignes que `build_training_frame` retiendra pour cette source de cotes.
    Même filtre que lui, à un seul endroit: les bornes et la couverture par
    surface calculées ci-dessous doivent décrire sa sortie, pas une
    approximation qui dériverait à la première divergence."""
    sub = dataset.dropna(subset=[odds_w_col, odds_l_col, "surface"])
    if "match_comment" in sub.columns:
        sub = sub[~sub["match_comment"].isin(NOT_PLAYED_COMMENTS)]
    return sub[(sub[odds_w_col] > 1.0) & (sub[odds_l_col] > 1.0)]


def surfaces_with_odds(dataset, odds_w_col, odds_l_col) -> set:
    """Surfaces sur lesquelles cette source de cotes a au moins un match.

    Toutes les sources ne couvrent pas les mêmes surfaces: le moquette a
    disparu du circuit en 2009 alors que les cotes moyennes ne commencent
    qu'en 2010, donc « Carpet » n'existe pas pour cette source. Sans ce
    filtre, l'UI laisse cocher une combinaison surface/source vide."""
    return set(_with_odds(dataset, odds_w_col, odds_l_col)["surface"].unique())


def usable_date_range(dataset, odds_w_col, odds_l_col):
    """Première et dernière date RÉELLEMENT exploitables pour cette source de
    cotes, et non les bornes du dataset brut.

    L'historique remonte à 1968 alors qu'aucune colonne de cotes ne commence
    avant 2002, et toutes ne s'arrêtent pas à la même date (Pinnacle est en
    retard sur B365/avg). Découper les fenêtres sur les bornes du dataset brut
    produit deux défauts silencieux: la garde `min_train_months` compare un
    historique d'entraînement mesuré depuis 1968 alors que les lignes
    utilisables commencent des décennies plus tard — elle ne se déclenche donc
    jamais — et la tranche de confirmation peut tomber entièrement après la
    dernière cote disponible (0 match, confirmation impossible sans que rien
    n'explique pourquoi). On applique donc ici le même filtre que
    `build_training_frame`, dont ces bornes décrivent la sortie.

    Retourne (None, None) si la source ne couvre aucun match."""
    sub = _with_odds(dataset, odds_w_col, odds_l_col)
    if sub.empty:
        return None, None
    return sub["tourney_date"].min(), sub["tourney_date"].max()


def generate_walk_forward_windows(min_date, max_date, n_windows: int = 4, test_months: int = 6,
                                   min_train_months: int = 24, holdout_months: int = 6):
    """Découpe [min_date, max_date] en `n_windows` fenêtres glissantes
    (entraînement extensible depuis min_date, test de `test_months` mois
    chacune, non chevauchantes) + une fenêtre de confirmation finale de
    `holdout_months` mois, réservée et jamais utilisée par les fenêtres.

    `min_date`/`max_date` doivent être les bornes EXPLOITABLES de la source de
    cotes (cf. usable_date_range) et non celles du dataset brut, sans quoi la
    garde `min_train_months` ci-dessous mesure un historique fictif.

    Les fenêtres sont semi-ouvertes [test_start, test_end): la borne haute de
    l'une est la borne basse de la suivante, donc l'inclure des deux côtés
    ferait évaluer deux fois les matchs datés pile sur la borne — et, à la
    dernière borne, mettrait des matchs à la fois dans une fenêtre de recherche
    et dans la confirmation censée ne rien partager avec elles. La
    confirmation, elle, est fermée à droite: `max_date` est le dernier jour de
    données disponible, l'exclure perdrait ces matchs.

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
            "test_end_inclusive": False,
        })
        test_end = test_start
    windows.reverse()

    holdout = {
        "train_start": min_date, "train_end": holdout_start,
        "test_start": holdout_start, "test_end": max_date,
        "test_end_inclusive": True,
    }
    return windows, holdout


def _evaluate_window(dataset, odds_w_col, odds_l_col, window, algo, params, features, strategy, seed):
    full = build_training_frame(dataset, odds_w_col, odds_l_col, seed=seed,
                                 min_date=window["train_start"], max_date=window["test_end"])
    if full.empty:
        return None
    train_f = full[full["tourney_date"] < window["test_start"]]
    test_f = full[full["tourney_date"] >= window["test_start"]]
    # build_training_frame borne à `<= test_end`: il reste donc à retirer la
    # borne haute elle-même quand la fenêtre est semi-ouverte (cf. supra).
    if not window.get("test_end_inclusive", False):
        test_f = test_f[test_f["tourney_date"] < window["test_end"]]
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
    min_date, max_date = usable_date_range(dataset, odds_w_col, odds_l_col)
    if min_date is None:
        raise ValueError(
            "Cette source de cotes ne couvre aucun match du périmètre sélectionné (surfaces, abandons) — "
            "impossible de générer des fenêtres de validation glissante."
        )
    windows, holdout = generate_walk_forward_windows(
        min_date, max_date, n_windows, test_months, min_train_months, holdout_months,
    )
    if not windows:
        raise ValueError(
            "Pas assez d'historique pour générer des fenêtres de validation glissante avec ces paramètres "
            f"(cotes disponibles du {min_date.date()} au {max_date.date()} — réduis le nombre de fenêtres, "
            "leur durée, ou le minimum d'entraînement)."
        )

    window_results = []
    for w in windows:
        result = _evaluate_window(dataset, odds_w_col, odds_l_col, w, algo, params, features, strategy, seed)
        if result is not None:
            window_results.append({"window": w, "result": result})

    holdout_result = _evaluate_window(dataset, odds_w_col, odds_l_col, holdout, algo, params, features,
                                       strategy, seed)

    return window_results, holdout_result, holdout
