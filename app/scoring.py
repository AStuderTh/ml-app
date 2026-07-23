"""Score composite de qualité d'un modèle: combine plusieurs métriques déjà
calculées (train.train_and_evaluate / backtest.run_backtest) en une seule note
0-100, pour classer les résultats du bruteforce ou du constructeur manuel.

Chaque métrique est ramenée à une échelle de "skill" 0-1 via une référence
ABSOLUE (pas un min-max relatif au lot de résultats comparés): un modèle unique
(constructeur manuel) doit pouvoir recevoir une note aussi bien qu'un lot de
500 modèles (bruteforce), et deux runs de sessions différentes doivent rester
comparables entre eux. Les références (baseline "coin flip" pour log loss/
Brier, etc.) sont documentées à côté de chaque constante ci-dessous.
"""
import numpy as np

# Poids par défaut. Ajustables librement (somme = 1.0 non obligatoire, la
# fonction normalise par la somme réelle des poids fournis).
#
# Pas de ROI ici: dans le bruteforce, la stratégie de mise (edge_threshold,
# kelly_fraction) est tirée AU HASARD indépendamment du modèle (cf.
# bruteforce.sample_config), donc le ROI mesure autant la chance du tirage de
# stratégie que la qualité du modèle — un même modèle avec la même
# probabilité prédite peut afficher un ROI du simple au double rien qu'en
# changeant de seuil. Le score ici vise la compétence intrinsèque du modèle
# (prédit-il un vrai signal ?); le ROI/nombre de paris restent affichés à part
# dans le tableau de résultats, à vérifier séparément (idéalement en walk-
# forward) avant d'exploiter un modèle pour de vrai.
DEFAULT_WEIGHTS = {
    "logloss": 0.32,
    "brier": 0.26,
    "calibration": 0.21,
    "auc": 0.11,
    "n_matches": 0.10,
}

# Chaque métrique est bornée par un plancher (skill=0, "pas d'info" / aussi
# mauvais qu'un pile-ou-face) et un plafond (skill=1). Le plafond n'est PAS la
# perfection théorique (logloss=0, AUC=1): sur des matchs de tennis réels,
# l'issue garde une part d'aléa irréductible, donc un modèle qui approche 0
# n'existe pas — utiliser 0 comme plafond écraserait un excellent modèle vers
# un score proche de 0. Les plafonds ci-dessous ont été calibrés empiriquement
# sur data/tennis.db (bruteforce de 150 modèles, features rank/Elo/forme/H2H,
# 2005-2022 train / 2023+ test): meilleur logloss observé ~0.62, meilleur
# Brier ~0.207, meilleur AUC ~0.743. Les plafonds visent légèrement au-dessus
# de ce maximum observé (marge pour une recherche plus poussée), PAS un niveau
# "état de l'art" générique de la littérature qui suppose souvent des features
# plus riches (cotes multi-bookmakers, stats in-play, etc.) absentes ici.
LOGLOSS_FLOOR, LOGLOSS_CEIL = np.log(2), 0.60
BRIER_FLOOR, BRIER_CEIL = 0.25, 0.20
AUC_FLOOR, AUC_CEIL = 0.5, 0.76
# ECE: 0 (parfaitement calibré) est réellement atteignable après calibration
# (CalibratedClassifierCV), donc plafond théorique conservé ici.
ECE_FLOOR, ECE_CEIL = 0.10, 0.0
# Nombre de matchs de test à partir duquel l'échantillon est jugé "large"
# (confiance statistique ~ sqrt(n), donc skill en sqrt(n) et pas linéaire: ça
# pénalise fortement les petits échantillons sans sur-récompenser les gros).
N_MATCHES_REF = 500


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE standard: écart moyen (pondéré par la taille de chaque bucket) entre
    probabilité prédite et fréquence réelle observée, sur des buckets de
    largeur égale — version scalaire du diagramme de fiabilité déjà affiché
    dans l'UI (cf. streamlit_app.render_calibration)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def _skill(value: float, floor: float, ceiling: float) -> float:
    """0 au niveau floor (aucune info), 1 au niveau ceiling (excellent), tronqué
    en dehors. Marche dans les deux sens: pour une métrique où plus petit est
    meilleur (log loss...), passer floor > ceiling."""
    if value is None or value != value:  # None ou NaN
        return float("nan")
    return float(np.clip((value - floor) / (ceiling - floor), 0.0, 1.0))


def _skill_anchored(value: float, floor: float, mid: float, ceiling: float) -> float:
    """Variante de _skill avec un point de pivot: skill=0.5 exactement à `mid`
    (repère intuitif, ex. 0% de ROI = point mort), skill=0 à floor, skill=1 à
    ceiling — deux segments linéaires au lieu d'un seul. Permet de garder un
    floor éloigné (pour différencier des stratégies toutes perdantes entre
    elles) sans que ça fasse remonter artificiellement le score des
    stratégies proches du point mort."""
    if value is None or value != value:  # None ou NaN
        return float("nan")
    if value >= mid:
        return float(0.5 + 0.5 * np.clip((value - mid) / (ceiling - mid), 0.0, 1.0))
    return float(0.5 * np.clip((value - floor) / (mid - floor), 0.0, 1.0))


def compute_quality_score(metrics: dict, weights: dict = None) -> dict:
    """Calcule la note composite pour UN résultat de train_and_evaluate, à
    partir des seules métriques du modèle (pas du backtest: cf. note sur
    DEFAULT_WEIGHTS plus haut). Renvoie {"score": 0-100, "components": {nom:
    (skill_0_1, poids)}} pour permettre d'afficher le détail du calcul dans
    l'UI plutôt qu'un chiffre opaque."""
    weights = weights or DEFAULT_WEIGHTS

    ece = metrics.get("ece", float("nan"))
    n_test = metrics.get("n_test", 0)

    components = {
        "logloss": _skill(metrics.get("logloss", float("nan")), LOGLOSS_FLOOR, LOGLOSS_CEIL),
        "brier": _skill(metrics.get("brier", float("nan")), BRIER_FLOOR, BRIER_CEIL),
        "calibration": _skill(ece, ECE_FLOOR, ECE_CEIL),
        "auc": _skill(metrics.get("auc", float("nan")), AUC_FLOOR, AUC_CEIL),
        "n_matches": _skill(np.sqrt(max(n_test, 0)), 0.0, np.sqrt(N_MATCHES_REF)),
    }

    weighted_sum, total_weight = 0.0, 0.0
    for name, skill in components.items():
        w = weights.get(name, 0.0)
        if w <= 0 or skill != skill:  # poids nul ou métrique indisponible (NaN) -> exclue du calcul
            continue
        weighted_sum += skill * w
        total_weight += w

    score = round(weighted_sum / total_weight * 100, 1) if total_weight > 0 else float("nan")
    return {"score": score, "components": {k: (v, weights.get(k, 0.0)) for k, v in components.items()}}


# ------------------------------------------------------------------------
# Score ROI: pour classer des STRATÉGIES de mise (edge_threshold, kelly...)
# sur un modèle déjà entraîné et figé (cf. app.roi_bruteforce). N'a rien à
# voir avec compute_quality_score ci-dessus, qui juge le modèle indépendamment
# de toute stratégie — ici c'est l'inverse: le modèle est fixe, seule la
# stratégie varie.
# ------------------------------------------------------------------------
ROI_SCORE_WEIGHTS = {"roi": 0.5, "n_bets": 0.25, "drawdown": 0.25}
# Plage calibrée empiriquement (app.roi_bruteforce sur des modèles réels sans edge):
# ROI observé de -60% à -8%, drawdown de 91% à 282% en mise flat sur des milliers de
# paris — un plancher à 0%/40% (assez raisonnable pour une stratégie "normale" bornée
# en risque) écrasait TOUT ce lot au même plancher (skill=0 partout), rendant le score
# incapable de distinguer "un peu perdant" de "catastrophique". Élargi pour retrouver
# une vraie différenciation, y compris entre stratégies toutes perdantes.
# La composante ROI utilise _skill_anchored (pas _skill): 0% de ROI (point mort)
# est ancré à skill=0.5 pile, quel que soit le floor élargi ci-dessous — sinon un
# floor à -30% faisait déjà remonter une stratégie à 0% de ROI à skill=0.75.
ROI_FLOOR, ROI_CEIL = -0.30, 0.10             # -30% (très mauvais) -> 10% (edge réel déjà très bon)
N_BETS_REF = 500                              # nombre de paris à partir duquel l'échantillon est jugé large
DRAWDOWN_FLOOR, DRAWDOWN_CEIL = 1.50, 0.05    # 150% de drawdown -> 0, 5% -> 1


def compute_roi_score(backtest_metrics: dict, weights: dict = None) -> dict:
    """Note composite 0-100 pour classer des stratégies de mise entre elles,
    sur un modèle fixe: récompense un ROI élevé, un nombre de paris suffisant
    pour y faire confiance, et un drawdown maîtrisé (une stratégie gagnante en
    moyenne mais avec 80% de drawdown reste inexploitable en pratique)."""
    weights = weights or ROI_SCORE_WEIGHTS
    n_bets = backtest_metrics.get("n_bets", 0)

    if n_bets <= 0:
        return {"score": 0.0, "components": {
            "roi": (0.0, weights.get("roi", 0.0)),
            "n_bets": (0.0, weights.get("n_bets", 0.0)),
            "drawdown": (float("nan"), weights.get("drawdown", 0.0)),
        }}

    components = {
        "roi": _skill_anchored(backtest_metrics.get("roi", float("nan")), ROI_FLOOR, 0.0, ROI_CEIL),
        "n_bets": _skill(np.sqrt(max(n_bets, 0)), 0.0, np.sqrt(N_BETS_REF)),
        "drawdown": _skill(backtest_metrics.get("max_drawdown", float("nan")), DRAWDOWN_FLOOR, DRAWDOWN_CEIL),
    }

    weighted_sum, total_weight = 0.0, 0.0
    for name, skill in components.items():
        w = weights.get(name, 0.0)
        if w <= 0 or skill != skill:
            continue
        weighted_sum += skill * w
        total_weight += w

    score = round(weighted_sum / total_weight * 100, 1) if total_weight > 0 else float("nan")
    return {"score": score, "components": {k: (v, weights.get(k, 0.0)) for k, v in components.items()}}
