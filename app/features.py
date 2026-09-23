"""Feature engineering. Tout est calculé en un seul passage chronologique
sur l'historique COMPLET des matchs (pas seulement ceux avec cotes), pour que
les ratings Elo/forme/H2H soient les plus fiables possible avant d'être
utilisés sur le sous-ensemble avec cotes (nécessaire pour le ROI)."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0

# Demi-vie de décroissance des features Elo "récentes" (elo_diff_recent,
# elo_surface_diff_recent): un rating non mis à jour depuis ce nombre de
# jours d'inactivité (globale, ou sur cette surface précise) perd la moitié
# de son écart à la moyenne (régression vers 1500). Capture le fait qu'une
# forme d'il y a plusieurs années — ou sur une surface plus jouée depuis
# longtemps — est une information moins fiable qu'une forme toute récente.
ELO_DECAY_HALF_LIFE_DAYS = 365.0

# --- Bloc service/retour ------------------------------------------------
# Moyennes de tour, servant de prior bayésien tant qu'un joueur a peu de
# points observés. Mesurées sur la base (69,4 % côté vainqueur, 58,5 % côté
# perdant); par construction moyenne_service + moyenne_retour = 1, puisque
# chaque point est gagné soit par le serveur soit par le relanceur.
TOUR_AVG_SERVE = 0.64
TOUR_AVG_RETURN = 1.0 - TOUR_AVG_SERVE

# Force du prior, en points de service fictifs. 300 ≈ 2 matchs: un joueur
# débutant est tiré vers la moyenne du tour au lieu d'afficher un taux extrême
# calculé sur 40 points.
SERVE_PRIOR_POINTS = 300.0

# Décroissance appliquée à l'historique à chaque nouveau match du joueur.
# 0.98 -> demi-vie d'environ 34 matchs, soit à peu près une saison: assez long
# pour être stable, assez court pour suivre une progression ou un déclin.
SERVE_DECAY = 0.98

# En-deçà, la statistique reste NaN plutôt que d'être servie comme une valeur
# quasi égale au prior: le dropna par modèle écarte alors la ligne, au lieu de
# faire passer une absence d'information pour une information moyenne.
MIN_SERVE_POINTS = 200.0

# NOTE — "meilleures victoires / pires défaites sur 6 mois": implémenté puis
# RETIRÉ, sur mesure. L'idée (décrire la FORME de la distribution des résultats
# et non seulement son centre) est saine, mais la version fondée sur l'écart
# d'Elo à l'adversaire ne mesure pas ce qu'elle prétend:
#     corr(ceiling_diff, elo_diff) = -0.78     AUC brute 0.3257 (donc inversée)
# En normalisant par l'Elo du joueur lui-même, un joueur faible qui gagne bat
# presque toujours "plus fort que lui": le plafond encode la faiblesse. À
# lignes identiques, l'ajout dégradait le log loss dans les deux algorithmes
# (0.6577 -> 0.6626 en logistique, 0.6553 -> 0.6678 en boosting).
# Signe révélateur: `consistency_diff` (amplitude plafond-plancher), la seule
# variante orthogonale à l'Elo (corr +0.05), était totalement inerte
# (AUC 0.5099). Une reprise devrait partir de la SURPRISE par rapport à
# l'attente Elo, seule formulation orthogonale par construction.

# --- Bloc fatigue -------------------------------------------------------
FATIGUE_WINDOW_DAYS = 14
# Au-delà, on considère le match précédent comme "long" (5 sets disputés, ou
# 3 sets très accrochés).
LONG_MATCH_MINUTES = 180.0
# Les durées aberrantes existent (0 min, 1146 min sur 29 667 matchs récents):
# elles fausseraient une somme glissante.
MIN_PLAUSIBLE_MINUTES, MAX_PLAUSIBLE_MINUTES = 20.0, 360.0

# "surface" est un pseudo-feature du pool: quand il est sélectionné (bruteforce
# ou constructeur manuel), il est développé en 3 colonnes one-hot juste avant
# l'entraînement (cf. expand_features) — Carpet sert de catégorie de référence
# implicite (les 3 colonnes valent 0), car c'est la surface la plus rare et
# quasi abandonnée par le circuit depuis ~2009.
SURFACE_FEATURE = "surface"
SURFACE_DUMMY_COLS = ["surface_clay", "surface_grass", "surface_hard"]

# Même mécanique que SURFACE_FEATURE (un seul choix dans l'UI/le bruteforce,
# développé en one-hot juste avant l'entraînement). Catégorie de référence
# implicite: 'A', les ATP 250/500, de loin la plus fréquente.
LEVEL_FEATURE = "tourney_level"
LEVEL_DUMMY_COLS = ["level_grandslam", "level_masters", "level_finals"]

# Statuts tennis-data.co ne correspondant à aucun match réellement disputé:
# les paris y sont annulés par les bookmakers, il n'y a donc rien à prédire.
NOT_PLAYED_COMMENTS = {"Walkover", "Awarded", "Disqualified", "Sched"}
RETIRED_COMMENT = "Retired"

# Pool de features candidates pour le constructeur manuel / le bruteforce.
# Convention: une valeur positive favorise p1 (pour les features
# différentielles; les indicateurs de contexte ci-dessous décrivent le match
# lui-même et n'avantagent aucun des deux joueurs).
FEATURE_POOL = [
    "elo_diff", "elo_surface_diff", "elo_diff_recent", "elo_surface_diff_recent",
    "rank_diff", "rank_points_diff",
    "points_dominance_diff", "points_dominance_surface_diff",
    "rest_days_diff", "minutes_14d_diff", "matches_14d_diff", "long_match_prev_diff",
    "age_diff", "ht_diff", "form_diff", "h2h_diff",
    "hand_advantage", "implied_prob_p1", "experience_diff", SURFACE_FEATURE,
    LEVEL_FEATURE, "best_of_5", "is_indoor",
]

# `rank_trend_diff` / `points_trend_diff` (trajectoire de classement sur 52
# semaines) sont bien CALCULÉES par build_training_frame, mais volontairement
# HORS de ce pool: mesurées sur les 40 305 matchs cotés, elles sont inertes.
#   AUC seule            : 0.5065 et 0.4936  (0.50 = hasard)
#   AUC sur les 10% de matchs les plus serrés : 0.4648 et 0.4703
#   corrélation au résidu du marché : -0.0012 et -0.0015
#     (à comparer à +0.0170 pour elo_diff_recent, déjà dans le pool)
# La raison est structurelle: une progression au classement s'obtient EN
# GAGNANT des matchs, victoires qui ont déjà fait monter l'Elo. La trajectoire
# ne fait que remesurer l'Elo avec plus de bruit et 52 semaines de retard.
#
# Elles restent volontairement hors du pool plutôt que supprimées: le pool
# alimente le tirage aléatoire du bruteforce, et deux features de bruit
# supplémentaires n'y feraient qu'élargir l'espace de recherche et augmenter
# le risque qu'un modèle paraisse bon par hasard sur le jeu de test.
# Pour les réactiver malgré tout, il suffit de les ajouter à la liste ci-dessus.
TREND_FEATURES = ["rank_trend_diff", "points_trend_diff"]

# Colonnes de classement officiel ATP (référentiel hebdomadaire Sackmann)
# nécessaires aux features de trajectoire. Absentes des bases construites avant
# l'ajout de scripts/consolidate/rankings.py: les features correspondantes
# valent alors NaN et le dropna par modèle de train.py écarte les lignes, plutôt
# que de faire échouer tout l'entraînement.
TREND_SOURCE_COLUMNS = [
    "winner_atp_rank", "loser_atp_rank",
    "winner_atp_rank_prev", "loser_atp_rank_prev",
    "winner_atp_points", "loser_atp_points",
    "winner_atp_points_prev", "loser_atp_points_prev",
]


def expand_features(features):
    """Développe les pseudo-features catégorielles ('surface', 'tourney_level')
    en leurs colonnes one-hot réelles. À appeler juste avant d'indexer un
    DataFrame de features (train, backtest, prédiction live) — jamais avant,
    pour que le tirage aléatoire du bruteforce et l'affichage des cases à
    cocher traitent chacune comme un seul choix cohérent plutôt que comme
    plusieurs cases indépendantes."""
    out = []
    for f in features:
        if f == SURFACE_FEATURE:
            out.extend(SURFACE_DUMMY_COLS)
        elif f == LEVEL_FEATURE:
            out.extend(LEVEL_DUMMY_COLS)
        else:
            out.append(f)
    return out


class _RateTracker:
    """Taux de points gagnés par joueur, lissé et décroissant dans le temps.

    On accumule des POINTS (gagnés / joués) et non des moyennes de matchs: un
    5-sets apporte trois fois plus d'information qu'une victoire expéditive, et
    le pondérer pareil reviendrait à jeter cette information.

    L'estimation est un lissage bayésien vers la moyenne du tour, ce qui évite
    qu'un joueur vu sur 40 points affiche 85 % de réussite.
    """

    __slots__ = ("won", "pts", "prior")

    def __init__(self, prior):
        self.won = defaultdict(float)
        self.pts = defaultdict(float)
        self.prior = prior

    def estimate(self, key):
        """Taux pré-match, ou None si l'historique est trop mince."""
        pts = self.pts[key]
        if pts < MIN_SERVE_POINTS:
            return None
        return (self.won[key] + SERVE_PRIOR_POINTS * self.prior) / (pts + SERVE_PRIOR_POINTS)

    def update(self, key, won, pts):
        self.won[key] = self.won[key] * SERVE_DECAY + won
        self.pts[key] = self.pts[key] * SERVE_DECAY + pts


def _adjust(observed, opponent_rate, opponent_avg):
    """Taux ATTENDU dans ce match précis, compte tenu de l'adversaire.

    Le taux historique brut d'un joueur suppose un adversaire moyen. Face à un
    relanceur d'élite, son pourcentage de points gagnés au service sera plus
    bas: on RETRANCHE donc l'excédent de l'adversaire par rapport à la moyenne
    du tour.

    Le signe compte énormément ici. En l'additionnant, l'écart entre les deux
    joueurs devient (service_p1 − service_p2) − (retour_p1 − retour_p2): comme
    un bon joueur est meilleur des DEUX côtés, les deux termes se compensent et
    le signal s'annule presque entièrement (AUC mesurée à 0.542, contre 0.725
    pour un simple écart d'Elo). En le retranchant, ils s'additionnent.

    `opponent_rate` est l'estimation PRÉ-MATCH de l'adversaire: elle ne dépend
    en rien du match en cours, donc aucune fuite.
    """
    if observed is None:
        return None
    if opponent_rate is None:
        return observed
    return observed - (opponent_rate - opponent_avg)


def _rank_trend(now, prev) -> np.ndarray:
    """Progression de classement sur 52 semaines, en écart de logarithmes.

    Le rang ATP est une échelle de RATIO, pas d'intervalle: passer de 500e à
    400e (100 places) est un progrès bien moindre que passer de 5e à 2e (3
    places). Une différence brute donnerait pourtant l'inverse et écraserait
    tout le haut du classement, là où se joue l'essentiel du circuit.
    `log(prev) - log(now)` mesure le facteur de progression, borné et
    comparable entre un 800e et un 3e.

    Signe: positif = le joueur a PROGRESSÉ (son numéro de rang a baissé).
    """
    now = np.asarray(now, dtype="float64")
    prev = np.asarray(prev, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.clip(prev, 1, None)) - np.log(np.clip(now, 1, None))
    return np.where(np.isfinite(out), out, np.nan)


def _points_trend(now, prev) -> np.ndarray:
    """Progression du capital de points sur 52 semaines, en écart de log1p.

    `log1p` et non `log`: un joueur peut légitimement avoir 0 point (sortie de
    classement, retour de blessure), ce qui rendrait un log brut infini.
    Signe: positif = le joueur a GAGNÉ des points.
    """
    now = np.asarray(now, dtype="float64")
    prev = np.asarray(prev, dtype="float64")
    with np.errstate(invalid="ignore"):
        out = np.log1p(np.clip(now, 0, None)) - np.log1p(np.clip(prev, 0, None))
    return np.where(np.isfinite(out), out, np.nan)


def player_key(df: pd.DataFrame, side: str) -> np.ndarray:
    """Clé d'identité d'un joueur pour l'accumulation Elo / forme / H2H.

    Le `player_id` du référentiel ATP (atp_players.csv) prime sur le nom: il
    est stable dans le temps et insensible aux variations d'orthographe entre
    sources. Le nom ne sert que de repli pour les rares lignes non résolues —
    sans quoi elles se retrouveraient toutes agrégées sous une même clé vide.
    """
    names = df[f"{side}_name"].astype(str)
    pid = df.get(f"{side}_player_id")
    if pid is None:
        return names.to_numpy(dtype=object)
    pid = pd.to_numeric(pid, errors="coerce")
    keyed = "id:" + pid.astype("Int64").astype(str)
    return keyed.where(pid.notna(), names).to_numpy(dtype=object)


def _days_between(date, prev_date):
    if prev_date is None:
        return None
    return (date - prev_date) / np.timedelta64(1, "D")


def _n(value):
    """None -> NaN, pour écrire dans un tableau numpy flottant."""
    return np.nan if value is None else value




def elo_decay(elo_value, days_inactive):
    """Régresse un rating Elo vers la moyenne (1500) en fonction du nombre de
    jours écoulés depuis la dernière mise à jour connue. Ne modifie pas le
    rating stocké (celui-ci continue d'évoluer normalement à chaque match) —
    c'est une transformation en lecture seule, utilisée uniquement pour les
    features '*_recent'."""
    if days_inactive is None or days_inactive <= 0:
        return elo_value
    factor = 0.5 ** (days_inactive / ELO_DECAY_HALF_LIFE_DAYS)
    return DEFAULT_ELO + (elo_value - DEFAULT_ELO) * factor


def compute_chronological_features(matches: pd.DataFrame, elo_k: float = 32.0,
                                    form_window: int = 10) -> pd.DataFrame:
    # kind="stable" (au lieu du quicksort par défaut, non stable): garantit un
    # ordre de traitement déterministe et reproductible pour les matchs qui
    # partagent la même tourney_date (l'heure exacte n'est pas connue dans la
    # base) — sans ça, l'Elo pourrait varier légèrement d'une exécution à
    # l'autre pour ces matchs à égalité de date.
    df = matches.sort_values("tourney_date", kind="stable").reset_index(drop=True)

    elo = defaultdict(lambda: DEFAULT_ELO)
    elo_surf = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
    recent = defaultdict(lambda: deque(maxlen=form_window))
    h2h = defaultdict(lambda: defaultdict(int))
    played = defaultdict(int)
    last_date = {}
    last_surf_date = defaultdict(dict)

    # Service/retour (Bloc 1) et charge récente (Bloc 2). Mêmes accumulateurs
    # chronologiques que l'Elo, donc mêmes garanties: tout est lu AVANT d'être
    # mis à jour, aucune valeur ne peut dépendre du match qu'elle décrit.
    srv = _RateTracker(TOUR_AVG_SERVE)
    ret = _RateTracker(TOUR_AVG_RETURN)
    srv_s = defaultdict(lambda: _RateTracker(TOUR_AVG_SERVE))
    ret_s = defaultdict(lambda: _RateTracker(TOUR_AVG_RETURN))
    load = defaultdict(deque)   # joueur -> (date, minutes) des 14 derniers jours
    prev_minutes = {}

    n = len(df)
    out = {k: np.empty(n) for k in [
        "elo_w_pre", "elo_l_pre", "elo_surf_w_pre", "elo_surf_l_pre",
        "elo_recent_w_pre", "elo_recent_l_pre",
        "elo_surf_recent_w_pre", "elo_surf_recent_l_pre",
        "form_w_pre", "form_l_pre", "h2h_w_pre", "h2h_l_pre",
        "played_w_pre", "played_l_pre",
    ]}
    # np.full et non np.empty: ces colonnes restent volontairement NaN quand
    # l'historique est insuffisant (cf. MIN_SERVE_POINTS), il ne faut donc pas
    # y laisser la mémoire non initialisée de np.empty.
    out.update({k: np.full(n, np.nan) for k in [
        "srv_w_pre", "srv_l_pre", "ret_w_pre", "ret_l_pre",
        "srv_surf_w_pre", "srv_surf_l_pre", "ret_surf_w_pre", "ret_surf_l_pre",
        "rest_w_pre", "rest_l_pre", "min14_w_pre", "min14_l_pre",
        "mat14_w_pre", "mat14_l_pre", "longprev_w_pre", "longprev_l_pre",
    ]})

    w_svpt = df["w_svpt"].to_numpy(dtype="float64") if "w_svpt" in df else np.full(n, np.nan)
    l_svpt = df["l_svpt"].to_numpy(dtype="float64") if "l_svpt" in df else np.full(n, np.nan)
    w_swon = ((df["w_1stWon"] + df["w_2ndWon"]).to_numpy(dtype="float64")
              if "w_1stWon" in df else np.full(n, np.nan))
    l_swon = ((df["l_1stWon"] + df["l_2ndWon"]).to_numpy(dtype="float64")
              if "l_1stWon" in df else np.full(n, np.nan))
    minutes = (df["minutes"].to_numpy(dtype="float64")
               if "minutes" in df else np.full(n, np.nan))
    window = np.timedelta64(FATIGUE_WINDOW_DAYS, "D")

    winners = player_key(df, "winner")
    losers = player_key(df, "loser")
    surfaces = df["surface"].fillna("Hard").values
    dates = df["tourney_date"].values

    for idx in range(n):
        w, l, surf, date = winners[idx], losers[idx], surfaces[idx], dates[idx]
        ew, el = elo[w], elo[l]
        esw, esl = elo_surf[surf][w], elo_surf[surf][l]
        rw = recent[w]
        rl = recent[l]
        fw = sum(rw) / len(rw) if rw else 0.5
        fl = sum(rl) / len(rl) if rl else 0.5

        days_w = _days_between(date, last_date.get(w))
        days_l = _days_between(date, last_date.get(l))
        days_sw = _days_between(date, last_surf_date[surf].get(w))
        days_sl = _days_between(date, last_surf_date[surf].get(l))

        out["elo_w_pre"][idx], out["elo_l_pre"][idx] = ew, el
        out["elo_surf_w_pre"][idx], out["elo_surf_l_pre"][idx] = esw, esl
        out["elo_recent_w_pre"][idx] = elo_decay(ew, days_w)
        out["elo_recent_l_pre"][idx] = elo_decay(el, days_l)
        out["elo_surf_recent_w_pre"][idx] = elo_decay(esw, days_sw)
        out["elo_surf_recent_l_pre"][idx] = elo_decay(esl, days_sl)
        out["form_w_pre"][idx], out["form_l_pre"][idx] = fw, fl
        out["h2h_w_pre"][idx], out["h2h_l_pre"][idx] = h2h[w][l], h2h[l][w]
        out["played_w_pre"][idx], out["played_l_pre"][idx] = played[w], played[l]

        # --- Bloc 1: service/retour, ajustés à la force de l'adversaire ---
        sw_raw, sl_raw = srv.estimate(w), srv.estimate(l)
        rw_raw, rl_raw = ret.estimate(w), ret.estimate(l)
        out["srv_w_pre"][idx] = _n(_adjust(sw_raw, rl_raw, TOUR_AVG_RETURN))
        out["srv_l_pre"][idx] = _n(_adjust(sl_raw, rw_raw, TOUR_AVG_RETURN))
        out["ret_w_pre"][idx] = _n(_adjust(rw_raw, sl_raw, TOUR_AVG_SERVE))
        out["ret_l_pre"][idx] = _n(_adjust(rl_raw, sw_raw, TOUR_AVG_SERVE))

        ssrf, rsrf = srv_s[surf], ret_s[surf]
        sw_s, sl_s = ssrf.estimate(w), ssrf.estimate(l)
        rw_s, rl_s = rsrf.estimate(w), rsrf.estimate(l)
        out["srv_surf_w_pre"][idx] = _n(_adjust(sw_s, rl_s, TOUR_AVG_RETURN))
        out["srv_surf_l_pre"][idx] = _n(_adjust(sl_s, rw_s, TOUR_AVG_RETURN))
        out["ret_surf_w_pre"][idx] = _n(_adjust(rw_s, sl_s, TOUR_AVG_SERVE))
        out["ret_surf_l_pre"][idx] = _n(_adjust(rl_s, sw_s, TOUR_AVG_SERVE))

        # --- Bloc 2: fraîcheur et charge des 14 derniers jours ---
        out["rest_w_pre"][idx] = _n(days_w)
        out["rest_l_pre"][idx] = _n(days_l)
        for who, mkey, ckey in ((w, "min14_w_pre", "mat14_w_pre"),
                                (l, "min14_l_pre", "mat14_l_pre")):
            dq = load[who]
            while dq and (date - dq[0][0]) > window:
                dq.popleft()
            out[mkey][idx] = sum(m for _, m in dq)
            out[ckey][idx] = len(dq)
        out["longprev_w_pre"][idx] = float(prev_minutes.get(w, 0.0) >= LONG_MATCH_MINUTES)
        out["longprev_l_pre"][idx] = float(prev_minutes.get(l, 0.0) >= LONG_MATCH_MINUTES)


        exp_w = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
        elo[w] = ew + elo_k * (1 - exp_w)
        elo[l] = el + elo_k * (0 - (1 - exp_w))

        exp_sw = 1.0 / (1.0 + 10 ** ((esl - esw) / 400.0))
        elo_surf[surf][w] = esw + elo_k * (1 - exp_sw)
        elo_surf[surf][l] = esl + elo_k * (0 - (1 - exp_sw))

        rw.append(1)
        rl.append(0)
        h2h[w][l] += 1
        played[w] += 1
        played[l] += 1
        last_date[w] = last_date[l] = date
        last_surf_date[surf][w] = last_surf_date[surf][l] = date

        # --- mise à jour service/retour, APRÈS lecture des valeurs pré-match ---
        wp, lp = w_svpt[idx], l_svpt[idx]
        ww, lw = w_swon[idx], l_swon[idx]
        if wp > 0 and lp > 0 and ww == ww and lw == lw:
            # les points de retour d'un joueur SONT les points de service que
            # son adversaire a perdus: rien à charger de plus.
            srv.update(w, ww, wp)
            srv.update(l, lw, lp)
            ret.update(w, lp - lw, lp)
            ret.update(l, wp - ww, wp)
            srv_s[surf].update(w, ww, wp)
            srv_s[surf].update(l, lw, lp)
            ret_s[surf].update(w, lp - lw, lp)
            ret_s[surf].update(l, wp - ww, wp)


        # --- mise à jour de la charge récente ---
        mn = minutes[idx]
        if mn == mn and MIN_PLAUSIBLE_MINUTES <= mn <= MAX_PLAUSIBLE_MINUTES:
            load[w].append((date, mn))
            load[l].append((date, mn))
            prev_minutes[w] = prev_minutes[l] = mn
        else:
            prev_minutes[w] = prev_minutes[l] = 0.0

    for k, v in out.items():
        df[k] = v
    return df


def build_training_frame(df_with_chrono: pd.DataFrame, odds_w_col: str, odds_l_col: str,
                          seed: int = 42, min_date=None, max_date=None) -> pd.DataFrame:
    """Filtre aux matchs avec cotes disponibles sur la période demandée, puis
    ré-attribue aléatoirement gagnant/perdant à p1/p2 (seed fixe -> résultat
    reproductible d'un run à l'autre) pour que le label ne soit pas toujours
    "p1 gagne".

    Les rencontres jamais disputées (forfait, match donné sur tapis vert) sont
    écartées d'office: il n'y a pas de tennis à prédire et les bookmakers y
    annulent les paris, elles ne feraient qu'ajouter du bruit d'étiquette. Les
    abandons en cours de match, eux, sont conservés — un vainqueur a bien été
    désigné et les paris sont généralement réglés — mais restent identifiables
    via la colonne `match_comment` pour qui veut les exclure aussi."""
    sub = df_with_chrono.dropna(subset=[odds_w_col, odds_l_col, "surface"]).copy()
    if "match_comment" in sub.columns:
        sub = sub[~sub["match_comment"].isin(NOT_PLAYED_COMMENTS)]
    if min_date is not None:
        sub = sub[sub["tourney_date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        sub = sub[sub["tourney_date"] <= pd.Timestamp(max_date)]
    sub = sub[(sub[odds_w_col] > 1.0) & (sub[odds_l_col] > 1.0)]
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return sub

    rng = np.random.RandomState(seed)
    p1_is_winner = rng.rand(len(sub)) < 0.5

    def pick(a, b):
        return np.where(p1_is_winner, sub[a].values, sub[b].values)

    out = pd.DataFrame(index=sub.index)
    out["match_id"] = sub["match_id"]
    out["tourney_date"] = sub["tourney_date"]
    out["surface"] = sub["surface"]
    out["tourney_level"] = sub["tourney_level"]
    out["p1_name"] = pick("winner_name", "loser_name")
    out["p2_name"] = pick("loser_name", "winner_name")
    out["label"] = p1_is_winner.astype(int)

    out["odds_p1"] = pick(odds_w_col, odds_l_col)
    out["odds_p2"] = pick(odds_l_col, odds_w_col)

    elo_p1 = pick("elo_w_pre", "elo_l_pre")
    elo_p2 = pick("elo_l_pre", "elo_w_pre")
    elo_surf_p1 = pick("elo_surf_w_pre", "elo_surf_l_pre")
    elo_surf_p2 = pick("elo_surf_l_pre", "elo_surf_w_pre")
    elo_recent_p1 = pick("elo_recent_w_pre", "elo_recent_l_pre")
    elo_recent_p2 = pick("elo_recent_l_pre", "elo_recent_w_pre")
    elo_surf_recent_p1 = pick("elo_surf_recent_w_pre", "elo_surf_recent_l_pre")
    elo_surf_recent_p2 = pick("elo_surf_recent_l_pre", "elo_surf_recent_w_pre")
    rank_p1 = pick("winner_rank", "loser_rank")
    rank_p2 = pick("loser_rank", "winner_rank")
    pts_p1 = pick("winner_rank_points", "loser_rank_points")
    pts_p2 = pick("loser_rank_points", "winner_rank_points")
    age_p1 = pick("winner_age", "loser_age")
    age_p2 = pick("loser_age", "winner_age")
    ht_p1 = pick("winner_ht", "loser_ht")
    ht_p2 = pick("loser_ht", "winner_ht")
    hand_p1 = pick("winner_hand", "loser_hand")
    hand_p2 = pick("loser_hand", "winner_hand")
    form_p1 = pick("form_w_pre", "form_l_pre")
    form_p2 = pick("form_l_pre", "form_w_pre")
    h2h_p1 = pick("h2h_w_pre", "h2h_l_pre")
    h2h_p2 = pick("h2h_l_pre", "h2h_w_pre")
    played_p1 = pick("played_w_pre", "played_l_pre")
    played_p2 = pick("played_l_pre", "played_w_pre")

    out["elo_diff"] = elo_p1 - elo_p2
    out["elo_surface_diff"] = elo_surf_p1 - elo_surf_p2
    out["elo_diff_recent"] = elo_recent_p1 - elo_recent_p2
    out["elo_surface_diff_recent"] = elo_surf_recent_p1 - elo_surf_recent_p2
    out["surface_clay"] = (sub["surface"] == "Clay").astype(int).values
    out["surface_grass"] = (sub["surface"] == "Grass").astype(int).values
    out["surface_hard"] = (sub["surface"] == "Hard").astype(int).values

    # Contexte du match. NaN (et non 0) quand l'info manque, pour que le
    # dropna par modèle de train.py écarte ces lignes au lieu de les faire
    # passer en douce pour la catégorie de référence.
    lvl, bo = sub["tourney_level"], sub["best_of"]
    lvl_known, bo_known = lvl.notna().values, bo.notna().values
    out["level_grandslam"] = np.where(lvl_known, (lvl == "G").astype(float).values, np.nan)
    out["level_masters"] = np.where(lvl_known, (lvl == "M").astype(float).values, np.nan)
    out["level_finals"] = np.where(lvl_known, (lvl == "F").astype(float).values, np.nan)
    out["best_of_5"] = np.where(bo_known, (bo == 5).astype(float).values, np.nan)
    if "match_comment" in sub.columns:
        out["match_comment"] = sub["match_comment"].values
    out["rank_diff"] = rank_p2 - rank_p1  # positif => p1 mieux classé
    out["rank_points_diff"] = pts_p1 - pts_p2

    # --- Bloc 1: domination aux points (convention: positif => p1) ---
    #
    # Une seule feature par périmètre, et non une "service" plus une "retour":
    # une fois l'ajustement à l'adversaire appliqué, les deux sont ALGÉBRIQUEMENT
    # identiques (corrélation mesurée 1.000000, écart max 1e-16), car
    #   (srv_p1 − ret_p2) − (srv_p2 − ret_p1) = (srv_p1 − srv_p2) + (ret_p1 − ret_p2)
    # est symétrique en service/retour. D'où le nom: ce que ça mesure est la
    # domination TOTALE aux points, pas le service seul. En garder deux
    # n'ajouterait aucune information, mais introduirait une colinéarité
    # parfaite et gonflerait l'espace de recherche du bruteforce.
    for name, w_col, l_col in [
        ("points_dominance_diff", "srv_w_pre", "srv_l_pre"),
        ("points_dominance_surface_diff", "srv_surf_w_pre", "srv_surf_l_pre"),
    ]:
        if w_col in sub.columns:
            out[name] = pick(w_col, l_col) - pick(l_col, w_col)
        else:
            out[name] = np.nan

    # --- Bloc 2: fraîcheur. Signe choisi pour que POSITIF FAVORISE p1 ---
    # p1 est avantagé quand il est le plus reposé (plus de jours de repos, mais
    # MOINS de minutes et de matchs récents), d'où l'inversion sur la charge.
    if "rest_w_pre" in sub.columns:
        out["rest_days_diff"] = pick("rest_w_pre", "rest_l_pre") - pick("rest_l_pre", "rest_w_pre")
        out["minutes_14d_diff"] = pick("min14_l_pre", "min14_w_pre") - pick("min14_w_pre", "min14_l_pre")
        out["matches_14d_diff"] = pick("mat14_l_pre", "mat14_w_pre") - pick("mat14_w_pre", "mat14_l_pre")
        out["long_match_prev_diff"] = (pick("longprev_l_pre", "longprev_w_pre")
                                       - pick("longprev_w_pre", "longprev_l_pre"))
    else:
        for c in ("rest_days_diff", "minutes_14d_diff", "matches_14d_diff", "long_match_prev_diff"):
            out[c] = np.nan


    # --- Bloc 4: contexte. Ne favorise aucun joueur, mais interagit avec la
    # domination au service, qui pèse davantage en salle. ---
    if "indoor" in sub.columns:
        ind = sub["indoor"].astype(str).str.upper()
        out["is_indoor"] = np.where(ind.isin(["I", "O"]), (ind == "I").astype(float), np.nan)
    else:
        out["is_indoor"] = np.nan

    # Trajectoire de classement sur 52 semaines. Seule feature du pool à
    # capturer une DYNAMIQUE et non un état: deux joueurs peuvent être 30e au
    # même moment, l'un en train de monter depuis la 120e place, l'autre en
    # chute depuis la 8e — le classement seul les rend indiscernables.
    # 52 semaines et non 4 ou 12: l'ATP se défend à date anniversaire, donc ce
    # recul neutralise la saisonnalité du calendrier (cf. rankings.py).
    if all(c in sub.columns for c in TREND_SOURCE_COLUMNS):
        trend_r1 = _rank_trend(pick("winner_atp_rank", "loser_atp_rank"),
                               pick("winner_atp_rank_prev", "loser_atp_rank_prev"))
        trend_r2 = _rank_trend(pick("loser_atp_rank", "winner_atp_rank"),
                               pick("loser_atp_rank_prev", "winner_atp_rank_prev"))
        trend_p1 = _points_trend(pick("winner_atp_points", "loser_atp_points"),
                                 pick("winner_atp_points_prev", "loser_atp_points_prev"))
        trend_p2 = _points_trend(pick("loser_atp_points", "winner_atp_points"),
                                 pick("loser_atp_points_prev", "winner_atp_points_prev"))
        out["rank_trend_diff"] = trend_r1 - trend_r2
        out["points_trend_diff"] = trend_p1 - trend_p2
    else:
        out["rank_trend_diff"] = np.nan
        out["points_trend_diff"] = np.nan

    out["age_diff"] = age_p1 - age_p2
    out["ht_diff"] = ht_p1 - ht_p2
    out["form_diff"] = form_p1 - form_p2
    out["h2h_diff"] = h2h_p1 - h2h_p2
    out["hand_advantage"] = np.where(hand_p1 == "L", 1, 0) - np.where(hand_p2 == "L", 1, 0)
    out["experience_diff"] = played_p1 - played_p2

    # Profondeur d'historique du joueur le MOINS documenté du match. Ce n'est
    # pas une feature (elle n'avantage ni p1 ni p2, et n'est pas dans
    # FEATURE_POOL) mais un indicateur de FIABILITÉ: sous quelques dizaines de
    # matchs, l'Elo d'un joueur n'a pas convergé depuis sa valeur initiale, et
    # la probabilité produite reflète surtout cette valeur par défaut. Sert à
    # découper l'évaluation par profondeur d'historique (cf.
    # scripts/experience_threshold.py) puis à couper le signal en production
    # sur le même critère.
    out["min_played"] = np.minimum(played_p1, played_p2)

    implied_p1_raw = 1.0 / out["odds_p1"]
    implied_p2_raw = 1.0 / out["odds_p2"]
    out["implied_prob_p1"] = implied_p1_raw / (implied_p1_raw + implied_p2_raw)

    # NB: on ne fait PAS de dropna global sur FEATURE_POOL ici: certaines
    # features (âge, taille, classement) sont parfois absentes selon les
    # joueurs, et un modèle qui ne les utilise pas ne doit pas perdre ces
    # lignes. Le dropna se fait par modèle, uniquement sur ses features
    # sélectionnées (cf. train.train_and_evaluate).
    return out
