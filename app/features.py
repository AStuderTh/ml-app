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

# "surface" est un pseudo-feature du pool: quand il est sélectionné (bruteforce
# ou constructeur manuel), il est développé en 3 colonnes one-hot juste avant
# l'entraînement (cf. expand_features) — Carpet sert de catégorie de référence
# implicite (les 3 colonnes valent 0), car c'est la surface la plus rare et
# quasi abandonnée par le circuit depuis ~2009.
SURFACE_FEATURE = "surface"
SURFACE_DUMMY_COLS = ["surface_clay", "surface_grass", "surface_hard"]

# Pool de features candidates pour le constructeur manuel / le bruteforce.
# Convention: une valeur positive favorise p1.
FEATURE_POOL = [
    "elo_diff", "elo_surface_diff", "elo_diff_recent", "elo_surface_diff_recent",
    "rank_diff", "rank_points_diff",
    "age_diff", "ht_diff", "form_diff", "h2h_diff",
    "hand_advantage", "implied_prob_p1", "experience_diff", SURFACE_FEATURE,
]


def expand_features(features):
    """Développe le pseudo-feature 'surface' en ses 3 colonnes one-hot
    réelles. À appeler juste avant d'indexer un DataFrame de features (train,
    backtest, prédiction live) — jamais avant, pour que le tirage aléatoire du
    bruteforce et l'affichage des cases à cocher traitent 'surface' comme un
    seul choix cohérent plutôt que 3 cases indépendantes."""
    out = []
    for f in features:
        if f == SURFACE_FEATURE:
            out.extend(SURFACE_DUMMY_COLS)
        else:
            out.append(f)
    return out


def _days_between(date, prev_date):
    if prev_date is None:
        return None
    return (date - prev_date) / np.timedelta64(1, "D")


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

    n = len(df)
    out = {k: np.empty(n) for k in [
        "elo_w_pre", "elo_l_pre", "elo_surf_w_pre", "elo_surf_l_pre",
        "elo_recent_w_pre", "elo_recent_l_pre",
        "elo_surf_recent_w_pre", "elo_surf_recent_l_pre",
        "form_w_pre", "form_l_pre", "h2h_w_pre", "h2h_l_pre",
        "played_w_pre", "played_l_pre",
    ]}

    winners = df["winner_name"].values
    losers = df["loser_name"].values
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

    for k, v in out.items():
        df[k] = v
    return df


def build_training_frame(df_with_chrono: pd.DataFrame, odds_w_col: str, odds_l_col: str,
                          seed: int = 42, min_date=None, max_date=None) -> pd.DataFrame:
    """Filtre aux matchs avec cotes disponibles sur la période demandée, puis
    ré-attribue aléatoirement gagnant/perdant à p1/p2 (seed fixe -> résultat
    reproductible d'un run à l'autre) pour que le label ne soit pas toujours
    "p1 gagne"."""
    sub = df_with_chrono.dropna(subset=[odds_w_col, odds_l_col, "surface"]).copy()
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
    out["rank_diff"] = rank_p2 - rank_p1  # positif => p1 mieux classé
    out["rank_points_diff"] = pts_p1 - pts_p2
    out["age_diff"] = age_p1 - age_p2
    out["ht_diff"] = ht_p1 - ht_p2
    out["form_diff"] = form_p1 - form_p2
    out["h2h_diff"] = h2h_p1 - h2h_p2
    out["hand_advantage"] = np.where(hand_p1 == "L", 1, 0) - np.where(hand_p2 == "L", 1, 0)
    out["experience_diff"] = played_p1 - played_p2

    implied_p1_raw = 1.0 / out["odds_p1"]
    implied_p2_raw = 1.0 / out["odds_p2"]
    out["implied_prob_p1"] = implied_p1_raw / (implied_p1_raw + implied_p2_raw)

    # NB: on ne fait PAS de dropna global sur FEATURE_POOL ici: certaines
    # features (âge, taille, classement) sont parfois absentes selon les
    # joueurs, et un modèle qui ne les utilise pas ne doit pas perdre ces
    # lignes. Le dropna se fait par modèle, uniquement sur ses features
    # sélectionnées (cf. train.train_and_evaluate).
    return out
