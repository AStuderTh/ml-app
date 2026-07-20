"""Feature engineering. Tout est calculé en un seul passage chronologique
sur l'historique COMPLET des matchs (pas seulement ceux avec cotes), pour que
les ratings Elo/forme/H2H soient les plus fiables possible avant d'être
utilisés sur le sous-ensemble avec cotes (nécessaire pour le ROI)."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0

# Pool de features candidates pour le constructeur manuel / le bruteforce.
# Convention: une valeur positive favorise p1.
FEATURE_POOL = [
    "elo_diff", "elo_surface_diff", "rank_diff", "rank_points_diff",
    "age_diff", "ht_diff", "form_diff", "h2h_diff",
    "hand_advantage", "implied_prob_p1", "experience_diff",
]


def compute_chronological_features(matches: pd.DataFrame, elo_k: float = 32.0,
                                    form_window: int = 10) -> pd.DataFrame:
    df = matches.sort_values("tourney_date").reset_index(drop=True)

    elo = defaultdict(lambda: DEFAULT_ELO)
    elo_surf = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
    recent = defaultdict(lambda: deque(maxlen=form_window))
    h2h = defaultdict(lambda: defaultdict(int))
    played = defaultdict(int)

    n = len(df)
    out = {k: np.empty(n) for k in [
        "elo_w_pre", "elo_l_pre", "elo_surf_w_pre", "elo_surf_l_pre",
        "form_w_pre", "form_l_pre", "h2h_w_pre", "h2h_l_pre",
        "played_w_pre", "played_l_pre",
    ]}

    winners = df["winner_name"].values
    losers = df["loser_name"].values
    surfaces = df["surface"].fillna("Hard").values

    for idx in range(n):
        w, l, surf = winners[idx], losers[idx], surfaces[idx]
        ew, el = elo[w], elo[l]
        esw, esl = elo_surf[surf][w], elo_surf[surf][l]
        rw = recent[w]
        rl = recent[l]
        fw = sum(rw) / len(rw) if rw else 0.5
        fl = sum(rl) / len(rl) if rl else 0.5

        out["elo_w_pre"][idx], out["elo_l_pre"][idx] = ew, el
        out["elo_surf_w_pre"][idx], out["elo_surf_l_pre"][idx] = esw, esl
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
