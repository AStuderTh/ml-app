"""Ajoute la couche cotes/série/court de tennis-data.co par-dessus le pool
Sackmann+TML, et produit le schéma final prêt pour la base de sortie."""
import hashlib

import numpy as np
import pandas as pd

from . import normalize as nm

CONF_RANK = {"high": 4, "manual_confirmed": 4, "medium": 3, "single_source": 2, "pending_review": 1}

FINAL_COLUMNS = [
    "match_id", "sources", "match_confidence", "has_pending_review", "review_ids",
    "tourney_id", "tourney_name", "tourney_date", "surface", "indoor", "court", "series",
    "tourney_level", "draw_size", "round", "best_of",
    "winner_name", "winner_id_sackmann", "winner_id_tml", "winner_seed", "winner_entry",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_name", "loser_id_sackmann", "loser_id_tml", "loser_seed", "loser_entry",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",
    "score", "minutes", "match_comment",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "odds_w_b365", "odds_l_b365", "odds_w_ps", "odds_l_ps", "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg",
]


def _combine_conf(a, b):
    ra, rb = CONF_RANK.get(a, 0), CONF_RANK.get(b, 0)
    return a if ra <= rb else b


def _make_match_id(row):
    # le score est inclus pour distinguer 2 confrontations réellement
    # distinctes entre les mêmes joueurs le même jour/tournoi/round (rare,
    # mais arrive sur les anciens formats "round robin à léges" type
    # Champions Classic 1970s)
    key = (f"{row.get('tourney_id')}|{row.get('round')}|{row.get('winner_name')}|"
           f"{row.get('loser_name')}|{row.get('tourney_date')}|{row.get('score')}")
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def _is_set(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def finalize_row(pool_row, td_row=None):
    r = dict(pool_row)
    review_ids = [r["review_id"]] if _is_set(r.get("review_id")) else []
    if td_row is not None:
        r["sources"] = r["sources"] + ",tennis_data" if r["sources"] else "tennis_data"
        r["court"] = td_row.get("Court")
        r["series"] = td_row.get("Series")
        r["match_comment"] = td_row.get("Comment")
        r["odds_w_b365"], r["odds_l_b365"] = td_row.get("B365W"), td_row.get("B365L")
        r["odds_w_ps"], r["odds_l_ps"] = td_row.get("PSW"), td_row.get("PSL")
        r["odds_w_max"], r["odds_l_max"] = td_row.get("MaxW"), td_row.get("MaxL")
        r["odds_w_avg"], r["odds_l_avg"] = td_row.get("AvgW"), td_row.get("AvgL")
        if _is_set(td_row.get("review_id")):
            review_ids.append(td_row["review_id"])
        r["match_confidence"] = _combine_conf(r.get("match_confidence", "single_source"),
                                               td_row.get("_stage2_confidence", "single_source"))
    else:
        for c in ("court", "series", "match_comment", "odds_w_b365", "odds_l_b365",
                  "odds_w_ps", "odds_l_ps", "odds_w_max", "odds_l_max", "odds_w_avg", "odds_l_avg"):
            r.setdefault(c, np.nan)

    r["has_pending_review"] = int(r.get("match_confidence") == "pending_review")
    r["review_ids"] = ";".join(review_ids) if review_ids else None
    r["match_id"] = _make_match_id(r)
    return {k: r.get(k) for k in FINAL_COLUMNS}


def row_from_td_only(td_row, confidence="single_source", review_id=None):
    r = {c: None for c in FINAL_COLUMNS}
    r["sources"] = "tennis_data"
    r["tourney_name"] = td_row.get("Tournament")
    r["tourney_date"] = td_row.get("Date")
    r["surface"] = td_row.get("Surface")
    r["court"] = td_row.get("Court")
    r["series"] = td_row.get("Series")
    # réexprimé dans la nomenclature Sackmann/TML ('1st Round' -> 'R32'),
    # pour que la colonne `round` reste homogène quelle que soit l'origine
    # de la ligne.
    r["round"] = nm.atp_round_from_rank(td_row.get("round_rank"), td_row.get("is_rr", False))
    r["best_of"] = td_row.get("Best of")
    r["score"] = td_row.get("score")
    r["winner_name"] = td_row.get("Winner")
    r["winner_rank"] = td_row.get("WRank")
    r["winner_rank_points"] = td_row.get("WPts")
    r["loser_name"] = td_row.get("Loser")
    r["loser_rank"] = td_row.get("LRank")
    r["loser_rank_points"] = td_row.get("LPts")
    r["match_comment"] = td_row.get("Comment")
    r["odds_w_b365"], r["odds_l_b365"] = td_row.get("B365W"), td_row.get("B365L")
    r["odds_w_ps"], r["odds_l_ps"] = td_row.get("PSW"), td_row.get("PSL")
    r["odds_w_max"], r["odds_l_max"] = td_row.get("MaxW"), td_row.get("MaxL")
    r["odds_w_avg"], r["odds_l_avg"] = td_row.get("AvgW"), td_row.get("AvgL")
    r["match_confidence"] = confidence
    r["has_pending_review"] = int(confidence == "pending_review")
    r["review_ids"] = review_id
    r["match_id"] = _make_match_id(r)
    return r
