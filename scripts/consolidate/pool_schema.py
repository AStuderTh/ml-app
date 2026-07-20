"""Construit le "pool" intermédiaire (avant ajout des cotes tennis-data.co)
en fusionnant, ligne à ligne, les enregistrements Sackmann/TML appariés.
Politique de priorité en cas de conflit de valeur: Sackmann d'abord (source
la plus ancienne/complète), puis TML pour combler les trous.
"""
import pandas as pd

from . import normalize as nm

SHARED_STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]
SHARED_META_COLS = [
    "tourney_name", "tourney_date", "surface", "draw_size", "tourney_level",
    "round", "best_of", "score", "minutes",
    "winner_name", "winner_seed", "winner_entry", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_rank", "winner_rank_points",
    "loser_name", "loser_seed", "loser_entry", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_rank", "loser_rank_points",
]


def _c(a, b):
    if a is not None and pd.notna(a) and a != "":
        return a
    return b


def pool_row_from_pair(sack_row, tml_row, confidence, reason, review_id=None):
    r = {}
    for col in SHARED_META_COLS + SHARED_STAT_COLS:
        r[col] = _c(sack_row.get(col) if sack_row is not None else None,
                     tml_row.get(col) if tml_row is not None else None)
    r["tourney_id"] = _c(sack_row.get("tourney_id") if sack_row is not None else None,
                          tml_row.get("tourney_id") if tml_row is not None else None)
    r["indoor"] = tml_row.get("indoor") if tml_row is not None else None
    r["winner_id_sackmann"] = sack_row.get("winner_id") if sack_row is not None else None
    r["loser_id_sackmann"] = sack_row.get("loser_id") if sack_row is not None else None
    r["winner_id_tml"] = tml_row.get("winner_id") if tml_row is not None else None
    r["loser_id_tml"] = tml_row.get("loser_id") if tml_row is not None else None

    sources = []
    src_ids = []
    if sack_row is not None:
        sources.append("sackmann")
        src_ids.append(sack_row["src_row_id"])
    if tml_row is not None:
        sources.append("tml")
        src_ids.append(tml_row["src_row_id"])
    r["sources"] = ",".join(sources)
    r["pool_id"] = "+".join(src_ids)
    r["match_confidence"] = confidence
    r["match_reason"] = reason
    r["review_id"] = review_id

    r["wn"] = _c(sack_row.get("wn") if sack_row is not None else None,
                 tml_row.get("wn") if tml_row is not None else None)
    r["ln"] = _c(sack_row.get("ln") if sack_row is not None else None,
                  tml_row.get("ln") if tml_row is not None else None)
    r["w_ginit"] = _c(sack_row.get("w_ginit") if sack_row is not None else None,
                       tml_row.get("w_ginit") if tml_row is not None else None)
    r["l_ginit"] = _c(sack_row.get("l_ginit") if sack_row is not None else None,
                       tml_row.get("l_ginit") if tml_row is not None else None)
    r["round"] = r["round"]  # déjà coalescé au-dessus
    return r


def build_pool(sack: pd.DataFrame, tml: pd.DataFrame, match_result, decisions: dict, review_id_fn):
    """Construit le pool Sackmann+TML fusionné. `decisions` = review_id -> 'merge'/'reject'
    issues d'une exécution précédente. Retourne (pool_df, review_rows_for_csv)."""
    rows = []
    review_rows = []

    for i, j, score, reason in match_result.auto_pairs:
        conf = "high" if reason == "exact_key" else "medium"
        rows.append(pool_row_from_pair(sack.loc[i], tml.loc[j], conf, reason))

    for i, j, score, reason in match_result.review_pairs:
        rid = review_id_fn("sackmann_vs_tml", sack.loc[i, "src_row_id"], tml.loc[j, "src_row_id"])
        decision = decisions.get(rid)
        sa, tb = sack.loc[i], tml.loc[j]
        review_rows.append({
            "review_id": rid, "stage": "sackmann_vs_tml", "reason": reason, "score": round(score, 3),
            "a_source": "sackmann", "a_id": sa["src_row_id"], "a_tourney": sa["tourney_name"],
            "a_date": sa["tourney_date"], "a_round": sa["round"], "a_winner": sa["winner_name"],
            "a_loser": sa["loser_name"], "a_score": sa["score"],
            "b_source": "tml", "b_id": tb["src_row_id"], "b_tourney": tb["tourney_name"],
            "b_date": tb["tourney_date"], "b_round": tb["round"], "b_winner": tb["winner_name"],
            "b_loser": tb["loser_name"], "b_score": tb["score"],
        })
        if decision == "merge":
            rows.append(pool_row_from_pair(sa, tb, "manual_confirmed", reason, review_id=rid))
        elif decision == "reject":
            rows.append(pool_row_from_pair(sa, None, "single_source", "rejected_by_review", review_id=rid))
            rows.append(pool_row_from_pair(None, tb, "single_source", "rejected_by_review", review_id=rid))
        else:
            rows.append(pool_row_from_pair(sa, None, "pending_review", reason, review_id=rid))
            rows.append(pool_row_from_pair(None, tb, "pending_review", reason, review_id=rid))

    for i in match_result.unmatched_a:
        rows.append(pool_row_from_pair(sack.loc[i], None, "single_source", "no_candidate"))
    for j in match_result.unmatched_b:
        rows.append(pool_row_from_pair(None, tml.loc[j], "single_source", "no_candidate"))

    pool = pd.DataFrame(rows)
    pool["tourney_date"] = pd.to_datetime(pool["tourney_date"])
    return pool.reset_index(drop=True), review_rows
