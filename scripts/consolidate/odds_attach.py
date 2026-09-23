"""Attache les cotes tennis-data.co aux lignes de la colonne vertébrale.

Différence de nature avec l'ancienne étape 2: tennis-data.co n'est plus une
source de MATCHS, seulement une source de COTES. Elle ne peut donc plus créer
de ligne. Conséquences directes sur la qualité de la base:

  - un rapprochement raté coûte une cote manquante, plus un match en double;
  - `winner_name` ne peut plus contenir de forme abrégée ('Alcaraz C.'), donc
    plus de joueur fantôme parallèle dans les features Elo/H2H;
  - la file de validation manuelle ne tranche plus l'EXISTENCE d'un match,
    seulement l'attribution d'une cote — un arbitrage bien moins critique,
    et dont l'absence de réponse est sans danger.
"""
import numpy as np
import pandas as pd

from . import review_io
from .matching import match_with_tennisdata

ODDS_FIELDS = [
    ("odds_w_b365", "odds_l_b365", "B365W", "B365L"),
    ("odds_w_ps", "odds_l_ps", "PSW", "PSL"),
    ("odds_w_max", "odds_l_max", "MaxW", "MaxL"),
    ("odds_w_avg", "odds_l_avg", "AvgW", "AvgL"),
]

CONTEXT_FIELDS = [("court", "Court"), ("series", "Series"), ("match_comment", "Comment")]

ODDS_COLUMNS = [w for w, _, _, _ in ODDS_FIELDS] + [l for _, l, _, _ in ODDS_FIELDS]


def _blank_odds(n_index):
    cols = {c: np.nan for c in ODDS_COLUMNS}
    cols.update({c: None for c, _ in CONTEXT_FIELDS})
    return pd.DataFrame(cols, index=n_index)


def _apply_bulk(odds: pd.DataFrame, match_date: pd.Series, td: pd.DataFrame, accepted):
    """Recopie cotes + contexte pour tous les couples retenus, en une passe.

    `swapped` signale que le vainqueur tennis-data est le PERDANT de la ligne
    de référence (erreur de saisie d'un côté ou de l'autre). Il faut alors
    croiser les colonnes de cotes: sans ça on écrirait la cote du perdant dans
    `odds_w_*`, c'est-à-dire un signal exactement inversé — le pire cas
    possible pour un modèle, qui apprendrait que le favori perd.

    Écriture vectorisée et non cellule par cellule: 70 000 couples × 11
    colonnes font 770 000 affectations `.at`, chacune repassant par le
    gestionnaire de blocs de pandas.
    """
    if not accepted:
        return
    i_lbl = [a for a, _, _ in accepted]
    j_lbl = [b for _, b, _ in accepted]
    sw = np.array([s for _, _, s in accepted], dtype=bool)
    tpos = td.index.get_indexer(j_lbl)

    for w_col, l_col, w_src, l_src in ODDS_FIELDS:
        wv = pd.to_numeric(td[w_src], errors="coerce").to_numpy()[tpos]
        lv = pd.to_numeric(td[l_src], errors="coerce").to_numpy()[tpos]
        odds.loc[i_lbl, w_col] = np.where(sw, lv, wv)
        odds.loc[i_lbl, l_col] = np.where(sw, wv, lv)
    for col, src in CONTEXT_FIELDS:
        odds.loc[i_lbl, col] = td[src].to_numpy()[tpos]
    match_date.loc[i_lbl] = td["Date"].to_numpy()[tpos]


def attach_odds(spine: pd.DataFrame, td: pd.DataFrame, decisions: dict, log=print):
    """Retourne (odds_df, review_rows, match_date, stats).

    `odds_df` est aligné sur l'index de `spine`. `match_date` est la date
    réelle du match quand tennis-data.co la fournit (les sources Sackmann/TML
    ne datent que le DÉBUT du tournoi).
    """
    result = match_with_tennisdata(spine, td)
    log(f"  cotes: auto={len(result.auto_pairs)} a_valider={len(result.review_pairs)} "
        f"lignes_sans_cotes={len(result.unmatched_a)} td_orphelines={len(result.unmatched_b)}")

    odds = _blank_odds(spine.index)
    match_date = pd.Series(pd.NaT, index=spine.index, dtype="datetime64[ns]")
    pending = pd.Series(0, index=spine.index, dtype="int64")
    review_ids = pd.Series(None, index=spine.index, dtype="object")

    n_auto = n_manual = n_swap = 0
    accepted = [(i, j, False) for i, j, _, _ in result.auto_pairs]
    n_auto = len(accepted)

    review_rows = []
    for i, j, score, reason in result.review_pairs:
        sr, tr = spine.loc[i], td.loc[j]
        rid = review_io.make_review_id("odds_vs_tennisdata", str(sr["match_uid"]), str(tr["src_row_id"]))
        review_rows.append({
            "review_id": rid, "stage": "odds_vs_tennisdata", "reason": reason,
            "score": round(score, 3),
            "a_source": sr["sources"], "a_id": sr["match_uid"], "a_tourney": sr["tourney_name"],
            "a_date": sr["tourney_date"], "a_round": sr["round"], "a_winner": sr["winner_name"],
            "a_loser": sr["loser_name"], "a_score": sr["score"],
            "b_source": "tennis_data", "b_id": tr["src_row_id"], "b_tourney": tr["Tournament"],
            "b_date": tr["Date"], "b_round": tr["Round"], "b_winner": tr["Winner"],
            "b_loser": tr["Loser"], "b_score": tr["score"],
        })
        review_ids.at[i] = rid
        decision = decisions.get(rid)
        if decision == "merge":
            swapped = reason == "swapped_winner_loser"
            accepted.append((i, j, swapped))
            n_manual += 1
            n_swap += int(swapped)
        elif decision != "reject":
            # non tranché: la ligne reste, simplement sans cote. Aucun risque.
            pending.at[i] = 1

    _apply_bulk(odds, match_date, td, accepted)
    log(f"  cotes attachées: {n_auto} auto + {n_manual} validées à la main "
        f"(dont {n_swap} avec inversion vainqueur/perdant corrigée)")

    stats = {"n_odds_auto": n_auto, "n_odds_manual": n_manual,
             "n_odds_pending": int(pending.sum()), "n_td_orphan": len(result.unmatched_b)}
    odds["has_pending_odds"] = pending
    odds["odds_review_id"] = review_ids
    return odds, review_rows, match_date, stats
