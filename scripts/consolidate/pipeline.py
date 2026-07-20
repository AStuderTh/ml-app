import pandas as pd

from . import normalize as nm
from . import review_io
from .db_writer import write_sqlite
from .final_schema import finalize_row, row_from_td_only
from .loaders import load_sackmann, load_tml, load_tennisdata
from .matching import match_sackmann_tml, match_with_tennisdata
from .pool_schema import build_pool


def run(data_dir: str, review_csv_path: str, sqlite_path: str, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    log("Chargement des sources...")
    sack = load_sackmann(data_dir)
    tml = load_tml(data_dir)
    td = load_tennisdata(data_dir)
    log(f"  Sackmann (tour principal): {len(sack)} matchs")
    log(f"  TML (tour principal):      {len(tml)} matchs")
    log(f"  tennis-data.co:            {len(td)} matchs")

    existing_decisions = review_io.get_decisions_for_run(review_csv_path)
    log(f"  Décisions déjà saisies dans {review_csv_path}: {len(existing_decisions)}")

    log("Étape 1/2 : appariement Sackmann <-> TML...")
    m1 = match_sackmann_tml(sack, tml)
    log(f"  auto: {len(m1.auto_pairs)}  a_valider: {len(m1.review_pairs)}  "
        f"sackmann_seul: {len(m1.unmatched_a)}  tml_seul: {len(m1.unmatched_b)}")

    pool, review_rows_1 = build_pool(sack, tml, m1, existing_decisions, review_io.make_review_id)
    pool["round_rank"] = pool["round"].map(nm.atp_round_rank)

    log("Étape 2/2 : appariement (Sackmann+TML) <-> tennis-data.co (cotes)...")
    m2 = match_with_tennisdata(pool, td)
    log(f"  auto: {len(m2.auto_pairs)}  a_valider: {len(m2.review_pairs)}  "
        f"sans_cotes: {len(m2.unmatched_a)}  tennisdata_seul: {len(m2.unmatched_b)}")

    final_rows = []
    review_rows_2 = []

    for i, j, score, reason in m2.auto_pairs:
        pr = pool.loc[i].to_dict()
        tr = td.loc[j].to_dict()
        tr["_stage2_confidence"] = "medium"
        final_rows.append(finalize_row(pr, tr))

    for i, j, score, reason in m2.review_pairs:
        pr_row, td_row = pool.loc[i], td.loc[j]
        rid = review_io.make_review_id("matches_vs_tennisdata", pr_row["pool_id"], td_row["src_row_id"])
        decision = existing_decisions.get(rid)
        review_rows_2.append({
            "review_id": rid, "stage": "matches_vs_tennisdata", "reason": reason, "score": round(score, 3),
            "a_source": pr_row["sources"], "a_id": pr_row["pool_id"], "a_tourney": pr_row["tourney_name"],
            "a_date": pr_row["tourney_date"], "a_round": pr_row["round"], "a_winner": pr_row["winner_name"],
            "a_loser": pr_row["loser_name"], "a_score": pr_row["score"],
            "b_source": "tennis_data", "b_id": td_row["src_row_id"], "b_tourney": td_row["Tournament"],
            "b_date": td_row["Date"], "b_round": td_row["Round"], "b_winner": td_row["Winner"],
            "b_loser": td_row["Loser"], "b_score": None,
        })
        if decision == "merge":
            trd = td_row.to_dict()
            trd["_stage2_confidence"] = "manual_confirmed"
            trd["review_id"] = rid
            final_rows.append(finalize_row(pr_row.to_dict(), trd))
        elif decision == "reject":
            final_rows.append(finalize_row(pr_row.to_dict(), None))
            final_rows.append(row_from_td_only(td_row.to_dict(), confidence="single_source", review_id=rid))
        else:
            pr_pending = dict(pr_row)
            pr_pending["match_confidence"] = "pending_review"
            pr_pending["review_id"] = rid
            final_rows.append(finalize_row(pr_pending, None))
            final_rows.append(row_from_td_only(td_row.to_dict(), confidence="pending_review", review_id=rid))

    for i in m2.unmatched_a:
        final_rows.append(finalize_row(pool.loc[i].to_dict(), None))
    for j in m2.unmatched_b:
        final_rows.append(row_from_td_only(td.loc[j].to_dict()))

    matches_df = pd.DataFrame(final_rows)

    dup_mask = matches_df["match_id"].duplicated(keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        log(f"  Attention: {n_dup} match_id en collision malgré tout (repli), désambiguïsation par suffixe.")
        counters = {}
        new_ids = []
        for mid in matches_df["match_id"]:
            n = counters.get(mid, 0)
            counters[mid] = n + 1
            new_ids.append(mid if n == 0 else f"{mid}-{n}")
        matches_df["match_id"] = new_ids

    all_review_rows = review_rows_1 + review_rows_2
    review_df = review_io.write_review_csv(review_csv_path, all_review_rows, existing_decisions)
    log(f"Fichier de validation manuelle écrit: {review_csv_path} ({len(review_df)} cas)")

    write_sqlite(sqlite_path, matches_df, review_df)
    log(f"Base consolidée écrite: {sqlite_path} ({len(matches_df)} matchs)")

    return matches_df, review_df
