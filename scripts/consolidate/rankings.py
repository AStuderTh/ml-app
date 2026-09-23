"""Classements ATP hebdomadaires officiels (atp_rankings_*.csv, ~3,4 M lignes).

Apport par rapport aux colonnes `winner_rank`/`loser_rank` déjà présentes dans
les fichiers de matchs:

  - couverture: la colonne embarquée dans les matchs est un instantané rempli
    à ~82 %; le référentiel hebdomadaire couvre tout joueur classé à toute
    date, y compris quand la ligne de match ne le porte pas.
  - antériorité: on peut lire le classement à N semaines AVANT le match, donc
    construire une trajectoire (progression/chute), impossible avec un simple
    instantané.

La jointure est un `merge_asof` en arrière: pour chaque joueur on prend le
dernier classement publié À OU AVANT la date du match. Jamais le suivant —
ce serait lire le résultat du match dans sa propre feature.
"""
import glob
import os

import pandas as pd

# Recul utilisé pour la trajectoire de classement. 52 semaines: compare le
# joueur à lui-même à la même période de la saison précédente, ce qui neutralise
# la saisonnalité du calendrier ATP (les points se défendent à date anniversaire).
TRAJECTORY_WEEKS = 52


def load_rankings(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "tennis_atp", "atp_rankings_*.csv")))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ranking_date", "player"])
    df["player"] = df["player"].astype("int64")
    # atp_rankings_current.csv recoupe partiellement la décennie en cours:
    # deux lignes pour un même (joueur, date) casseraient le merge_asof.
    df = df.drop_duplicates(subset=["player", "ranking_date"], keep="last")
    return df.sort_values("ranking_date", kind="stable").reset_index(drop=True)


def _asof(left: pd.DataFrame, rk: pd.DataFrame, date_col: str, id_col: str,
          suffix: str) -> pd.DataFrame:
    """merge_asof en arrière (dernier classement publié <= date du match)."""
    l = left[[id_col, date_col]].copy()
    l["_pos"] = range(len(l))
    l = l.dropna(subset=[id_col, date_col])
    if l.empty:
        return pd.DataFrame(index=left.index,
                            columns=[f"rank{suffix}", f"points{suffix}"], dtype="float64")
    l[id_col] = l[id_col].astype("int64")
    l = l.sort_values(date_col, kind="stable")
    merged = pd.merge_asof(
        l, rk, left_on=date_col, right_on="ranking_date",
        left_by=id_col, right_by="player", direction="backward",
    )
    out = pd.DataFrame(index=left.index, columns=[f"rank{suffix}", f"points{suffix}"],
                       dtype="float64")
    pos = merged["_pos"].to_numpy()
    out.iloc[pos, 0] = merged["rank"].to_numpy()
    out.iloc[pos, 1] = merged["points"].to_numpy()
    return out


def attach_rankings(matches: pd.DataFrame, rankings: pd.DataFrame,
                    date_col: str = "match_date") -> pd.DataFrame:
    """Ajoute, pour chaque camp, le classement/points officiels à la date du
    match et leur valeur 52 semaines plus tôt.

    Colonnes produites: {winner,loser}_atp_rank, _atp_points,
    _atp_rank_prev, _atp_points_prev.
    """
    df = matches.copy()
    back = pd.Timedelta(weeks=TRAJECTORY_WEEKS)
    df["_date_prev"] = pd.to_datetime(df[date_col]) - back

    for side in ("winner", "loser"):
        id_col = f"{side}_player_id"
        now = _asof(df, rankings, date_col, id_col, "_now")
        prev = _asof(df, rankings, "_date_prev", id_col, "_prev")
        df[f"{side}_atp_rank"] = now["rank_now"].to_numpy()
        df[f"{side}_atp_points"] = now["points_now"].to_numpy()
        df[f"{side}_atp_rank_prev"] = prev["rank_prev"].to_numpy()
        df[f"{side}_atp_points_prev"] = prev["points_prev"].to_numpy()

    return df.drop(columns="_date_prev")


RANKING_COLUMNS = [
    "winner_atp_rank", "winner_atp_points", "winner_atp_rank_prev", "winner_atp_points_prev",
    "loser_atp_rank", "loser_atp_points", "loser_atp_rank_prev", "loser_atp_points_prev",
]
