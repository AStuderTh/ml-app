"""Chargement des 3 sources, restreintes au tour principal ATP (simples), avec
les colonnes normalisées nécessaires à l'appariement (voir normalize.py).

Portée volontairement exclue de toutes les sources: qualifications,
Challengers, Futures, double, Davis Cup — tennis-data.co ne couvre que le
tour principal, donc c'est le seul périmètre où la fusion à 3 sources (stats
+ cotes) a du sens.
"""
import glob
import os
import re

import pandas as pd

from . import normalize as nm

SACKMANN_MAIN_LEVELS = {"G", "M", "A", "F"}
TML_MAIN_LEVELS = {"G", "M", "A", "F", "250", "500"}


def _drop_exact_duplicates(df, key_cols):
    """Certaines sources contiennent de vrais doublons de ligne (même match
    saisi deux fois, ex. TML 2022 Gijon Paul-Landaluce en match_num 1 et 5
    avec un score identique). On ne déduplique QUE sur une clé incluant le
    score, pour ne jamais fusionner deux confrontations réellement
    distinctes entre les 2 mêmes joueurs le même jour (ex. les doubles
    léges du Champions Classic 1971, scores différents)."""
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    removed = before - len(df)
    return df, removed


def _add_name_fields(df, winner_col="winner_name", loser_col="loser_name"):
    df["wn"] = df[winner_col].map(nm.norm_name_full)
    df["ln"] = df[loser_col].map(nm.norm_name_full)
    df["w_surname"] = df["wn"].map(nm.surname_guess)
    df["l_surname"] = df["ln"].map(nm.surname_guess)
    df["w_ginit"] = df.apply(lambda r: nm.given_initials(r["wn"], r["w_surname"]), axis=1)
    df["l_ginit"] = df.apply(lambda r: nm.given_initials(r["ln"], r["l_surname"]), axis=1)
    return df


def load_sackmann(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "tennis_atp", "atp_matches_[0-9][0-9][0-9][0-9].csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["tourney_level"].isin(SACKMANN_MAIN_LEVELS)].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df["tid_norm"] = df["tourney_id"].map(nm.norm_tourney_id)
    df["round_rank"] = df["round"].map(nm.atp_round_rank)
    df = _add_name_fields(df)
    df, n_dup = _drop_exact_duplicates(df, ["tid_norm", "round", "wn", "ln", "score"])
    if n_dup:
        print(f"  [sackmann] {n_dup} lignes en doublon exact retirées")
    df["src_row_id"] = "sack:" + df.index.astype(str)
    df["source"] = "sackmann"
    return df.reset_index(drop=True)


def load_tml(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "TML-Database", "[0-9][0-9][0-9][0-9].csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["tourney_level"].isin(TML_MAIN_LEVELS)].copy()
    # rejette les ids Davis Cup mal filtrés par tourney_level (format non numérique)
    df = df[df["tourney_id"].astype(str).str.match(r"^\d{4}-\d+$")].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df["tid_norm"] = df["tourney_id"].map(nm.norm_tourney_id)
    df["round_rank"] = df["round"].map(nm.atp_round_rank)
    df = _add_name_fields(df)
    df, n_dup = _drop_exact_duplicates(df, ["tid_norm", "round", "wn", "ln", "score"])
    if n_dup:
        print(f"  [tml] {n_dup} lignes en doublon exact retirées")
    df["src_row_id"] = "tml:" + df.index.astype(str)
    df["source"] = "tml"
    return df.reset_index(drop=True)


_TD_ROUND_MAP_KEEP = set(nm.TD_ROUND_LABELS_IN_ORDER) | nm.TD_ROUND_SPECIAL


def load_tennisdata(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "tennis-data.co", "*.xls*")))
    frames = []
    for f in files:
        d = pd.read_excel(f)
        d["__source_file"] = os.path.basename(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Round"].isin(_TD_ROUND_MAP_KEEP)].copy()
    df["round_rank"] = df["Round"].map(nm.td_round_rank)
    df["is_rr"] = df["Round"] == "Round Robin"

    parsed_w = df["Winner"].map(nm.parse_abbrev_name)
    parsed_l = df["Loser"].map(nm.parse_abbrev_name)
    df["w_surname"] = parsed_w.map(lambda t: nm.norm_name_full(t[0]))
    df["w_initials"] = parsed_w.map(lambda t: t[1])
    df["l_surname"] = parsed_l.map(lambda t: nm.norm_name_full(t[0]))
    df["l_initials"] = parsed_l.map(lambda t: t[1])

    df["tourney_name_norm"] = df["Tournament"].map(nm.norm_tourney_name)
    df["year"] = df["Date"].dt.year
    df, n_dup = _drop_exact_duplicates(df, ["tourney_name_norm", "Date", "Round", "Winner", "Loser"])
    if n_dup:
        print(f"  [tennis_data] {n_dup} lignes en doublon exact retirées")
    df["src_row_id"] = "td:" + df.index.astype(str)
    df["source"] = "tennis_data"
    return df.reset_index(drop=True)
