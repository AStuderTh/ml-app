"""Chargement des 3 sources, restreintes au tour principal ATP (simples), avec
les colonnes normalisées nécessaires à l'appariement (voir normalize.py).

Portée volontairement exclue de toutes les sources: qualifications,
Challengers, Futures, double, Davis Cup — tennis-data.co ne couvre que le
tour principal, donc c'est le seul périmètre où la fusion à 3 sources (stats
+ cotes) a du sens.
"""
import glob
import hashlib
import os
import re

import pandas as pd

from . import normalize as nm


def _content_row_id(df: pd.DataFrame, prefix: str, cols: list) -> pd.Series:
    """Identifiant de ligne dérivé du CONTENU du match, jamais de sa position
    dans le fichier source.

    Les dépôts amont réécrivent régulièrement d'anciennes saisons (Sackmann
    republie tout l'historique à chaque fin d'année): une seule ligne
    insérée en 2011 décalerait la position de toutes les suivantes. Comme
    l'id de la file de validation manuelle en dérive, la totalité des
    décisions déjà saisies dans data/match_review.csv serait alors perdue au
    prochain `git pull`."""
    # concaténation colonne par colonne en objets Python: une valeur manquante
    # (score absent, walkover...) doit devenir la chaîne vide et non propager
    # un NaN à toute la clé.
    parts = [df[c].where(df[c].notna(), "").astype(str).to_numpy(dtype=object) for c in cols]
    key = pd.Series(["|".join(t) for t in zip(*parts)], index=df.index)
    h = key.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:12])
    # deux lignes au contenu strictement identique (vrai doublon que la
    # déduplication n'a pas retiré) doivent rester distinguables.
    rank = h.groupby(h).cumcount()
    return prefix + ":" + h + rank.map(lambda n: "" if n == 0 else f"-{n}")

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
    df["src_row_id"] = _content_row_id(
        df, "sack", ["tid_norm", "tourney_date", "round", "wn", "ln", "score"])
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
    df["src_row_id"] = _content_row_id(
        df, "tml", ["tid_norm", "tourney_date", "round", "wn", "ln", "score"])
    df["source"] = "tml"
    return df.reset_index(drop=True)


_TD_ROUND_MAP_KEEP = set(nm.TD_ROUND_LABELS_IN_ORDER) | nm.TD_ROUND_SPECIAL


def _td_score_string(row) -> str:
    """tennis-data.co stocke le score en colonnes par set (W1/L1..W5/L5), pas
    en texte comme Sackmann/TML. On reconstruit une chaîne "6-4 6-3" pour
    permettre la comparaison visuelle dans la file de validation manuelle
    (le score du match est le signal le plus fiable pour trancher un cas
    ambigu). Pas de détail du score du tie-break: l'info n'existe pas dans
    cette source (seulement le nombre de jeux par set)."""
    sets = []
    for n in range(1, 6):
        w = pd.to_numeric(row.get(f"W{n}"), errors="coerce")
        l = pd.to_numeric(row.get(f"L{n}"), errors="coerce")
        if pd.isna(w) or pd.isna(l):
            continue
        sets.append(f"{int(w)}-{int(l)}")
    score = " ".join(sets)
    comment = row.get("Comment")
    if comment in ("Retired", "Walkover"):
        score = f"{score} {comment}".strip()
    return score


def _td_instance_year(df: pd.DataFrame) -> pd.Series:
    """Année de l'ÉDITION du tournoi (= année de son 1er match), et non année
    de chaque match. Sackmann/TML datent un tournoi par sa date de DÉBUT:
    sans ça, une édition à cheval sur le nouvel an (Brisbane, Adelaide,
    United Cup...) est coupée en deux instances côté tennis-data.co, et tout
    ce qui tombe après le 31/12 ne peut plus être relié au tournoi d'en face.

    Les éditions successives d'un même tournoi sont séparées par ~1 an: on
    coupe sur les trous de plus de 60 jours, très au-dessus de la durée d'un
    tournoi (≤ 2 semaines) et très en-dessous de l'écart entre 2 éditions."""
    out = pd.Series(index=df.index, dtype="float64")
    for _, g in df.groupby("tourney_name_norm", sort=False):
        g = g.sort_values("Date")
        instance = (g["Date"].diff() > pd.Timedelta(days=60)).cumsum()
        out.loc[g.index] = g.groupby(instance)["Date"].transform("min").dt.year
    return out


def _td_round_ranks(df: pd.DataFrame) -> pd.Series:
    """Rang-depuis-la-finale de chaque ligne tennis-data.co, calculé PAR
    instance de tournoi (année + nom normalisé): le nombre de tours numérotés
    présents révèle la taille du tableau, que la source ne publie pas
    (cf. nm.td_numbered_round_rank). Sans ça, un "1st Round" d'un ATP 250
    serait pris pour un R128 et ne pourrait jamais s'apparier au R32
    correspondant côté Sackmann/TML."""
    n_numbered = (
        df[df["Round"].isin(nm.TD_NUMBERED_ROUNDS)]
        .groupby(["year", "tourney_name_norm"])["Round"]
        .nunique()
    )
    keys = pd.MultiIndex.from_arrays([df["year"], df["tourney_name_norm"]])
    counts = n_numbered.reindex(keys).fillna(0).astype(int).values
    return [
        nm.td_round_rank(lbl, c) for lbl, c in zip(df["Round"].values, counts)
    ]


def load_tennisdata(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "tennis-data.co", "*.xls*")))
    frames = []
    for f in files:
        d = pd.read_excel(f)
        d["__source_file"] = os.path.basename(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    # la source contient des noms à espaces parasites ('Federer R. '), qui
    # créent sinon une identité de joueur supplémentaire dans la base finale
    # (les lignes sans correspondance y gardent le nom brut).
    for col in ("Winner", "Loser", "Tournament", "Round"):
        df[col] = df[col].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Round"].isin(_TD_ROUND_MAP_KEEP)].copy()
    df["is_rr"] = df["Round"] == "Round Robin"

    parsed_w = df["Winner"].map(nm.parse_abbrev_name)
    parsed_l = df["Loser"].map(nm.parse_abbrev_name)
    df["w_surname"] = parsed_w.map(lambda t: nm.norm_name_full(t[0]))
    df["w_initials"] = parsed_w.map(lambda t: t[1])
    df["l_surname"] = parsed_l.map(lambda t: nm.norm_name_full(t[0]))
    df["l_initials"] = parsed_l.map(lambda t: t[1])

    df["tourney_name_norm"] = df["Tournament"].map(nm.norm_tourney_name)
    df["year"] = _td_instance_year(df)
    df["score"] = df.apply(_td_score_string, axis=1)
    df["round_rank"] = _td_round_ranks(df)
    df, n_dup = _drop_exact_duplicates(df, ["tourney_name_norm", "Date", "Round", "Winner", "Loser"])
    if n_dup:
        print(f"  [tennis_data] {n_dup} lignes en doublon exact retirées")
    df["src_row_id"] = _content_row_id(
        df, "td", ["tourney_name_norm", "Date", "Round", "Winner", "Loser"])
    df["source"] = "tennis_data"
    return df.reset_index(drop=True)
