"""Consolidation des sources en une base unique prête pour le ML.

Hiérarchie des sources — et non fusion de sources pairs:

    Sackmann  : colonne vertébrale. Seul à pouvoir définir un match, seul à
                porter l'identité joueur (`player_id`), qui donne accès aux
                3,4 M de lignes de classement hebdomadaire.
    TML        : enrichissement sur clé EXACTE (`indoor`, trous comblés) +
                 reprise des tournois entièrement absents de Sackmann (la
                 saison en cours, où TML a ~2 mois d'avance).
    tennis-data: cotes uniquement. Ne crée jamais de ligne.

La règle qui gouverne tout: une comparaison incertaine laisse une colonne à
NULL, elle ne crée jamais de ligne. C'est ce qui garantit qu'un match réel
n'apparaît qu'une fois — condition sine qua non pour que l'Elo, la forme et
le H2H ne comptent pas deux fois le même résultat.
"""
import numpy as np
import pandas as pd

from . import normalize as nm
from . import players as pl
from . import rankings as rk
from . import review_io
from .db_writer import write_sqlite
from .final_schema import FINAL_COLUMNS, confidence_from_sources, make_match_id
from .loaders import load_sackmann, load_tennisdata, load_tml
from .odds_attach import attach_odds
from .spine import build_spine


def _resolve_identities(spine: pd.DataFrame, players: pd.DataFrame, log):
    """Rattache chaque camp à un `player_id` Sackmann et homogénéise les noms.

    Les lignes Sackmann portent déjà l'id: aucune ambiguïté. Seules les lignes
    reprises de TML passent par le nom — format complet identique à Sackmann,
    donc résolution fiable.
    """
    known = set(pd.concat([
        spine["winner_id_sackmann"], spine["loser_id_sackmann"]]).dropna().astype("int64"))
    resolver = pl.PlayerResolver(players, known_ids=known)

    for side in ("winner", "loser"):
        # travail sur un tableau float64 brut: une affectation pandas de liste
        # contenant des None dans une colonne float lève une LossySetitemError.
        values = pd.to_numeric(
            spine[f"{side}_id_sackmann"], errors="coerce").to_numpy(dtype="float64").copy()
        need = np.isnan(values)
        if need.any():
            sub = spine.loc[need]
            key = "wn" if side == "winner" else "ln"
            ioc_col = f"{side}_ioc"
            iocs = sub[ioc_col] if ioc_col in sub.columns else [None] * len(sub)
            got = [resolver.resolve(n, i) for n, i in zip(sub[key], iocs)]
            values[need] = [np.nan if g is None else float(g) for g in got]
        spine[f"{side}_player_id"] = (
            pd.Series(values, index=spine.index).astype("Float64").astype("Int64"))

    s = resolver.stats
    log(f"  identités résolues par nom: {s['unique']} directes, {s['by_ioc']} par nationalité, "
        f"{s['by_active']} par activité, {s['unresolved']} non résolues")

    # Nom d'affichage canonique: les features Elo/forme/H2H s'indexent sur ces
    # chaînes, deux orthographes du même joueur y créeraient deux ratings.
    canon = pl.canonical_name_map(players)
    n_fixed = 0
    for side in ("winner", "loser"):
        ids = spine[f"{side}_player_id"]
        names = spine[f"{side}_name"].astype(object)
        mapped = ids.map(lambda v: canon.get(int(v)) if pd.notna(v) else None)
        take = mapped.notna() & (mapped != names)
        spine.loc[take, f"{side}_name"] = mapped[take]
        n_fixed += int(take.sum())
    log(f"  noms réécrits sous leur forme canonique: {n_fixed}")
    return spine


def run(data_dir: str, review_csv_path: str, sqlite_path: str, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    log("Chargement des sources...")
    sack = load_sackmann(data_dir)
    tml = load_tml(data_dir)
    td = load_tennisdata(data_dir)
    players = pl.load_players(data_dir)
    log(f"  Sackmann: {len(sack)} matchs | TML: {len(tml)} | tennis-data.co: {len(td)}"
        f" | référentiel joueurs: {len(players)}")

    existing_decisions = review_io.get_decisions_for_run(review_csv_path)
    log(f"  Décisions déjà saisies dans {review_csv_path}: {len(existing_decisions)}")

    log("Étape 1/5 — colonne vertébrale (Sackmann, enrichi par TML)...")
    spine, spine_stats = build_spine(sack, tml, log=log)
    spine["match_uid"] = spine["src_row_id"]

    log("Étape 2/5 — identité joueur et noms canoniques...")
    spine = _resolve_identities(spine, players, log)

    log("Étape 3/5 — attachement des cotes tennis-data.co...")
    odds, review_rows, match_date, odds_stats = attach_odds(
        spine, td, existing_decisions, log=log)
    # date réelle du match: tennis-data.co d'abord (la plus fiable), à défaut
    # celle que TML porte pour les tournois de la saison en cours.
    spine["match_date"] = match_date.fillna(spine["match_date_src"])
    for col in odds.columns:
        spine[col] = odds[col].to_numpy()

    log("Étape 4/5 — classements ATP officiels (as-of, sans fuite)...")
    ranks = rk.load_rankings(data_dir)
    log(f"  {len(ranks)} lignes de classement hebdomadaire chargées "
        f"({ranks['ranking_date'].min():%Y-%m-%d} -> {ranks['ranking_date'].max():%Y-%m-%d})")
    spine = rk.attach_rankings(spine, ranks, date_col="tourney_date")
    cov = spine["winner_atp_rank"].notna().mean()
    log(f"  classement officiel disponible pour {cov:.1%} des vainqueurs "
        f"(colonne embarquée d'origine: {spine['winner_rank'].notna().mean():.1%})")

    log("Étape 5/5 — assemblage et écriture...")
    spine["sources"] = np.where(
        spine[[c for c in ("odds_w_b365", "odds_w_avg", "odds_w_ps", "odds_w_max")]].notna().any(axis=1),
        spine["sources"] + ",tennis_data", spine["sources"])
    spine["match_confidence"] = confidence_from_sources(
        spine["sources"].str.replace(",tennis_data", "", regex=False))
    spine["match_id"] = make_match_id(spine)

    matches_df = spine.reindex(columns=FINAL_COLUMNS)

    # Ordre chronologique réel, jusqu'à l'intérieur d'un tournoi.
    # `tourney_date` est la date de DÉBUT du tournoi: identique pour tous les
    # matchs d'une même édition, elle ne suffit donc pas à les ordonner. Sans
    # ce tri, les features chronologiques calculées en aval — Elo, forme, H2H —
    # verraient le résultat de la finale AVANT celui du 1er tour du même
    # tournoi, et fuiteraient donc du futur dans le passé.
    matches_df["_rank"] = matches_df["round"].map(nm.atp_round_rank)
    matches_df = matches_df.sort_values(
        ["tourney_date", "_rank"], ascending=[True, False], kind="stable",
    ).drop(columns="_rank").reset_index(drop=True)

    dup = matches_df["match_id"].duplicated(keep=False)
    if dup.any():
        log(f"  {int(dup.sum())} match_id en collision, désambiguïsation par suffixe.")
        counters, new_ids = {}, []
        for mid in matches_df["match_id"]:
            n = counters.get(mid, 0)
            counters[mid] = n + 1
            new_ids.append(mid if n == 0 else f"{mid}-{n}")
        matches_df["match_id"] = new_ids

    review_df = review_io.write_review_csv(review_csv_path, review_rows, existing_decisions)
    log(f"  file de validation (cotes ambiguës uniquement): {len(review_df)} cas")

    write_sqlite(sqlite_path, matches_df, review_df)
    log(f"Base écrite: {sqlite_path} ({len(matches_df)} matchs)")

    stats = {**spine_stats, **odds_stats}
    log("Récapitulatif: " + " | ".join(f"{k}={v}" for k, v in stats.items()))
    return matches_df, review_df
