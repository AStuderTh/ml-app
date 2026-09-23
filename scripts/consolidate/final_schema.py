"""Schéma de la table `matches` produite par la consolidation.

Deux colonnes de date, volontairement distinctes — les confondre était l'un
des défauts de l'ancien schéma, où `tourney_date` portait tantôt la date de
début du tournoi (lignes Sackmann/TML) tantôt la date réelle du match (lignes
tennis-data.co), ce qui rendait tout tri chronologique et tout découpage
train/test temporel incohérents:

  tourney_date : date de DÉBUT du tournoi. Identique pour tous les matchs
                 d'une même édition. C'est la clé de tri chronologique et la
                 date d'observation des classements (sans fuite: ce qui est
                 publié le lundi est connu avant le 1er tour).
  match_date   : date réelle du match, quand tennis-data.co la fournit.
                 NULL sinon. Jamais utilisée pour trier.
"""
import hashlib

import pandas as pd

from .odds_attach import ODDS_COLUMNS
from .rankings import RANKING_COLUMNS

STAT_COLUMNS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]

FINAL_COLUMNS = (
    [
        "match_id", "match_uid", "sources", "match_confidence",
        "has_pending_odds", "odds_review_id",
        "tourney_id", "tourney_name", "tourney_date", "match_date",
        "surface", "indoor", "court", "series",
        "tourney_level", "draw_size", "round", "best_of",
        "winner_name", "winner_player_id", "winner_id_sackmann", "winner_id_tml",
        "winner_seed", "winner_entry", "winner_hand", "winner_ht", "winner_ioc",
        "winner_age", "winner_rank", "winner_rank_points",
        "loser_name", "loser_player_id", "loser_id_sackmann", "loser_id_tml",
        "loser_seed", "loser_entry", "loser_hand", "loser_ht", "loser_ioc",
        "loser_age", "loser_rank", "loser_rank_points",
    ]
    + RANKING_COLUMNS
    + ["score", "minutes", "match_comment"]
    + STAT_COLUMNS
    + ODDS_COLUMNS
)

# Une ligne = un match Sackmann (ou un match TML d'un tournoi que Sackmann ne
# couvre pas du tout). `match_confidence` ne qualifie donc plus l'existence du
# match — jamais douteuse — mais la richesse de ses sources.
CONFIDENCE_BY_SOURCES = {
    "sackmann,tml": "high",      # les deux référentiels concordent sur clé exacte
    "sackmann": "medium",        # Sackmann seul: pas de contrepartie TML trouvée
    "tml": "provisional",        # tournoi pas encore publié par Sackmann
}


def make_match_id(df: pd.DataFrame) -> pd.Series:
    """Identifiant public, dérivé du contenu du match.

    Le score entre dans la clé pour distinguer deux confrontations réellement
    distinctes entre les mêmes joueurs le même jour/tournoi/tour (formats à
    léges des années 1970).
    """
    parts = []
    for c in ("tourney_id", "round", "winner_name", "loser_name", "tourney_date", "score"):
        parts.append(df[c].where(df[c].notna(), "").astype(str).to_numpy(dtype=object))
    key = ["|".join(t) for t in zip(*parts)]
    return pd.Series([hashlib.md5(k.encode("utf-8")).hexdigest()[:16] for k in key],
                     index=df.index)


def confidence_from_sources(sources: pd.Series) -> pd.Series:
    return sources.map(CONFIDENCE_BY_SOURCES).fillna("medium")
