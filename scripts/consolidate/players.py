"""Pont d'identité joueur vers le référentiel Sackmann (atp_players.csv).

C'est la pièce qui rend exploitables les 3,4 M de lignes de classement
hebdomadaire: elles sont indexées par `player_id` Sackmann, jamais par nom.
Sans ce pont, chaque orthographe ('Alcaraz C.' vs 'Carlos Alcaraz') est un
joueur distinct pour les features Elo/H2H, et aucun classement officiel ne
peut être rattaché à un match.

Les lignes issues de Sackmann portent déjà `winner_id`/`loser_id`: elles sont
résolues sans ambiguïté possible. Seules les lignes créées depuis TML doivent
passer par le nom — et TML écrit les noms au MÊME format complet que Sackmann
('Carlos Alcaraz'), ce qui rend cette résolution fiable, contrairement au
format abrégé de tennis-data.co.
"""
import os

import pandas as pd

from . import normalize as nm


def load_players(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "tennis_atp", "atp_players.csv")
    df = pd.read_csv(path, low_memory=False)
    df["name_display"] = (df["name_first"].fillna("").astype(str) + " "
                          + df["name_last"].fillna("").astype(str)).str.strip()
    df["name_norm"] = df["name_display"].map(nm.norm_name_full)
    # dob est stocké en float YYYYMMDD (19131122.0) — inutilisable tel quel
    # pour un calcul d'âge.
    df["dob"] = pd.to_datetime(
        df["dob"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return df[["player_id", "name_display", "name_norm", "hand", "dob", "ioc", "height"]]


class PlayerResolver:
    """Résout un nom complet normalisé en `player_id` Sackmann.

    842 noms du référentiel sont portés par plusieurs joueurs (homonymes
    répartis sur des décennies différentes). Trois niveaux de désambiguïsation,
    du plus sûr au moins sûr:
      1. le nom est unique dans le référentiel      -> résolu
      2. le couple (nom, nationalité) est unique    -> résolu
      3. un seul des homonymes a déjà joué sur le circuit principal -> résolu
    Au-delà, on renonce (None) plutôt que de risquer de fusionner deux
    carrières distinctes: un `player_id` faux est bien pire qu'un `player_id`
    manquant, car il contamine silencieusement l'Elo et les classements.
    """

    def __init__(self, players: pd.DataFrame, known_ids=None):
        self.players = players
        known_ids = set(known_ids or ())

        by_name = players.groupby("name_norm")["player_id"].apply(list)
        self._unique = {n: ids[0] for n, ids in by_name.items() if len(ids) == 1}
        self._ambiguous = {n: ids for n, ids in by_name.items() if len(ids) > 1}

        amb = players[players["name_norm"].isin(self._ambiguous)]
        self._by_name_ioc = {}
        for (n, ioc), grp in amb.groupby(["name_norm", "ioc"]):
            if len(grp) == 1:
                self._by_name_ioc[(n, ioc)] = grp["player_id"].iloc[0]

        # dernier recours: parmi les homonymes, ceux qui ont réellement disputé
        # un match du tour principal (donc présents dans le corpus Sackmann).
        self._by_name_active = {}
        for n, ids in self._ambiguous.items():
            seen = [i for i in ids if i in known_ids]
            if len(seen) == 1:
                self._by_name_active[n] = seen[0]

        self.stats = {"unique": 0, "by_ioc": 0, "by_active": 0, "unresolved": 0}

    def resolve(self, name_norm: str, ioc=None):
        if not name_norm:
            self.stats["unresolved"] += 1
            return None
        pid = self._unique.get(name_norm)
        if pid is not None:
            self.stats["unique"] += 1
            return pid
        if ioc is not None and pd.notna(ioc):
            pid = self._by_name_ioc.get((name_norm, ioc))
            if pid is not None:
                self.stats["by_ioc"] += 1
                return pid
        pid = self._by_name_active.get(name_norm)
        if pid is not None:
            self.stats["by_active"] += 1
            return pid
        self.stats["unresolved"] += 1
        return None


def canonical_name_map(players: pd.DataFrame) -> dict:
    """player_id -> nom d'affichage canonique ('Carlos Alcaraz').

    Sert à réécrire `winner_name`/`loser_name` de façon homogène quelle que
    soit la source de la ligne: les features Elo/forme/H2H s'indexent sur ces
    chaînes (cf. app/features.py), donc deux orthographes du même joueur y
    créent deux ratings distincts.
    """
    return dict(zip(players["player_id"], players["name_display"]))
