"""File de validation manuelle: un CSV unique, ré-écrit à chaque exécution,
mais qui conserve les décisions déjà saisies par l'utilisateur (identifiées
par un id stable basé sur le contenu, pas la position dans le fichier).

Colonne `decision` à remplir à la main:
  - "merge"  : fusionner les deux enregistrements
  - "reject" : ce ne sont PAS le même match, les garder séparés
  - (vide)   : pas encore tranché -> les deux lignes restent séparées dans
               la sortie, marquées has_pending_review=1, jusqu'à la prochaine
               exécution après renseignement de cette colonne.
"""
import hashlib
import os

import pandas as pd

REVIEW_COLUMNS = [
    "review_id", "decision", "stage", "reason", "score",
    "a_source", "a_id", "a_tourney", "a_date", "a_round", "a_winner", "a_loser", "a_score",
    "b_source", "b_id", "b_tourney", "b_date", "b_round", "b_winner", "b_loser", "b_score",
]

VALID_DECISIONS = {"merge", "reject"}


def make_review_id(stage: str, a_id: str, b_id: str) -> str:
    h = hashlib.md5(f"{stage}|{a_id}|{b_id}".encode("utf-8")).hexdigest()[:12]
    return h


def load_existing_decisions(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    prev = pd.read_csv(path, dtype=str, keep_default_na=False)
    decisions = {}
    for _, row in prev.iterrows():
        d = str(row.get("decision", "")).strip().lower()
        if d in VALID_DECISIONS:
            decisions[row["review_id"]] = d
    return decisions


def write_review_csv(path: str, rows: list, existing_decisions: dict = None):
    existing_decisions = existing_decisions or {}
    for r in rows:
        r["decision"] = existing_decisions.get(r["review_id"], "")
    df = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    df = df.sort_values(["stage", "a_tourney", "a_date"], na_position="last")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def get_decisions_for_run(path: str) -> dict:
    """Décisions à appliquer pour la construction en cours (relit le fichier
    tel quel, avant qu'on le ré-écrive)."""
    return load_existing_decisions(path)
