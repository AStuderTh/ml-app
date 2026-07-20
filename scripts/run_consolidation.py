"""Point d'entrée: fusionne tennis_atp (Sackmann), TML-Database et
tennis-data.co en une base unique, dédupliquée, prête pour du ML.

Usage:
    python scripts/run_consolidation.py

Sorties:
    data/tennis.db          table `matches` (propre, dédupliquée) + `match_review`
    data/match_review.csv   cas ambigus/à faible confiance à valider à la main

Pour valider un cas ambigu: ouvrir data/match_review.csv, remplir la colonne
`decision` avec "merge" (même match) ou "reject" (matchs différents), puis
relancer ce script. Les décisions déjà saisies sont conservées d'une
exécution à l'autre (identifiées par contenu, pas par position).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.consolidate.pipeline import run

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    review_csv = os.path.join(data_dir, "match_review.csv")
    sqlite_path = os.path.join(data_dir, "tennis.db")
    run(data_dir, review_csv, sqlite_path)
