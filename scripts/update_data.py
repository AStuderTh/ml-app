"""Rafraîchit les 3 sources de données et relance la consolidation.

Usage (CLI):
    python scripts/update_data.py

Étapes:
    1. clonage (ou git pull) de data/tennis_atp et data/TML-Database (dépôts amont)
    2. téléchargement du fichier tennis-data.co.uk de la saison en cours
       (+ année précédente, pour couvrir les tournois à cheval sur le
       nouvel an déjà présents en base mais dont l'année vient d'être
       complétée par la source)
    3. relance scripts/consolidate/pipeline.run(...)

Appelé aussi depuis l'app Streamlit (onglet "Données") via `run_update`,
qui accepte un callback `log` pour afficher la progression dans l'UI.
"""
import datetime as dt
import os
import subprocess
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
GIT_REPOS = {
    "tennis_atp": "https://github.com/Kadantte/tennis_atp.git",
    "TML-Database": "https://github.com/Tennismylife/TML-Database.git",
}
TENNISDATA_BASE_URL = "http://www.tennis-data.co.uk"


def _log(msg, cb=None):
    print(msg)
    if cb:
        cb(msg)


def pull_git_repos(log=None):
    """Clone les sources absentes, sinon les met à jour avec git pull."""
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}
    for name, url in GIT_REPOS.items():
        path = os.path.join(DATA_DIR, name)
        action = "vérification"
        try:
            if not os.path.exists(path):
                proc = subprocess.run(
                    ["git", "clone", url, path],
                    capture_output=True, text=True, timeout=600,
                )
                action = "clone"
            elif os.path.isdir(os.path.join(path, ".git")):
                proc = subprocess.run(
                    ["git", "-C", path, "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=120,
                )
                action = "pull"
            else:
                proc = None
                action = "déjà présent mais invalide"
            if proc is None:
                out = "supprimez ce dossier puis relancez la mise à jour"
                ok = False
            else:
                out = (proc.stdout + proc.stderr).strip()
                ok = proc.returncode == 0
        except Exception as e:
            out, ok = str(e), False
        _log(f"  [{name}] {'OK' if ok else 'ECHEC'} ({action}) — {out or 'déjà à jour'}", log)
        results[name] = {"ok": ok, "action": action, "output": out}
    return results


def download_tennisdata(years=None, log=None):
    """Télécharge/écrase les fichiers ATP de tennis-data.co.uk pour les
    années données (par défaut: année en cours + année précédente, pour
    rattraper les publications tardives de fin de saison précédente)."""
    if years is None:
        current_year = dt.date.today().year
        years = [current_year - 1, current_year]

    out_dir = os.path.join(DATA_DIR, "tennis-data.co")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for year in years:
        url = f"{TENNISDATA_BASE_URL}/{year}/{year}.xlsx"
        dest = os.path.join(out_dir, f"{year}.xlsx")
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            _log(f"  [{year}.xlsx] OK — {len(resp.content)/1024:.0f} Ko", log)
            results[year] = {"ok": True, "size_kb": len(resp.content) / 1024}
        except Exception as e:
            _log(f"  [{year}.xlsx] ECHEC — {e}", log)
            results[year] = {"ok": False, "error": str(e)}
    return results


def run_update(log=None):
    """Orchestre le rafraîchissement complet: pull git + download + consolidation.
    Retourne un résumé (dict) exploitable pour l'affichage (nb de matchs,
    dernière date en base, détail par étape)."""
    from scripts.consolidate.pipeline import run as run_consolidation

    _log("Étape 1/3 — mise à jour des dépôts git (tennis_atp, TML-Database)...", log)
    git_results = pull_git_repos(log)

    _log("Étape 2/3 — téléchargement des résultats/cotes tennis-data.co...", log)
    download_results = download_tennisdata(log=log)

    _log("Étape 3/3 — consolidation (fusion + dédup + écriture de tennis.db)...", log)
    review_csv = os.path.join(DATA_DIR, "match_review.csv")
    sqlite_path = os.path.join(DATA_DIR, "tennis.db")
    matches_df, review_df = run_consolidation(
        DATA_DIR, review_csv, sqlite_path, verbose=True,
    )

    max_date = matches_df["tourney_date"].max() if len(matches_df) else None
    summary = {
        "git_results": git_results,
        "download_results": download_results,
        "n_matches": len(matches_df),
        "n_review_pending": int((review_df["decision"].fillna("").astype(str).str.strip() == "").sum())
                            if "decision" in review_df.columns else None,
        "max_date": str(max_date) if max_date is not None else None,
    }
    _log(f"Terminé — {summary['n_matches']} matchs en base, dernière date: {summary['max_date']}", log)
    return summary


if __name__ == "__main__":
    run_update()
