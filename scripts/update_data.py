"""Rafraîchit les 3 sources de données et relance la consolidation.

Usage (CLI):
    python scripts/update_data.py

Étapes:
    1. git pull sur data/tennis_atp (dépôt amont Sackmann)
    2. téléchargement des CSV ATP depuis l'API tennismylife. Le dépôt GitHub
       TML-Database n'est plus alimenté depuis janvier 2026, alors que l'API
       reste à jour et sert le même schéma de colonnes; on écrit donc dans
       data/TML-Database/ pour que loaders.load_tml() reste inchangé.
    3. téléchargement du fichier tennis-data.co.uk de la saison en cours
       (+ année précédente, pour couvrir les tournois à cheval sur le
       nouvel an déjà présents en base mais dont l'année vient d'être
       complétée par la source)
    4. relance scripts/consolidate/pipeline.run(...)

Appelé aussi depuis l'app Streamlit (onglet "Données") via `run_update`,
qui accepte un callback `log` pour afficher la progression dans l'UI.
"""
import datetime as dt
import os
import re
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
GIT_REPOS = ["tennis_atp"]
GIT_REPO_URLS = {
    "tennis_atp": "https://github.com/Kadantte/tennis_atp.git",
}
# Le site utilise un préfixe de répertoire public pour les fichiers saisonniers.
# Les URLs historiques /{année}/{année}.xlsx renvoient désormais 404.
TENNISDATA_HOSTS = [
    "https://www.tennis-data.co.uk",
    "https://tennis-data.co.uk",
]
TENNISDATA_PATH_PREFIX = "hrjk-85HytOjkhth76j_ygh4jf7"
TENNISDATA_TIMEOUT = 60
# Pause entre deux requêtes vers tennis-data.co.uk (cf. bride de débit ci-dessus).
TENNISDATA_PAUSE = 3
# Un .xls(x) est soit une archive zip (PK), soit un conteneur OLE2. Toute autre
# entête signale une page d'erreur HTML renvoyée en 200 par le serveur: l'écrire
# détruirait un fichier local parfaitement valide.
_XLS_MAGIC = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
TML_API_URL = "https://stats.tennismylife.org/api/data-files"
# L'API expose aussi le WTA (`AAAA_wta.csv`), le Challenger (`AAAA_challenger.csv`),
# les qualifs (`atp_quali/...`) et des agrégats (`ATP_Database.csv`). On ne veut que
# le main tour ATP, seul format lu par loaders.load_tml().
TML_ATP_FILE_RE = re.compile(r"^\d{4}\.csv$")


def _log(msg, cb=None):
    print(msg)
    if cb:
        cb(msg)


def pull_git_repos(log=None):
    """Clone les dépôts amont absents, sinon fait un git pull --ff-only.

    Un échec (conflit, pas de réseau) sur l'un n'empêche pas d'essayer l'autre
    ni de continuer la suite du pipeline avec les données déjà présentes.
    """
    results = {}
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in GIT_REPOS:
        path = os.path.join(DATA_DIR, name)
        if not os.path.isdir(os.path.join(path, ".git")):
            url = GIT_REPO_URLS.get(name)
            if not url:
                _log(f"  [{name}] URL absente, ignoré", log)
                results[name] = {"ok": False, "output": "URL absente"}
                continue
            try:
                proc = subprocess.run(
                    ["git", "clone", url, path],
                    capture_output=True, text=True, timeout=300,
                )
                out = (proc.stdout + proc.stderr).strip()
                ok = proc.returncode == 0
            except Exception as e:
                out, ok = str(e), False
            _log(f"  [{name}] {'OK' if ok else 'ECHEC'} — {out or 'dépôt cloné'}", log)
            results[name] = {"ok": ok, "output": out}
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", path, "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            out = (proc.stdout + proc.stderr).strip()
            ok = proc.returncode == 0
        except Exception as e:
            out, ok = str(e), False
        _log(f"  [{name}] {'OK' if ok else 'ECHEC'} — {out or 'déjà à jour'}", log)
        results[name] = {"ok": ok, "output": out}
    return results


def download_tml(log=None, refresh_from_year=None):
    """Télécharge les CSV ATP main tour de l'API tennismylife dans
    data/TML-Database/.

    Les saisons closes ne bougeant plus, on ne retélécharge par défaut que
    l'année en cours et la précédente (tournois à cheval sur le nouvel an);
    les années plus anciennes ne sont récupérées que si le fichier manque
    localement. Un échec sur un fichier laisse la version locale en place et
    n'interrompt pas les autres."""
    if refresh_from_year is None:
        refresh_from_year = dt.date.today().year - 1

    out_dir = os.path.join(DATA_DIR, "TML-Database")
    os.makedirs(out_dir, exist_ok=True)
    results = {"downloaded": [], "skipped": 0, "errors": {}}

    try:
        files = requests.get(TML_API_URL, timeout=30).json()["files"]
    except Exception as e:
        _log(f"  [tml] ECHEC listing API — {e}", log)
        results["errors"]["_api"] = str(e)
        return results

    atp = sorted((f for f in files if TML_ATP_FILE_RE.match(f["name"])),
                 key=lambda f: f["name"])
    _log(f"  [tml] {len(atp)} fichiers ATP sur {len(files)} exposés par l'API", log)

    for f in atp:
        dest = os.path.join(out_dir, f["name"])
        year = int(f["name"][:4])
        if year < refresh_from_year and os.path.exists(dest):
            results["skipped"] += 1
            continue
        try:
            resp = requests.get(f["url"], timeout=60)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(resp.content)
            _log(f"  [tml] {f['name']} OK — {len(resp.content)/1024:.0f} Ko", log)
            results["downloaded"].append(f["name"])
        except Exception as e:
            _log(f"  [tml] {f['name']} ECHEC — {e}", log)
            results["errors"][f["name"]] = str(e)

    if results["skipped"]:
        _log(f"  [tml] {results['skipped']} saisons closes déjà présentes, ignorées", log)
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
    first_request = True
    for year in years:
        dest = os.path.join(out_dir, f"{year}.xlsx")
        errors = []
        for host in TENNISDATA_HOSTS:
            # Le serveur limite le débit: des requêtes rapprochées finissent
            # toutes en timeout, alors que la même URL répond en 2 s isolément.
            # Une courte pause entre requêtes évite de déclencher la bride.
            if not first_request:
                time.sleep(TENNISDATA_PAUSE)
            first_request = False
            url = f"{host}/{TENNISDATA_PATH_PREFIX}/{year}/{year}.xlsx"
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                                    timeout=TENNISDATA_TIMEOUT)
                resp.raise_for_status()
                if not resp.content.startswith(_XLS_MAGIC):
                    raise ValueError(
                        f"réponse non-Excel ({len(resp.content)} octets), fichier local conservé")
                with open(dest, "wb") as f:
                    f.write(resp.content)
                _log(f"  [{year}.xlsx] OK — {len(resp.content)/1024:.0f} Ko ({host})", log)
                results[year] = {"ok": True, "size_kb": len(resp.content) / 1024, "host": host}
                break
            except Exception as e:
                errors.append(f"{host}: {e}")
        else:
            # Aucun hôte n'a répondu. Le fichier local (s'il existe) reste en
            # place et la consolidation continue avec: une saison close ne
            # bouge plus, et l'année en cours perd au pire quelques jours de
            # cotes jusqu'à la prochaine tentative.
            kept = "fichier local conservé" if os.path.exists(dest) else "AUCUN fichier local"
            _log(f"  [{year}.xlsx] ECHEC sur {len(TENNISDATA_HOSTS)} hôtes — {kept}", log)
            for err in errors:
                _log(f"      {err}", log)
            results[year] = {"ok": False, "error": " | ".join(errors),
                             "local_kept": os.path.exists(dest)}
    return results


def run_update(log=None):
    """Orchestre le rafraîchissement complet: pull git + download + consolidation.
    Retourne un résumé (dict) exploitable pour l'affichage (nb de matchs,
    dernière date en base, détail par étape)."""
    from scripts.consolidate.pipeline import run as run_consolidation

    _log("Étape 1/4 — mise à jour du dépôt git tennis_atp (Sackmann)...", log)
    git_results = pull_git_repos(log)

    _log("Étape 2/4 — téléchargement des CSV ATP tennismylife...", log)
    tml_results = download_tml(log=log)

    _log("Étape 3/4 — téléchargement des résultats/cotes tennis-data.co...", log)
    download_results = download_tennisdata(log=log)

    _log("Étape 4/4 — consolidation (fusion + dédup + écriture de tennis.db)...", log)
    review_csv = os.path.join(DATA_DIR, "match_review.csv")
    sqlite_path = os.path.join(DATA_DIR, "tennis.db")
    matches_df, review_df = run_consolidation(
        DATA_DIR, review_csv, sqlite_path, verbose=True,
    )

    # date du dernier match RÉELLEMENT disputé, et non date de début du dernier
    # tournoi (cf. commentaire dans app/data_management.get_db_stats).
    if len(matches_df):
        max_date = matches_df[["tourney_date", "match_date"]].max(axis=1).max()
    else:
        max_date = None
    summary = {
        "git_results": git_results,
        "tml_results": tml_results,
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
