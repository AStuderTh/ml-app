"""Onglet 'Données': état de fraîcheur des 3 sources + bouton de mise à
jour (git pull sur les dépôts sources, téléchargement tennis-data.co,
relance de la consolidation dans data/tennis.db)."""
import contextlib
import io
import os
import subprocess

import pandas as pd
import sqlite3
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "tennis.db")

SOURCE_LABELS = {
    "sackmann": "tennis_atp (Sackmann, git)",
    "tml": "TML-Database (git)",
    "tennis_data": "tennis-data.co (xlsx)",
}


class _StreamToCallback(io.StringIO):
    """Redirige stdout vers un callback ligne par ligne, pour afficher les
    print() du pipeline de consolidation en direct dans l'UI."""

    def __init__(self, cb):
        super().__init__()
        self._cb = cb
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._cb(line)
        return len(s)


def _git_head_info(repo_name):
    path = os.path.join(DATA_DIR, repo_name)
    if not os.path.isdir(os.path.join(path, ".git")):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", path, "log", "-1", "--format=%ad|%s", "--date=short"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        date, _, msg = proc.stdout.strip().partition("|")
        return {"last_commit_date": date, "last_commit_msg": msg}
    except Exception:
        return None


def get_db_stats():
    if not os.path.exists(DB_PATH):
        return None
    con = sqlite3.connect(DB_PATH)
    try:
        total = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        max_date = con.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
        by_source = pd.read_sql(
            "SELECT sources, COUNT(*) AS n_matchs, MAX(tourney_date) AS dernier_match "
            "FROM matches GROUP BY sources ORDER BY n_matchs DESC",
            con,
        )
    finally:
        con.close()
    def _relabel(s):
        return " + ".join(SOURCE_LABELS.get(tok, tok) for tok in str(s).split(","))

    by_source["sources"] = by_source["sources"].map(_relabel)
    by_source = by_source.rename(columns={"sources": "source(s)"})
    return {
        "total": total,
        "max_date": max_date,
        "by_source": by_source,
        "db_mtime": pd.Timestamp(os.path.getmtime(DB_PATH), unit="s"),
    }


def render_data_management():
    st.subheader("🗄️ Gestion des données")
    st.caption(
        "data/tennis.db fusionne 3 sources : le dépôt git `tennis_atp` (Sackmann), le dépôt "
        "git `TML-Database`, et les fichiers `tennis-data.co` (résultats + cotes, "
        "téléchargés depuis tennis-data.co.uk)."
    )

    stats = get_db_stats()
    if stats is None:
        st.warning("data/tennis.db introuvable — lance une première mise à jour ci-dessous.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Matchs en base", f"{stats['total']:,}".replace(",", " "))
        c2.metric("Dernier match en base", stats["max_date"] or "n/a")
        c3.metric("Dernière consolidation", stats["db_mtime"].strftime("%Y-%m-%d %H:%M"))

        days_stale = (pd.Timestamp.now().normalize() - pd.Timestamp(stats["max_date"])).days if stats["max_date"] else None
        if days_stale is not None and days_stale > 10:
            st.warning(f"Le dernier match en base date de {days_stale} jours. Une mise à jour est recommandée.")

        st.markdown("**Répartition par source (nombre de matchs et dernière date)**")
        st.dataframe(stats["by_source"], width="stretch", hide_index=True)

    with st.expander("État des dépôts git sources"):
        for repo in ["tennis_atp", "TML-Database"]:
            info = _git_head_info(repo)
            if info:
                st.write(f"**{repo}** — dernier commit local : {info['last_commit_date']} — _{info['last_commit_msg']}_")
            else:
                st.write(f"**{repo}** — introuvable ou pas un dépôt git")

    st.divider()
    st.markdown("### Mettre à jour la base")
    st.caption(
        "1) `git pull` sur les 2 dépôts sources  ·  2) téléchargement des derniers fichiers "
        "tennis-data.co (année en cours + précédente)  ·  3) relance de la consolidation "
        "(fusion, dédoublonnage, écriture de tennis.db). Peut prendre plusieurs minutes."
    )

    if st.button("🔄 Mettre à jour la base de données", type="primary"):
        log_lines = []
        log_box = st.empty()

        def on_log(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines[-200:]))

        with st.spinner("Mise à jour en cours..."):
            from scripts.update_data import run_update
            stream = _StreamToCallback(on_log)
            try:
                with contextlib.redirect_stdout(stream):
                    summary = run_update()
            except Exception as e:
                st.error(f"Échec de la mise à jour : {e}")
                st.exception(e)
                return

        n = summary["n_matches"]
        st.success(f"Mise à jour terminée — {n:,} matchs en base".replace(",", " ") +
                   f", dernier match : {summary['max_date']}.")
        if summary.get("n_review_pending"):
            st.info(
                f"{summary['n_review_pending']} cas ambigus à valider manuellement dans "
                f"data/match_review.csv (colonne `decision`: `merge` ou `reject`), "
                f"puis relance la mise à jour."
            )

        st.cache_data.clear()
        if st.button("Rafraîchir la page"):
            st.rerun()
