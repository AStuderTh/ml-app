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
REVIEW_CSV_PATH = os.path.join(DATA_DIR, "match_review.csv")

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
        has_matches = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='matches'"
        ).fetchone()
        if not has_matches:
            # fichier présent mais vide/sans schéma (ex: recréé automatiquement par un
            # sqlite3.connect() après suppression manuelle de tennis.db) — traité comme
            # absent, pour laisser l'utilisateur relancer une mise à jour depuis zéro.
            return None
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
                f"{summary['n_review_pending']} cas ambigus à valider — utilise la section "
                f"« Cas ambigus à valider » ci-dessous, puis relance la consolidation."
            )

        st.cache_data.clear()
        if st.button("Rafraîchir la page"):
            st.rerun()


def _load_review_df():
    if not os.path.exists(REVIEW_CSV_PATH):
        return None
    df = pd.read_csv(REVIEW_CSV_PATH, dtype=str, keep_default_na=False)
    df["decision"] = df["decision"].fillna("").astype(str).str.strip().str.lower()
    df.loc[~df["decision"].isin(["merge", "reject"]), "decision"] = ""
    return df


def _save_review_decisions(decisions_by_id: dict) -> int:
    """Réécrit data/match_review.csv en appliquant `decisions_by_id`
    ({review_id: "" | "merge" | "reject"}) par-dessus le fichier tel qu'il
    est actuellement sur disque, sans toucher aux autres lignes (celles pas
    affichées dans la page courante gardent leur décision existante)."""
    from scripts.consolidate.review_io import REVIEW_COLUMNS

    df = pd.read_csv(REVIEW_CSV_PATH, dtype=str, keep_default_na=False)
    df = df.set_index("review_id", drop=False)
    n_changed = 0
    for rid, dec in decisions_by_id.items():
        if rid not in df.index:
            continue
        prev = str(df.at[rid, "decision"]).strip().lower()
        if prev != dec:
            df.at[rid, "decision"] = dec
            n_changed += 1
    df = df.reset_index(drop=True)[REVIEW_COLUMNS]
    df.to_csv(REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")
    return n_changed


def render_review_queue():
    """Interface de validation manuelle des cas ambigus de la consolidation
    (data/match_review.csv), pour éviter d'avoir à éditer le CSV à la main."""
    st.subheader("🧩 Cas ambigus à valider")

    df = _load_review_df()
    if df is None or df.empty:
        st.caption("Aucun cas ambigu en attente (data/match_review.csv absent ou vide).")
        return

    n_pending = int((df["decision"] == "").sum())
    n_merge = int((df["decision"] == "merge").sum())
    n_reject = int((df["decision"] == "reject").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("⏳ En attente", n_pending)
    c2.metric("✅ Fusionnés (merge)", n_merge)
    c3.metric("❌ Rejetés (reject)", n_reject)

    if n_pending == 0:
        st.success("Tous les cas ont été tranchés — relance la consolidation ci-dessous pour appliquer les décisions.")

    stages = sorted(df["stage"].dropna().unique().tolist())
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    stage_filter = fc1.selectbox("Étape", ["Toutes"] + stages, key="review_stage_filter")
    status_filter = fc2.selectbox(
        "Statut", ["En attente", "Tous", "merge", "reject"], key="review_status_filter",
    )
    search = fc3.text_input("Recherche (joueur ou tournoi)", key="review_search")

    view = df
    if stage_filter != "Toutes":
        view = view[view["stage"] == stage_filter]
    if status_filter == "En attente":
        view = view[view["decision"] == ""]
    elif status_filter != "Tous":
        view = view[view["decision"] == status_filter]
    if search:
        s = search.strip().lower()
        text_cols = ["a_tourney", "a_winner", "a_loser", "b_tourney", "b_winner", "b_loser"]
        mask = pd.Series(False, index=view.index)
        for c in text_cols:
            mask = mask | view[c].astype(str).str.lower().str.contains(s, na=False, regex=False)
        view = view[mask]

    st.caption(f"{len(view)} cas correspondent aux filtres (sur {len(df)} au total).")
    if view.empty:
        return

    page_size = st.slider("Cas affichés à la fois", 10, 300, 50, step=10, key="review_page_size")
    view = view.head(page_size)

    display_cols = [
        "review_id", "decision", "stage", "reason", "score",
        "a_source", "a_tourney", "a_date", "a_round", "a_winner", "a_loser", "a_score",
        "b_source", "b_tourney", "b_date", "b_round", "b_winner", "b_loser", "b_score",
    ]
    edited = st.data_editor(
        view[display_cols],
        hide_index=True,
        width="stretch",
        key="review_editor",
        disabled=[c for c in display_cols if c != "decision"],
        column_config={
            "review_id": st.column_config.TextColumn("id", width="small"),
            "decision": st.column_config.SelectboxColumn(
                "Décision", options=["", "merge", "reject"], required=False, width="small",
            ),
            "stage": st.column_config.TextColumn("Étape", width="small"),
            "reason": st.column_config.TextColumn("Raison", width="small"),
            "score": st.column_config.NumberColumn("Similarité", format="%.2f", width="small"),
            "a_source": st.column_config.TextColumn("A: source"),
            "a_tourney": st.column_config.TextColumn("A: tournoi"),
            "a_date": st.column_config.TextColumn("A: date"),
            "a_round": st.column_config.TextColumn("A: tour"),
            "a_winner": st.column_config.TextColumn("A: vainqueur"),
            "a_loser": st.column_config.TextColumn("A: perdant"),
            "a_score": st.column_config.TextColumn("A: score"),
            "b_source": st.column_config.TextColumn("B: source"),
            "b_tourney": st.column_config.TextColumn("B: tournoi"),
            "b_date": st.column_config.TextColumn("B: date"),
            "b_round": st.column_config.TextColumn("B: tour"),
            "b_winner": st.column_config.TextColumn("B: vainqueur"),
            "b_loser": st.column_config.TextColumn("B: perdant"),
            "b_score": st.column_config.TextColumn("B: score"),
        },
    )
    st.caption(
        "`merge` = même match (les deux lignes seront fusionnées) · `reject` = matchs "
        "différents (gardés séparés) · vide = pas encore tranché."
    )

    original_by_id = view.set_index("review_id")["decision"]
    changed = {
        row["review_id"]: row["decision"]
        for _, row in edited.iterrows()
        if row["decision"] != original_by_id.loc[row["review_id"]]
    }

    if st.button(
        "💾 Enregistrer les décisions", type="primary", key="review_save_btn", disabled=not changed,
    ):
        n = _save_review_decisions(changed)
        st.success(f"{n} décision(s) enregistrée(s) dans data/match_review.csv.")
        st.rerun()

    with st.expander("Relancer seulement la consolidation (rapide, sans re-télécharger les sources)"):
        st.caption(
            "Applique les décisions enregistrées ci-dessus en refusionnant les données déjà "
            "présentes sur disque, sans repasser par git pull / téléchargement tennis-data.co."
        )
        if st.button("🔁 Relancer la consolidation", key="review_reconsolidate_btn"):
            log_lines = []
            log_box = st.empty()

            def on_log(msg):
                log_lines.append(msg)
                log_box.code("\n".join(log_lines[-200:]))

            with st.spinner("Consolidation en cours..."):
                from scripts.consolidate.pipeline import run as run_consolidation
                stream = _StreamToCallback(on_log)
                try:
                    with contextlib.redirect_stdout(stream):
                        matches_df, review_df = run_consolidation(
                            DATA_DIR, REVIEW_CSV_PATH, DB_PATH, verbose=True,
                        )
                except Exception as e:
                    st.error(f"Échec de la consolidation : {e}")
                    st.exception(e)
                    return

            st.success(f"Consolidation terminée — {len(matches_df):,} matchs en base.".replace(",", " "))
            st.cache_data.clear()
            st.rerun()


def render_db_explorer():
    """Explorateur générique de data/tennis.db: table/colonnes/recherche +
    requête SQL libre. Connexion ouverte en lecture seule (URI mode=ro) —
    donc même une requête UPDATE/DELETE tapée par erreur échoue proprement au
    niveau SQLite plutôt que de dépendre d'un filtrage de texte contournable."""
    st.subheader("🔍 Explorateur de base de données")

    if not os.path.exists(DB_PATH):
        st.info("data/tennis.db introuvable — lance d'abord une mise à jour ci-dessus.")
        return

    st.caption("Lecture seule : aucune requête ci-dessous ne peut modifier data/tennis.db.")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
        )["name"].tolist()
        if not tables:
            st.info("data/tennis.db est vide (aucune table) — lance d'abord une mise à jour ci-dessus.")
            return
        table = st.selectbox("Table", tables, key="explorer_table")

        col_info = pd.read_sql(f"PRAGMA table_info('{table}')", con)
        all_columns = col_info["name"].tolist()
        total_rows = con.execute(f"SELECT COUNT(*) FROM \"{table}\"").fetchone()[0]

        c1, c2 = st.columns([3, 1])
        selected_columns = c1.multiselect(
            "Colonnes affichées", all_columns, default=all_columns, key="explorer_cols",
        )
        row_limit = c2.number_input(
            "Nb de lignes max", min_value=10, max_value=5000, value=200, step=10, key="explorer_limit",
        )
        search = st.text_input(
            "Recherche texte (dans les colonnes texte affichées, ex: un nom de joueur)",
            key="explorer_search",
        )
        st.caption(f"{total_rows:,} lignes au total dans `{table}`.".replace(",", " "))

        cols_sql = ", ".join(f'"{c}"' for c in selected_columns) if selected_columns else "*"
        query = f'SELECT {cols_sql} FROM "{table}"'
        params = []
        if search:
            text_cols = col_info[col_info["type"].str.upper().str.contains("TEXT", na=False)]["name"].tolist()
            text_cols = [c for c in text_cols if c in (selected_columns or all_columns)]
            if text_cols:
                where = " OR ".join(f'"{c}" LIKE ?' for c in text_cols)
                query += f" WHERE {where}"
                params = [f"%{search}%"] * len(text_cols)
        query += f" LIMIT {int(row_limit)}"

        try:
            df = pd.read_sql(query, con, params=params)
            st.caption(f"{len(df)} ligne(s) affichée(s) (clique un en-tête de colonne pour trier).")
            st.dataframe(df, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Erreur de requête : {e}")

        with st.expander("Requête SQL personnalisée (lecture seule)"):
            custom_sql = st.text_area(
                "SQL", value=f"SELECT * FROM {table} LIMIT 50", key="explorer_custom_sql", height=100,
            )
            if st.button("▶️ Exécuter", key="explorer_run_sql"):
                try:
                    result_df = pd.read_sql(custom_sql, con)
                    st.caption(f"{len(result_df)} ligne(s).")
                    st.dataframe(result_df, width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")
    finally:
        con.close()
