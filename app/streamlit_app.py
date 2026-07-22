import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import store
from app.backtest import DEFAULT_STRATEGY
from app.bruteforce import run_bruteforce
from app.data import ODDS_CHOICES, load_matches
from app.data_management import render_data_management
from app.features import FEATURE_POOL, build_training_frame, compute_chronological_features, expand_features
from app.modeling import ALGOS
from app.train import train_and_evaluate
from app.upcoming import render_upcoming
from app.validation import run_walk_forward

st.set_page_config(page_title="Tennis ML Lab", layout="wide")


# ----------------------------------------------------------------------
# Glossaire des paramètres (affiché dans la sidebar quand on clique sur le
# bouton "Glossaire" en haut de page — la sidebar n'est plus utilisée pour
# les réglages, elle est donc libre pour ça).
# ----------------------------------------------------------------------
GLOSSARY = {
    "Features (variables du modèle)": {
        "elo_diff": "Écart de rating Elo global entre p1 et p2, calculé match après match sur tout l'historique "
                    "(comme aux échecs: un joueur gagne des points en battant un adversaire mieux classé que lui). "
                    "Positif = p1 est jugé plus fort. Ex: +200 signifie un net avantage pour p1 (~76% de proba théorique).",
        "elo_surface_diff": "Même principe que elo_diff, mais le rating est recalculé séparément par surface "
                            "(un joueur peut être fort sur terre battue et faible sur gazon). Capture la spécialisation surface.",
        "elo_diff_recent": "Comme elo_diff, mais régressé vers la moyenne (1500) en fonction du temps écoulé depuis "
                           "le dernier match connu du joueur (demi-vie 1 an): un rating vieux de plusieurs années pèse "
                           "moins qu'un rating tout juste mis à jour. Utile pour ne pas surestimer un joueur inactif.",
        "elo_surface_diff_recent": "Comme elo_surface_diff, mais avec la même régression temporelle appliquée à la "
                                   "dernière fois où le joueur a joué SUR CETTE SURFACE précise (un joueur peut être "
                                   "très actif en général mais n'avoir pas touché la terre battue depuis longtemps).",
        "surface": "Développe automatiquement 3 features one-hot (surface_clay, surface_grass, surface_hard — Carpet "
                  "en catégorie de référence implicite). Permet au modèle d'apprendre un ajustement direct par "
                  "surface, en complément (pas en remplacement) de elo_surface_diff.",
        "rank_diff": "Écart de classement ATP officiel (rang p2 - rang p1). Positif = p1 mieux classé (rang plus petit). "
                    "Ex: p1 classé 10, p2 classé 50 -> rank_diff = +40.",
        "rank_points_diff": "Écart de points au classement ATP (p1 - p2). Plus fin que rank_diff car reflète l'écart de "
                            "niveau réel (le rang seul écrase les différences en fin de classement).",
        "age_diff": "Écart d'âge en années (p1 - p2). Peut capturer un effet 'expérience vs fraîcheur physique'.",
        "ht_diff": "Écart de taille en cm (p1 - p2). Utile pour des styles de jeu où le service prime (ex: gazon).",
        "form_diff": "Écart de forme récente: taux de victoire sur les 10 derniers matchs de chaque joueur, avant ce "
                    "match (p1 - p2). Ex: p1 a gagné 8/10, p2 a gagné 4/10 -> form_diff = +0.4.",
        "h2h_diff": "Écart de face-à-face historique: (victoires de p1 contre p2) - (victoires de p2 contre p1), "
                   "avant ce match. 0 si c'est leur première confrontation.",
        "hand_advantage": "Indique si p1 (+1) ou p2 (-1) est gaucher et pas l'autre (0 si les deux ont la même main). "
                          "Un gaucher peut avoir un léger avantage stylistique face à des droitiers peu habitués.",
        "implied_prob_p1": "Probabilité de victoire de p1 implicite dans les cotes du marché (1/cote, normalisée pour "
                           "retirer la marge du bookmaker). C'est l'estimation du marché — le modèle doit faire mieux "
                           "que ça pour espérer être rentable en pariant contre.",
        "experience_diff": "Écart du nombre de matchs déjà joués dans l'historique (p1 - p2), un proxy d'expérience "
                           "sur le circuit.",
    },
    "Hyperparamètres par algorithme": {
        "C (régression logistique)": "Inverse de la régularisation. Petit C = modèle plus 'prudent'/simple (moins de "
                                     "surapprentissage), grand C = modèle plus flexible, colle plus aux données "
                                     "d'entraînement (risque de surapprentissage).",
        "max_depth (arbre, forêt, boosting)": "Profondeur maximale d'un arbre de décision. Plus c'est profond, plus le "
                                              "modèle peut capturer des interactions complexes entre features, mais "
                                              "risque de surapprendre le bruit des données d'entraînement.",
        "min_samples_leaf": "Nombre minimum de matchs requis dans une feuille de l'arbre. Plus c'est grand, plus le "
                            "modèle est régularisé (évite de créer des règles basées sur 2-3 matchs isolés).",
        "n_estimators (forêt, boosting)": "Nombre d'arbres combinés dans l'ensemble. Plus il y en a, plus le modèle "
                                          "est stable, mais plus l'entraînement est long.",
        "learning_rate (gradient boosting)": "Vitesse d'apprentissage: à quel point chaque nouvel arbre corrige les "
                                             "erreurs des précédents. Petit = apprentissage plus lent mais souvent "
                                             "plus robuste.",
        "n_neighbors (KNN)": "Nombre de matchs historiques les plus 'similaires' (en termes de features) utilisés "
                             "pour prédire l'issue d'un nouveau match.",
    },
    "Stratégie de mise (backtest)": {
        "Seuil d'edge (edge_threshold)": "Marge minimale exigée entre la probabilité du modèle et la probabilité "
                                         "implicite du marché avant de placer un pari. Ex: seuil de 0.03 -> on ne "
                                         "parie que si le modèle estime au moins 3 points de % de plus que le marché.",
        "Mode de mise — flat": "Mise fixe identique à chaque pari (ex: toujours 1 unité), quel que soit l'edge estimé.",
        "Mode de mise — kelly": "Mise proportionnelle à l'edge et à la cote (critère de Kelly), pour maximiser la "
                                "croissance de la bankroll sur le long terme. Plus agressif quand l'edge est grand.",
        "Fraction de Kelly": "Fraction du Kelly 'plein' réellement misée (ex: 0.25 = un quart du Kelly théorique). "
                             "Réduit le risque/la volatilité, car le Kelly plein est très agressif en pratique.",
        "Bankroll initiale": "Capital de départ simulé pour le backtest, utilisé pour calculer les mises en mode Kelly.",
    },
    "Métriques affichées": {
        "ROI": "Retour sur investissement = profit total / total misé. Ex: ROI de 5% -> pour 100 unités misées au "
              "total sur la période, 5 unités de profit net.",
        "Profit total": "Gain ou perte net en unités de mise sur toute la période de test.",
        "Paris placés (n_bets)": "Nombre de matchs sur lesquels la stratégie a effectivement misé (edge suffisant).",
        "Taux de réussite (win_rate)": "Proportion de paris gagnés (attention: pas forcément corrélé au ROI, un "
                                       "modèle peut gagner souvent mais sur de petites cotes et perdre au global).",
        "Drawdown max": "Pire perte relative depuis un sommet de bankroll pendant le backtest — indicateur de risque.",
        "AUC": "Capacité du modèle à bien classer/ordonner les matchs (0.5 = hasard, 1.0 = parfait). Ne dépend pas "
              "d'un seuil de décision, contrairement à l'accuracy.",
        "Accuracy": "Proportion de matchs où le modèle prédit le bon vainqueur (probabilité >= 50%).",
        "Log loss": "Pénalise fortement les prédictions confiantes mais fausses. Plus c'est bas, mieux c'est.",
        "Brier score": "Erreur quadratique moyenne entre probabilité prédite et résultat réel (0/1). Plus c'est bas, "
                       "mieux le modèle est calibré.",
    },
}


def render_glossary():
    st.header("📖 Glossaire des paramètres")
    st.caption("Définitions des variables, hyperparamètres et métriques utilisés dans l'application.")
    for category, terms in GLOSSARY.items():
        with st.expander(category, expanded=False):
            for term, definition in terms.items():
                st.markdown(f"**{term}**")
                st.caption(definition)


# ----------------------------------------------------------------------
# Données (mises en cache: le calcul Elo/forme/H2H sur tout l'historique
# prend plusieurs secondes, on ne veut le faire qu'une fois par session).
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des matchs et calcul des features (Elo, forme, H2H)...")
def get_dataset():
    matches = load_matches()
    return compute_chronological_features(matches)


@st.cache_data(show_spinner="Construction du jeu d'entraînement...")
def get_training_frame(odds_label, min_date, max_date):
    df = get_dataset()
    odds_w, odds_l = ODDS_CHOICES[odds_label]
    return build_training_frame(df, odds_w, odds_l, min_date=min_date, max_date=max_date)


def split_train_test(frame: pd.DataFrame, test_start):
    train = frame[frame["tourney_date"] < pd.Timestamp(test_start)]
    test = frame[frame["tourney_date"] >= pd.Timestamp(test_start)]
    return train, test


# ----------------------------------------------------------------------
# Composants d'affichage partagés
# ----------------------------------------------------------------------
def render_metrics(metrics: dict, bt: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROI", f"{bt['roi']*100:.1f}%")
    c2.metric("Profit total", f"{bt['total_profit']:.1f} u")
    c3.metric("Paris placés", f"{bt['n_bets']}")
    c4.metric("Taux de réussite", f"{bt['win_rate']*100:.1f}%")
    c5.metric("Drawdown max", f"{bt['max_drawdown']*100:.1f}%")
    c6.metric("Bankroll finale", f"{bt['final_bankroll']:.1f} u")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{metrics['auc']:.3f}" if metrics["auc"] == metrics["auc"] else "n/a")
    c2.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%" if metrics["accuracy"] == metrics["accuracy"] else "n/a")
    c3.metric("Log loss", f"{metrics['logloss']:.3f}" if metrics["logloss"] == metrics["logloss"] else "n/a")
    c4.metric("Brier score", f"{metrics['brier']:.3f}" if metrics["brier"] == metrics["brier"] else "n/a")


def render_equity_curve(bets_df: pd.DataFrame, key_prefix: str):
    if bets_df is None or bets_df.empty:
        st.info("Aucun pari placé avec cette configuration (seuil d'edge jamais atteint).")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bets_df["date"], y=bets_df["bankroll_after"], mode="lines", name="Bankroll"))
    fig.update_layout(title="Courbe de capital (bankroll après chaque pari)",
                       xaxis_title="Date", yaxis_title="Bankroll", height=380)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_equity")

    by_surface = bets_df.groupby("surface").agg(
        profit=("profit", "sum"), staked=("stake", "sum"), n=("profit", "size"),
    ).reset_index()
    by_surface["roi"] = by_surface["profit"] / by_surface["staked"].replace(0, np.nan)
    fig2 = px.bar(by_surface, x="surface", y="roi", text_auto=".1%", title="ROI par surface")
    fig2.update_layout(height=320, yaxis_tickformat=".0%")
    st.plotly_chart(fig2, width="stretch", key=f"{key_prefix}_roi_surface")


def render_calibration(test_frame: pd.DataFrame, key_prefix: str):
    if test_frame is None or "model_prob_p1" not in test_frame.columns or test_frame.empty:
        return
    df = test_frame.copy()
    df["bucket"] = pd.qcut(df["model_prob_p1"], q=min(10, df["model_prob_p1"].nunique()), duplicates="drop")
    calib = df.groupby("bucket", observed=True).agg(
        predicted=("model_prob_p1", "mean"), actual=("label", "mean"), n=("label", "size"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calib["predicted"], y=calib["actual"], mode="markers+lines", name="Modèle"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title="Calibration (probabilité prédite vs taux de victoire réel)",
                       xaxis_title="Probabilité prédite", yaxis_title="Taux de victoire observé", height=380)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_calibration")


def render_feature_importance(model, features, key_prefix: str):
    est = model
    # si modèle calibré (CalibratedClassifierCV sur FrozenEstimator), on redescend
    # jusqu'à l'estimateur de base réellement entraîné
    if hasattr(est, "calibrated_classifiers_"):
        est = est.calibrated_classifiers_[0].estimator.estimator
    # si pipeline (StandardScaler + estimateur), on prend le dernier step
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"feature": features, "importance": est.feature_importances_})
        imp = imp.sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Importance des features")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_importance")
    elif hasattr(est, "coef_"):
        imp = pd.DataFrame({"feature": features, "coefficient": est.coef_[0]})
        imp = imp.sort_values("coefficient", ascending=True)
        fig = px.bar(imp, x="coefficient", y="feature", orientation="h", title="Coefficients du modèle")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_importance")


def render_result_detail(result: dict, key_prefix: str):
    render_metrics(result["metrics"], result["backtest_metrics"])
    render_equity_curve(result["bets_df"], key_prefix)
    render_calibration(result.get("test_frame"), key_prefix)
    if result.get("model") is not None:
        render_feature_importance(result["model"], result["features"], key_prefix)

    with st.expander("Détails de la configuration"):
        st.write("**Algorithme:**", ALGOS[result["algo"]]["label"])
        st.json({"hyperparams": result["params"], "features": result["features"], "strategy": result["strategy"]})

    name = st.text_input("Nom du modèle", value=f"{result['algo']}_{key_prefix}", key=f"name_{key_prefix}")
    if st.button("💾 Sauvegarder ce modèle", key=f"save_{key_prefix}"):
        train_dates = result["test_frame"]["tourney_date"]
        model_id = store.save_model(
            result, name,
            train_start=st.session_state.get("train_start"), train_end=st.session_state.get("train_end"),
            test_start=st.session_state.get("test_start"), test_end=st.session_state.get("test_end"),
        )
        st.success(f"Modèle sauvegardé (id={model_id}).")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
if "show_glossary" not in st.session_state:
    st.session_state.show_glossary = False

title_col, glossary_col = st.columns([6, 1])
with title_col:
    st.title("🎾 Tennis ML Lab")
    st.caption("Backtesting de modèles de prédiction sur data/tennis.db")
with glossary_col:
    st.write("")
    if st.button("📖 Glossaire", help="Ouvre/ferme les définitions des paramètres dans le panneau latéral"):
        st.session_state.show_glossary = not st.session_state.show_glossary

if st.session_state.show_glossary:
    with st.sidebar:
        render_glossary()

try:
    dataset = get_dataset()
except Exception as e:
    st.error(f"Impossible de charger data/tennis.db ({e}).")
    st.divider()
    render_data_management()
    st.divider()
    render_upcoming()
    st.stop()

min_possible = dataset["tourney_date"].min().date()
max_possible = dataset["tourney_date"].max().date()
available_surfaces = [s for s in ["Hard", "Clay", "Grass", "Carpet"] if s in set(dataset["surface"].dropna().unique())]

st.subheader("⚙️ Paramètres généraux")
c1, c2, c3, c4 = st.columns(4)
odds_label = c1.selectbox("Source de cotes (marché de référence)", list(ODDS_CHOICES.keys()))
train_start = c2.date_input("Début période d'entraînement", value=max(min_possible, pd.Timestamp("2005-01-01").date()),
                             min_value=min_possible, max_value=max_possible)
test_start = c3.date_input("Début période de test (backtest)", value=pd.Timestamp("2023-01-01").date(),
                            min_value=min_possible, max_value=max_possible)
test_end = c4.date_input("Fin période de test", value=max_possible, min_value=min_possible, max_value=max_possible)
st.session_state["train_start"], st.session_state["train_end"] = str(train_start), str(test_start)
st.session_state["test_start"], st.session_state["test_end"] = str(test_start), str(test_end)

st.caption("Surfaces à inclure")
surf_cols = st.columns(len(available_surfaces) or 1)
selected_surfaces = []
for i, surf in enumerate(available_surfaces):
    if surf_cols[i].checkbox(surf, value=True, key=f"surf_{surf}"):
        selected_surfaces.append(surf)
selected_surfaces = selected_surfaces or available_surfaces

full_frame = get_training_frame(odds_label, train_start, test_end)
full_frame = full_frame[full_frame["surface"].isin(selected_surfaces)]

train_frame, test_frame = split_train_test(full_frame, test_start)
train_frame = train_frame[train_frame["tourney_date"] >= pd.Timestamp(train_start)]
test_frame = test_frame[test_frame["tourney_date"] <= pd.Timestamp(test_end)]

st.caption(f"{len(full_frame)} matchs avec cotes sur la période/surfaces sélectionnées "
           f"— Train: {len(train_frame)} · Test: {len(test_frame)}")
st.divider()

tab_brute, tab_manual, tab_saved, tab_upcoming, tab_data = st.tabs(
    ["🎲 Bruteforce", "🛠️ Construire un modèle", "💾 Modèles sauvegardés", "📅 Prochains matchs", "🗄️ Données"]
)

# ----------------------------------------------------------------------
# Onglet Bruteforce
# ----------------------------------------------------------------------
with tab_brute:
    st.subheader("Génération aléatoire de modèles")
    c1, c2, c3 = st.columns([1, 1, 1])
    n_models = c1.number_input("Nombre de modèles", min_value=1, max_value=500, value=25, key="bf_n")
    seed = c2.number_input("Seed", min_value=0, value=0, key="bf_seed")
    bf_surface = c3.selectbox(
        "Surface (pour ce bruteforce)", ["Toutes les surfaces sélectionnées"] + selected_surfaces, key="bf_surface",
    )

    bf_train_frame, bf_test_frame = train_frame, test_frame
    if bf_surface != "Toutes les surfaces sélectionnées":
        bf_train_frame = train_frame[train_frame["surface"] == bf_surface]
        bf_test_frame = test_frame[test_frame["surface"] == bf_surface]
    st.caption(f"Données utilisées pour ce bruteforce : {len(bf_train_frame)} train / {len(bf_test_frame)} test "
               f"({bf_surface}).")

    st.caption("Algorithmes autorisés (tirés au hasard parmi ceux cochés)")
    algo_items = list(ALGOS.items())
    algo_cols = st.columns(len(algo_items))
    algos_allowed = []
    for i, (algo_key_, spec) in enumerate(algo_items):
        if algo_cols[i].checkbox(spec["label"], value=True, key=f"bf_algo_{algo_key_}"):
            algos_allowed.append(algo_key_)
    algos_allowed = algos_allowed or list(ALGOS.keys())

    st.caption("Features autorisées (tirées au hasard, en sous-ensemble, parmi celles cochées)")
    feat_cols = st.columns(4)
    bf_features_allowed = []
    for i, feat in enumerate(FEATURE_POOL):
        # implied_prob_p1 décochée par défaut: un modèle qui apprend en partie sur la cote du
        # marché n'est plus comparable indépendamment à ce même marché (cf. onglet Données).
        default_checked = feat != "implied_prob_p1"
        if feat_cols[i % 4].checkbox(feat, value=default_checked, key=f"bf_feat_{feat}"):
            bf_features_allowed.append(feat)
    bf_features_allowed = bf_features_allowed or FEATURE_POOL
    st.caption(
        "⚠️ `implied_prob_p1` (probabilité du marché) comme feature: risque que le modèle se contente de "
        "reproduire le marché plutôt que de détecter un edge indépendant — décoche-la pour des modèles "
        "vraiment indépendants du marché. `surface` développe 3 colonnes one-hot (Clay/Grass/Hard)."
    )

    if st.button("🚀 Lancer le bruteforce", type="primary"):
        if bf_train_frame.empty or bf_test_frame.empty:
            st.error("Pas assez de données sur la période/surface choisies (train ou test vide).")
        else:
            progress = st.progress(0.0)
            rows, results = [], []
            for i, result, err in run_bruteforce(bf_train_frame, bf_test_frame, int(n_models), algos_allowed,
                                                  seed=int(seed), feature_pool=bf_features_allowed):
                progress.progress((i + 1) / n_models)
                if result is None:
                    continue
                m, bt = result["metrics"], result["backtest_metrics"]
                rows.append({
                    "#": i, "algo": ALGOS[result["algo"]]["label"], "n_features": len(result["features"]),
                    "edge": result["strategy"]["edge_threshold"], "stake": result["strategy"]["stake_mode"],
                    "brier": round(m["brier"], 4) if m["brier"] == m["brier"] else None,
                    "logloss": round(m["logloss"], 4) if m["logloss"] == m["logloss"] else None,
                    "auc": round(m["auc"], 3) if m["auc"] == m["auc"] else None,
                    "calibré": m.get("calibrated", False),
                    "n_bets": bt["n_bets"], "roi_%": round(bt["roi"] * 100, 2),
                    "profit": round(bt["total_profit"], 1), "win_rate_%": round(bt["win_rate"] * 100, 1),
                    "max_drawdown_%": round(bt["max_drawdown"] * 100, 1),
                })
                results.append(result)
            progress.empty()
            st.session_state["bf_rows"] = rows
            st.session_state["bf_results"] = results

    rows = st.session_state.get("bf_rows", [])
    results = st.session_state.get("bf_results", [])
    if rows:
        st.caption(
            "Trié par défaut par Brier score croissant (le plus bas = le mieux calibré) — c'est le critère "
            "de sélection recommandé, ROI/profit ne sont que des indicateurs secondaires (cf. onglet Données "
            "pour l'explication). Clique sur une ligne pour voir le détail, ou sur un en-tête de colonne pour "
            "trier autrement (ex: roi_%)."
        )
        df_rows = pd.DataFrame(rows)
        sorted_df = df_rows.sort_values("brier", ascending=True, na_position="last", kind="stable")
        sort_order = sorted_df.index.to_numpy()
        df_rows = sorted_df.reset_index(drop=True)
        results = [results[i] for i in sort_order]
        event = st.dataframe(
            df_rows, width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key="bf_table",
        )
        sel = event.selection.rows if event and event.selection else []
        if sel:
            st.divider()
            idx = sel[0]
            st.markdown(f"### Détail — modèle #{df_rows.iloc[idx]['#']} ({df_rows.iloc[idx]['algo']})")
            render_result_detail(results[idx], key_prefix=f"bf_{idx}")


# ----------------------------------------------------------------------
# Onglet constructeur manuel
# ----------------------------------------------------------------------
with tab_manual:
    st.subheader("Construire un modèle sur-mesure")
    c1, c2 = st.columns(2)
    algo_label = c1.selectbox("Algorithme", [v["label"] for v in ALGOS.values()], key="man_algo")
    algo_key = {v["label"]: k for k, v in ALGOS.items()}[algo_label]
    spec = ALGOS[algo_key]

    st.markdown("**Hyperparamètres**")
    params = {}
    param_cols = st.columns(len(spec["param_widgets"]) or 1)
    for i, (pname, (ptype, lo, hi, default)) in enumerate(spec["param_widgets"].items()):
        with param_cols[i % len(param_cols)]:
            if ptype == "int":
                params[pname] = st.slider(pname, int(lo), int(hi), int(default), key=f"man_{pname}")
            else:
                params[pname] = st.slider(pname, float(lo), float(hi), float(default), key=f"man_{pname}")

    st.markdown("**Features utilisées**")
    feat_cols = st.columns(4)
    features = []
    for i, feat in enumerate(FEATURE_POOL):
        default_checked = feat != "implied_prob_p1"
        if feat_cols[i % 4].checkbox(feat, value=default_checked, key=f"man_feat_{feat}"):
            features.append(feat)
    st.caption(
        "⚠️ `implied_prob_p1` (probabilité du marché) comme feature d'entraînement: le modèle peut se "
        "contenter de reproduire le marché plutôt que de détecter un edge indépendant. Pour identifier de "
        "vraies divergences model vs marché, entraîne plutôt sans cette feature et compare après coup "
        "(c'est déjà ce que fait le backtest, indépendamment de ce choix). "
        "`surface` développe automatiquement 3 colonnes one-hot (Clay/Grass/Hard, Carpet en référence)."
    )

    st.markdown("**Stratégie de mise**")
    c1, c2, c3, c4 = st.columns(4)
    edge_threshold = c1.slider("Seuil d'edge minimum", 0.0, 0.25, DEFAULT_STRATEGY["edge_threshold"], key="man_edge")
    stake_mode = c2.radio("Mode de mise", ["flat", "kelly"], key="man_stake_mode")
    kelly_fraction = c3.slider("Fraction de Kelly", 0.05, 1.0, DEFAULT_STRATEGY["kelly_fraction"], key="man_kelly")
    bankroll = c4.number_input("Bankroll initiale", min_value=1.0, value=100.0, key="man_bankroll")
    strategy = {
        "edge_threshold": edge_threshold, "stake_mode": stake_mode,
        "flat_stake": 1.0, "kelly_fraction": kelly_fraction, "bankroll": bankroll,
    }

    if st.button("🧠 Entraîner et backtester", type="primary", key="man_train_btn"):
        if not features:
            st.error("Sélectionne au moins une feature.")
        elif train_frame.empty or test_frame.empty:
            st.error("Pas assez de données sur la période choisie (train ou test vide).")
        else:
            with st.spinner("Entraînement en cours..."):
                result = train_and_evaluate(train_frame, test_frame, algo_key, params,
                                             expand_features(features), strategy)
            st.session_state["manual_result"] = result

    if st.session_state.get("manual_result"):
        st.divider()
        render_result_detail(st.session_state["manual_result"], key_prefix="manual")

    st.divider()
    st.markdown("#### 🔁 Validation glissante (walk-forward)")
    st.caption(
        "Réentraîne CETTE configuration (algo/hyperparamètres/features/stratégie ci-dessus) sur plusieurs "
        "fenêtres temporelles successives (entraînement extensible, test glissant), puis sur une tranche de "
        "confirmation finale jamais vue pendant la recherche. Un modèle qui n'est bon que sur une seule "
        "coupure train/test peut avoir simplement eu de la chance — la stabilité d'AUC/Brier sur plusieurs "
        "fenêtres est un bien meilleur indicateur de fiabilité qu'un seul run."
    )
    wc1, wc2, wc3, wc4 = st.columns(4)
    wf_n_windows = wc1.slider("Nombre de fenêtres", 2, 8, 4, key="wf_n_windows")
    wf_test_months = wc2.slider("Durée de test par fenêtre (mois)", 1, 12, 6, key="wf_test_months")
    wf_min_train_months = wc3.slider("Entraînement minimum (mois)", 6, 60, 24, key="wf_min_train")
    wf_holdout_months = wc4.slider("Confirmation finale (mois)", 1, 12, 6, key="wf_holdout")

    if st.button("🔁 Lancer la validation glissante", key="wf_run_btn"):
        if not features:
            st.error("Sélectionne au moins une feature.")
        else:
            odds_w, odds_l = ODDS_CHOICES[odds_label]
            with st.spinner("Validation glissante en cours (plusieurs entraînements successifs)..."):
                try:
                    wf_windows, wf_holdout, wf_holdout_bounds = run_walk_forward(
                        dataset, odds_w, odds_l, algo_key, params, expand_features(features), strategy,
                        n_windows=wf_n_windows, test_months=wf_test_months,
                        min_train_months=wf_min_train_months, holdout_months=wf_holdout_months,
                    )
                    st.session_state["wf_result"] = (wf_windows, wf_holdout, wf_holdout_bounds)
                except ValueError as e:
                    st.error(str(e))
                    st.session_state.pop("wf_result", None)

    if st.session_state.get("wf_result"):
        wf_windows, wf_holdout, wf_holdout_bounds = st.session_state["wf_result"]
        if not wf_windows:
            st.warning("Aucune fenêtre exploitable avec ces paramètres (pas assez de matchs avec cotes).")
        else:
            wf_rows = []
            for wr in wf_windows:
                w, r = wr["window"], wr["result"]
                m, bt = r["metrics"], r["backtest_metrics"]
                wf_rows.append({
                    "Fenêtre test": f"{w['test_start'].date()} → {w['test_end'].date()}",
                    "n_test": m["n_test"], "AUC": round(m["auc"], 3), "Brier": round(m["brier"], 4),
                    "LogLoss": round(m["logloss"], 4), "roi_%": round(bt["roi"] * 100, 2),
                    "n_bets": bt["n_bets"],
                })
            wf_df = pd.DataFrame(wf_rows)
            agg = wf_df[["AUC", "Brier", "LogLoss", "roi_%"]].agg(["mean", "std"]).round(4)
            st.dataframe(wf_df, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            c1.metric("AUC moyen (± écart-type)", f"{agg.loc['mean','AUC']:.3f} ± {agg.loc['std','AUC']:.3f}")
            c2.metric("Brier moyen (± écart-type)", f"{agg.loc['mean','Brier']:.4f} ± {agg.loc['std','Brier']:.4f}")
            st.caption(
                f"ROI moyen sur les {len(wf_df)} fenêtres : {agg.loc['mean','roi_%']:.1f}% "
                f"(écart-type {agg.loc['std','roi_%']:.1f} pts) — une forte variabilité d'une fenêtre à "
                "l'autre est un signe que le ROI observé n'est pas fiable."
            )

            st.markdown("**🔒 Confirmation finale (jamais vue pendant la recherche)**")
            if wf_holdout is None:
                st.warning("Pas assez de données sur la période de confirmation pour l'évaluer.")
            else:
                hm, hbt = wf_holdout["metrics"], wf_holdout["backtest_metrics"]
                st.caption(f"{wf_holdout_bounds['test_start'].date()} → {wf_holdout_bounds['test_end'].date()}")
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("AUC", f"{hm['auc']:.3f}")
                hc2.metric("Brier", f"{hm['brier']:.4f}")
                hc3.metric("ROI", f"{hbt['roi']*100:.1f}%")
                hc4.metric("Paris placés", f"{hbt['n_bets']}")


# ----------------------------------------------------------------------
# Onglet modèles sauvegardés
# ----------------------------------------------------------------------
with tab_saved:
    st.subheader("Modèles sauvegardés")
    saved = store.list_models()
    if saved.empty:
        st.info("Aucun modèle sauvegardé pour l'instant.")
    else:
        display_cols = ["id", "name", "algo", "test_start", "test_end", "n_bets",
                         "roi", "total_profit", "win_rate", "auc", "max_drawdown", "created_at"]
        event = st.dataframe(
            saved[display_cols], width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key="saved_table",
        )
        sel = event.selection.rows if event and event.selection else []
        if sel:
            row = saved.iloc[sel[0]]
            st.divider()
            st.markdown(f"### {row['name']} — {row['algo']}")
            bt = dict(n_bets=int(row["n_bets"]), roi=row["roi"], total_profit=row["total_profit"],
                      win_rate=row["win_rate"], final_bankroll=row["final_bankroll"], max_drawdown=row["max_drawdown"])
            m = dict(auc=row["auc"], accuracy=row["accuracy"], logloss=row["logloss"], brier=row["brier"])
            render_metrics(m, bt)
            bets_df = store.load_model_bets(row["id"])
            render_equity_curve(bets_df, f"saved_{row['id']}")
            import json as _json
            with st.expander("Détails de la configuration"):
                st.json({
                    "hyperparams": _json.loads(row["params_json"]),
                    "features": _json.loads(row["features_json"]),
                    "strategy": _json.loads(row["strategy_json"]),
                })
            if st.button("🗑️ Supprimer ce modèle", key=f"del_{row['id']}"):
                store.delete_model(row["id"])
                st.rerun()


# ----------------------------------------------------------------------
# Onglet Prochains matchs
# ----------------------------------------------------------------------
with tab_upcoming:
    render_upcoming(dataset)


# ----------------------------------------------------------------------
# Onglet Données
# ----------------------------------------------------------------------
with tab_data:
    render_data_management()
