import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import store
from app.backtest import DEFAULT_STRATEGY, run_backtest, segment_metrics
from app.bankroll import render_bankroll
from app.bruteforce import run_bruteforce
from app.data import ODDS_CHOICES, load_matches
from app.data_management import render_data_management, render_db_explorer, render_review_queue
from app.features import (FEATURE_POOL, RETIRED_COMMENT, build_training_frame,
                          compute_chronological_features, expand_features)
from app.modeling import ALGOS
from app.roi_bruteforce import run_roi_bruteforce
from app.scoring import (DEFAULT_WEIGHTS, ROI_SCORE_WEIGHTS, compute_quality_score, compute_roi_score,
                          compute_return_risk_ratio)
from app.train import train_and_evaluate
from app.upcoming import render_upcoming
from app.validation import run_walk_forward, surfaces_with_odds, usable_date_range

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
        "Cote minimum / maximum (min_odds / max_odds)": "Bornes optionnelles sur la cote du côté parié, en plus du "
                                "seuil d'edge. Exclut les grosses faveurs (une cote proche de 1 amplifie mécaniquement "
                                "l'edge pour une petite erreur de calibration du modèle) et/ou les gros outsiders "
                                "(variance très élevée pour un edge nominal identique). Laisser vide = pas de borne.",
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
# max_entries: sans borne, Streamlit conserve UNE ENTRÉE PAR COMBINAISON
# d'arguments — chaque déplacement d'un curseur de date ou changement de source
# de cotes ajoute un jeu d'entraînement complet en mémoire, définitivement.
# Le cache stocke par ailleurs l'objet ET sa version picklée, donc le coût réel
# est le double de la taille du DataFrame.
@st.cache_data(max_entries=1,
               show_spinner="Chargement des matchs et calcul des features (Elo, forme, H2H)...")
def get_dataset():
    matches = load_matches()
    return compute_chronological_features(matches)


@st.cache_data(max_entries=4, show_spinner="Construction du jeu d'entraînement...")
def get_training_frame(odds_label, min_date, max_date):
    df = get_dataset()
    odds_w, odds_l = ODDS_CHOICES[odds_label]
    return build_training_frame(df, odds_w, odds_l, min_date=min_date, max_date=max_date)


def compact_result(result: dict) -> dict:
    """Allège un résultat de bruteforce avant de le stocker en session.

    `train_and_evaluate` renvoie une COPIE COMPLÈTE du jeu de test par modèle
    (cf. app/train.py), plus les probabilités d'entraînement et le journal des
    paris. Conservé pour 1 000 modèles, cela représente plusieurs gigaoctets et
    fait tomber l'app en MemoryError — y compris à la simple relecture du cache,
    que Streamlit dépickle intégralement.

    On ne garde que l'index des lignes retenues (le dropna par modèle dépend de
    ses features, donc chaque modèle a son propre sous-ensemble) et ses
    probabilités en float32. Le reste est reconstruit à la demande, pour le
    seul modèle que l'utilisateur sélectionne (cf. rehydrate_result).
    """
    tf = result.pop("test_frame", None)
    if tf is not None:
        result["test_index"] = tf.index.to_numpy()
        result["test_probs"] = tf["model_prob_p1"].to_numpy(dtype="float32")
    result.pop("train_probs", None)
    result.pop("bets_df", None)
    return result


def rehydrate_result(result: dict, test_frame: pd.DataFrame) -> dict:
    """Reconstruit `test_frame` et `bets_df` d'un résultat allégé.

    Le backtest est rejoué plutôt que stocké: il est instantané, alors que
    conserver un journal de paris par modèle coûte des centaines de Mo.
    """
    if "test_index" not in result:
        return result          # résultat non compacté (onglet manuel)
    full = result.copy()
    tf = test_frame.loc[result["test_index"]].copy()
    tf["model_prob_p1"] = result["test_probs"].astype("float64")
    full["test_frame"] = tf
    full["bets_df"], _ = run_backtest(
        tf["model_prob_p1"].to_numpy(), tf, result.get("strategy"))
    return full


def split_train_test(frame: pd.DataFrame, test_start):
    train = frame[frame["tourney_date"] < pd.Timestamp(test_start)]
    test = frame[frame["tourney_date"] >= pd.Timestamp(test_start)]
    return train, test


def render_general_settings(key_prefix: str, min_possible, available_surfaces) -> dict:
    """Réglages qui définissent le jeu de données d'entraînement/test (période,
    source de cotes, surfaces, réserve de confirmation finale). Dupliqué avec des
    clés indépendantes dans chaque onglet qui en a besoin (🎲 Bruteforce et 🛠️
    Construire un modèle — les seuls à consommer train_frame/test_frame): un widget
    Streamlit ne peut pas être partagé visuellement entre deux onglets, et permettre
    à chaque onglet de régler sa propre période/réserve indépendamment est cohérent
    avec le reste de l'app (stratégies de mise déjà dupliquées par contexte)."""
    st.markdown("##### ⚙️ Paramètres généraux")

    # ORDRE DES WIDGETS. Source de cotes, surfaces et abandons d'abord, dates
    # ensuite: la borne haute explorable se déduit de la dernière date que ce
    # périmètre couvre réellement, pas de la fin du dataset brut (cf. plus bas).
    # Poser les dates en premier obligerait à les calculer avant de savoir sur
    # quoi elles portent.
    c_odds, c_conf = st.columns(2)
    odds_label = c_odds.selectbox("Source de cotes (marché de référence)", list(ODDS_CHOICES.keys()),
                                   key=f"{key_prefix}_odds_label")
    confirmation_months = c_conf.slider(
        "🔒 Réserve de confirmation finale (mois)", min_value=1, max_value=12, value=6,
        key=f"{key_prefix}_confirmation_months",
        help="Les X derniers mois COUVERTS PAR LA SOURCE DE COTES choisie sont mis de côté ci-dessous: ni le "
             "bruteforce ni le constructeur manuel ne peuvent les voir (impossible de régler une date au-delà). "
             "Seule la validation glissante (section 🔁 ci-dessous) les évalue, en confirmation finale — ça "
             "garantit qu'aucun réglage manuel ni aucun tri de résultat bruteforce n'a pu être influencé par "
             "cette période avant qu'elle serve de test.",
    )

    # Seules les surfaces que CETTE source de cotes couvre sont proposées: le
    # moquette a disparu du circuit avant que les cotes moyennes ne commencent,
    # donc « Carpet » + « avg » ne désigne aucun match. Offrir la case revient à
    # laisser construire un périmètre vide, qui casse tout ce qui suit — mieux
    # vaut que le choix n'existe pas.
    odds_w_col, odds_l_col = ODDS_CHOICES[odds_label]
    covered = surfaces_with_odds(get_dataset(), odds_w_col, odds_l_col)
    surfaces_here = [s for s in available_surfaces if s in covered]
    dropped = [s for s in available_surfaces if s not in covered]

    st.caption("Surfaces à inclure"
               + (f" — {', '.join(dropped)} indisponible(s) pour la source « {odds_label} »" if dropped else ""))
    surf_cols = st.columns(len(surfaces_here) or 1)
    selected_surfaces = []
    for i, surf in enumerate(surfaces_here):
        if surf_cols[i].checkbox(surf, value=True, key=f"{key_prefix}_surf_{surf}"):
            selected_surfaces.append(surf)
    selected_surfaces = selected_surfaces or surfaces_here

    exclude_retired = st.checkbox(
        "Exclure les matchs abandonnés en cours de jeu", value=False,
        key=f"{key_prefix}_exclude_retired",
        help="Un abandon désigne bien un vainqueur et les paris sont généralement réglés, ces matchs sont donc "
             "gardés par défaut. Mais leur issue dépend surtout de l'état physique du joueur, que le modèle ne "
             "voit pas: les exclure retire ce bruit d'étiquette, au prix d'environ 3 % des lignes. "
             "(Les forfaits, eux, sont toujours écartés: aucun match n'a été joué.)",
    )

    # La validation glissante reconstruit son propre jeu à chaque fenêtre, à
    # partir de l'historique complet: sans ce pré-filtrage, elle évaluerait
    # sur des surfaces ou des statuts de match que l'exploration a exclus, et
    # sa confirmation ne porterait donc pas sur la même population.
    wf_dataset = get_dataset()
    wf_dataset = wf_dataset[wf_dataset["surface"].isin(selected_surfaces)]
    if exclude_retired and "match_comment" in wf_dataset.columns:
        wf_dataset = wf_dataset[wf_dataset["match_comment"] != RETIRED_COMMENT]

    # Bornes réelles du périmètre, et non celles du dataset brut: toutes les
    # sources de cotes ne vont pas jusqu'au dernier match de la base (Pinnacle
    # s'arrête plusieurs mois avant B365/avg). Calculée sur la fin du dataset, la
    # réserve de confirmation pouvait tomber intégralement après la dernière
    # cote de la source — 0 match à évaluer, confirmation impossible. Le
    # walk-forward dérive ses fenêtres du MÊME appel sur le MÊME jeu filtré
    # (cf. validation.run_walk_forward), donc la borne affichée ici est bien
    # celle qui sépare exploration et confirmation.
    data_min, data_max = usable_date_range(wf_dataset, odds_w_col, odds_l_col)
    if data_min is None:
        # ne devrait plus arriver (les surfaces non couvertes ne sont pas
        # proposées ci-dessus), mais rien de la suite n'a de sens sans dates
        st.error(
            f"Aucun match avec des cotes « {odds_label} » sur ce périmètre — "
            "change de source de cotes ou élargis les surfaces."
        )
        st.stop()
    data_min, data_max = data_min.date(), data_max.date()

    confirmation_start = (pd.Timestamp(data_max) - pd.DateOffset(months=confirmation_months)).date()
    # le holdout du walk-forward inclut les matchs du jour confirmation_start lui-même
    # (tourney_date >= confirmation_start côté validation.py) — donc l'exploration doit
    # s'arrêter la veille pour qu'aucun match ne soit visible des deux côtés à la fois.
    exploration_max = (pd.Timestamp(confirmation_start) - pd.Timedelta(days=1)).date()
    st.caption(
        f"Cotes « {odds_label} » disponibles du **{data_min}** au **{data_max}** sur ce périmètre. "
        f"Zone réservée à la confirmation finale : **{confirmation_start} → {data_max}** — non explorable "
        "ci-dessous, y compris dans le bruteforce et le constructeur manuel."
    )

    # exploration_max recule dès qu'on allonge la réserve, qu'on change de
    # source de cotes ou qu'on décoche une surface. Une date déjà posée en
    # session peut donc se retrouver au-delà de la nouvelle borne: Streamlit
    # lève alors une exception au lieu de rendre le widget. On la ramène dans
    # la zone explorable, seul comportement qui ne laisse jamais une date de
    # test empiéter sur la réserve.
    for date_key in ("_train_start", "_test_start", "_test_end"):
        stored = st.session_state.get(f"{key_prefix}{date_key}")
        if stored is not None and stored > exploration_max:
            st.session_state[f"{key_prefix}{date_key}"] = exploration_max

    c1, c2, c3 = st.columns(3)
    train_start = c1.date_input("Début période d'entraînement", value=max(min_possible, pd.Timestamp("2005-01-01").date()),
                                 min_value=min_possible, max_value=exploration_max, key=f"{key_prefix}_train_start")
    test_start = c2.date_input("Début période de test (backtest)", value=pd.Timestamp("2023-01-01").date(),
                                min_value=min_possible, max_value=exploration_max, key=f"{key_prefix}_test_start")
    test_end = c3.date_input("Fin période de test", value=exploration_max, min_value=min_possible,
                              max_value=exploration_max, key=f"{key_prefix}_test_end")

    full_frame = get_training_frame(odds_label, train_start, test_end)
    full_frame = full_frame[full_frame["surface"].isin(selected_surfaces)]
    if exclude_retired and "match_comment" in full_frame.columns:
        full_frame = full_frame[full_frame["match_comment"] != RETIRED_COMMENT]

    train_frame, test_frame = split_train_test(full_frame, test_start)
    train_frame = train_frame[train_frame["tourney_date"] >= pd.Timestamp(train_start)]
    test_frame = test_frame[test_frame["tourney_date"] <= pd.Timestamp(test_end)]

    st.caption(f"{len(full_frame)} matchs avec cotes sur la période/surfaces sélectionnées "
               f"— Train: {len(train_frame)} · Test: {len(test_frame)}")
    st.divider()

    return {
        "train_frame": train_frame, "test_frame": test_frame,
        "odds_label": odds_label, "selected_surfaces": selected_surfaces,
        "exclude_retired": exclude_retired,
        "train_start": str(train_start), "train_end": str(test_start),
        "test_start": str(test_start), "test_end": str(test_end),
        "confirmation_start": confirmation_start, "confirmation_months": confirmation_months,
        "data_max": data_max, "wf_dataset": wf_dataset,
    }


# ----------------------------------------------------------------------
# Composants d'affichage partagés
# ----------------------------------------------------------------------
COMPONENT_LABELS = {
    "logloss": "Log loss", "brier": "Brier score", "calibration": "Calibration (ECE)",
    "auc": "AUC (discrimination)", "n_matches": "Nombre de matchs (robustesse)",
}


def render_quality_score(metrics: dict, key_prefix: str):
    result = compute_quality_score(metrics, DEFAULT_WEIGHTS)
    score = result["score"]
    st.metric("🏆 Score de qualité", f"{score:.1f}/100" if score == score else "n/a",
              help="Note composite pondérée: Log loss 32%, Brier 26%, Calibration 21%, AUC 11%, "
                   "Nombre de matchs 10%. Ne dépend PAS du ROI/backtest: dans le bruteforce la stratégie de "
                   "mise est tirée au hasard indépendamment du modèle, donc le ROI mesure la chance du tirage "
                   "de stratégie autant que la qualité du modèle (cf. onglet Données) — à vérifier séparément, "
                   "ROI/nombre de paris restent affichés à part. Chaque critère est ramené à une échelle 0-1 "
                   "par rapport à une référence absolue, pas relative aux autres modèles du lot — donc "
                   "comparable d'un run à l'autre.")
    with st.expander("Détail du score de qualité"):
        for name, (skill, weight) in result["components"].items():
            if weight <= 0:
                continue
            label = COMPONENT_LABELS.get(name, name)
            st.write(f"- **{label}** (poids {weight*100:.0f}%): "
                     f"{'n/a (métrique indisponible)' if skill != skill else f'{skill*100:.0f}/100'}")


def render_backtest_metrics(bt: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROI", f"{bt['roi']*100:.1f}%",
              help="Profit / total misé (turnover). Mesure la marge moyenne extraite par mise, "
                   "mais pas comparable entre flat et kelly: kelly mise proportionnellement plus "
                   "au fil du temps (les mises grossissent avec la bankroll), ce qui dilue son ROI% "
                   "sans refléter une moindre rentabilité réelle. Pour comparer la rentabilité "
                   "réelle entre stratégies, voir 'Rendement/capital' et 'Rendement/risque' ci-dessous.")
    c2.metric("Profit total", f"{bt['total_profit']:.1f} u")
    c3.metric("Paris placés", f"{bt['n_bets']}")
    c4.metric("Taux de réussite", f"{bt['win_rate']*100:.1f}%")
    c5.metric("Drawdown max", f"{bt['max_drawdown']*100:.1f}%")
    c6.metric("Bankroll finale", f"{bt['final_bankroll']:.1f} u")

    rr = compute_return_risk_ratio(bt)
    d1, d2 = st.columns(2)
    ror = rr["return_on_bankroll"]
    d1.metric("Rendement/capital", f"{ror*100:.1f}%" if ror == ror else "n/a",
              help="Profit total / bankroll de départ. Contrairement au ROI, comparable "
                   "directement entre stratégies flat et kelly: c'est le multiple réel de "
                   "croissance de votre capital initial.")
    calmar = rr["calmar_ratio"]
    d2.metric("Rendement/risque (Calmar)", f"{calmar:.2f}" if calmar == calmar else "n/a",
              help="Rendement/capital divisé par le drawdown max. Résume en un seul chiffre "
                   "la rentabilité RAPPORTÉE au risque de creux traversé pour l'obtenir: plus "
                   "c'est haut, plus le rendement obtenu compense le risque pris. 'n/a' si le "
                   "drawdown est quasi nul (rien à rapporter).")


def render_model_metrics(metrics: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC", f"{metrics['auc']:.3f}" if metrics["auc"] == metrics["auc"] else "n/a")
    c2.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%" if metrics["accuracy"] == metrics["accuracy"] else "n/a")
    c3.metric("Log loss", f"{metrics['logloss']:.3f}" if metrics["logloss"] == metrics["logloss"] else "n/a")
    c4.metric("Brier score", f"{metrics['brier']:.3f}" if metrics["brier"] == metrics["brier"] else "n/a")
    ece_val = metrics.get("ece", float("nan"))
    c5.metric("ECE (calibration)", f"{ece_val:.3f}" if ece_val == ece_val else "n/a")


def render_metrics(metrics: dict, bt: dict):
    render_backtest_metrics(bt)
    render_model_metrics(metrics)


def render_roi_score(backtest_metrics: dict, key_prefix: str):
    result = compute_roi_score(backtest_metrics, ROI_SCORE_WEIGHTS)
    score = result["score"]
    st.metric("🎯 Score ROI (règle)", f"{score:.1f}/100" if score == score else "n/a",
              help="Note composite pour classer des RÈGLES DE SÉLECTION entre elles sur un modèle fixe: "
                   "ROI 50%, nombre de paris 25% (confiance statistique), drawdown maîtrisé 25%. Sans rapport "
                   "avec le 🏆 Score de qualité du modèle — ici le modèle ne change pas, seule la règle varie.")
    with st.expander("Détail du score ROI"):
        for name, (skill, weight) in result["components"].items():
            if weight <= 0:
                continue
            label = {"roi": "ROI", "n_bets": "Nombre de paris (robustesse)",
                     "drawdown": "Drawdown maîtrisé"}.get(name, name)
            st.write(f"- **{label}** (poids {weight*100:.0f}%): "
                     f"{'n/a (métrique indisponible)' if skill != skill else f'{skill*100:.0f}/100'}")


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

    render_segment_table(bets_df, key_prefix)


LEVEL_LABELS = {"G": "Grand Chelem", "M": "Masters 1000", "A": "ATP 250/500",
                "F": "Finals", "250": "ATP 250", "500": "ATP 500"}


def render_segment_table(bets_df: pd.DataFrame, key_prefix: str):
    """ROI par niveau de tournoi, avec incertitude et pariabilité en production.

    Sépare deux questions que le ROI global confond: le modèle gagne-t-il, et
    gagne-t-il là où l'on peut réellement miser. Un ROI porté par les ATP 250
    n'est pas monétisable tant que app/odds.py ne couvre que les Grand Chelems
    et les Masters."""
    seg = segment_metrics(bets_df)
    if seg.empty:
        return

    st.markdown("**ROI par niveau de tournoi**")
    disp = seg.reset_index().rename(columns={"tourney_level": "niveau"})
    disp["niveau"] = disp["niveau"].map(lambda v: LEVEL_LABELS.get(v, v if v else "inconnu"))
    disp["ROI"] = disp.apply(lambda r: f"{r['roi']*100:+.1f}% ± {r['roi_se']*200:.1f}", axis=1)
    disp["Cote moy."] = disp["avg_odds"].round(2)
    disp["% gagnés"] = (disp["win_rate"] * 100).round(1)
    disp["Significatif"] = np.where(disp["significatif"], "oui", "—")
    if "pariable_live" not in seg.columns:
        return
    disp["Pariable live"] = np.where(disp["pariable_live"], "oui", "non")
    st.dataframe(
        disp[["niveau", "n_bets", "ROI", "Cote moy.", "% gagnés", "Significatif", "Pariable live"]],
        hide_index=True, width="stretch", key=f"{key_prefix}_segments",
    )

    jouable = seg[seg["pariable_live"]]
    hors = seg[~seg["pariable_live"]]
    if not jouable.empty and not hors.empty and hors["staked"].sum() > 0:
        roi_jouable = jouable["profit"].sum() / max(jouable["staked"].sum(), 1e-9)
        roi_hors = hors["profit"].sum() / max(hors["staked"].sum(), 1e-9)
        if roi_hors > 0 and roi_jouable <= 0:
            st.warning(
                f"⚠️ Le ROI est porté par des niveaux SANS cotes live : {roi_hors*100:+.1f}% "
                f"hors couverture contre {roi_jouable*100:+.1f}% sur les Grand Chelems/Masters. "
                "En production vous ne pourriez miser que sur la seconde moitié. Le chantier "
                "prioritaire est alors la couverture des cotes (app/odds.py), pas le modèle."
            )
        else:
            st.caption(
                f"ROI sur les niveaux pariables en live : {roi_jouable*100:+.1f}% "
                f"({int(jouable['n_bets'].sum())} paris) — contre {roi_hors*100:+.1f}% "
                f"sur les niveaux non couverts par app/odds.py."
            )


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


@st.dialog("Supprimer ce modèle ?")
def confirm_delete_model(model_id: str, model_name: str):
    st.write(f"Supprimer définitivement **{model_name}** ainsi que ses règles de sélection sauvegardées ? "
             "Cette action est irréversible.")
    c1, c2 = st.columns(2)
    if c1.button("Annuler", key=f"cancel_del_model_{model_id}"):
        st.rerun()
    if c2.button("🗑️ Supprimer définitivement", key=f"confirm_del_model_{model_id}", type="primary"):
        store.delete_model(model_id)
        st.session_state.pop("saved_table", None)  # évite de pointer sur la mauvaise ligne après suppression
        st.rerun()


@st.dialog("Supprimer cette règle de sélection ?")
def confirm_delete_selection_rule(rule_id: str, rule_name: str):
    st.write(f"Supprimer définitivement la règle de sélection **{rule_name}** ? Cette action est irréversible.")
    c1, c2 = st.columns(2)
    if c1.button("Annuler", key=f"cancel_del_strat_{rule_id}"):
        st.rerun()
    if c2.button("🗑️ Supprimer définitivement", key=f"confirm_del_strat_{rule_id}", type="primary"):
        store.delete_selection_rule(rule_id)
        st.rerun()


def _load_model_into_manual_constructor(row):
    """Prépare la config d'un modèle sauvegardé (algo/hyperparams/features/
    stratégie) dans st.session_state, avec les clés attendues par les widgets
    de l'onglet 🛠️ Construire un modèle — pour l'éditer sans réécrire son
    config à la main. N'entraîne rien: l'utilisateur ajuste puis relance
    lui-même l'entraînement dans cet onglet."""
    algo = row["algo"]
    params = json.loads(row["params_json"])
    features_expanded = json.loads(row["features_json"])
    strategy = json.loads(row["strategy_json"])

    st.session_state["man_algo"] = ALGOS[algo]["label"]
    spec = ALGOS[algo]
    for pname, val in params.items():
        if val is None or pname not in spec["param_widgets"]:
            continue  # ex: random_forest max_depth=None (bruteforce) n'a pas d'équivalent sur le slider
        ptype = spec["param_widgets"][pname][0]
        st.session_state[f"man_{pname}"] = int(val) if ptype == "int" else float(val)

    for feat in FEATURE_POOL:
        if feat == "surface":
            present = any(c in features_expanded for c in ["surface_clay", "surface_grass", "surface_hard"])
        else:
            present = feat in features_expanded
        st.session_state[f"man_feat_{feat}"] = present

    st.session_state["man_edge"] = strategy.get("edge_threshold", DEFAULT_STRATEGY["edge_threshold"])
    st.session_state["man_stake_mode"] = strategy.get("stake_mode", DEFAULT_STRATEGY["stake_mode"])
    st.session_state["man_kelly"] = strategy.get("kelly_fraction", DEFAULT_STRATEGY["kelly_fraction"])
    st.session_state["man_bankroll"] = strategy.get("bankroll", DEFAULT_STRATEGY["bankroll"])
    st.session_state["man_min_odds"] = strategy.get("min_odds") or 1.0
    st.session_state["man_max_odds"] = strategy.get("max_odds") or 0.0
    st.session_state["_loaded_from_saved_name"] = row["name"]


def render_result_detail(result: dict, key_prefix: str, odds_label: str = None,
                          train_start: str = None, train_end: str = None,
                          test_start: str = None, test_end: str = None,
                          surfaces=None, exclude_retired: bool = None):
    render_quality_score(result["metrics"], key_prefix)
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
        odds_w_col, odds_l_col = ODDS_CHOICES[odds_label] if odds_label else (None, None)
        model_id = store.save_model(
            result, name,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            odds_w_col=odds_w_col, odds_l_col=odds_l_col,
            # périmètre des données, pour que la validation glissante d'un modèle
            # rechargé porte sur la population qu'il a réellement apprise
            surfaces=surfaces, exclude_retired=exclude_retired,
        )
        st.session_state[f"saved_model_id_{key_prefix}"] = model_id
        st.success(f"Modèle sauvegardé (id={model_id}).")


def describe_strategy(strategy) -> str:
    """Résumé lisible d'une stratégie de mise, pour lever l'ambiguïté sur celle
    réellement appliquée quand plusieurs sont en jeu dans la même vue."""
    edge = f"edge ≥ {strategy['edge_threshold'] * 100:.1f}%"
    stake = f"Kelly ×{strategy.get('kelly_fraction', 0):.2f}" if strategy.get("stake_mode") == "kelly" else "mise plate"
    min_o, max_o = strategy.get("min_odds"), strategy.get("max_odds")
    odds_range = ""
    if min_o is not None or max_o is not None:
        odds_range = f", cote [{min_o or '-'}, {max_o or '-'}]"
    return f"{edge}, {stake}{odds_range}"


# Colonnes strictement nécessaires pour rejouer un backtest (cf. run_backtest)
# + la probabilité prédite. Le reste du jeu de test (toutes les features, Elo,
# H2H...) n'est plus utile une fois le modèle entraîné.
BACKTEST_COLS = ["model_prob_p1", "odds_p1", "odds_p2", "label", "tourney_date",
                 "match_id", "p1_name", "p2_name", "surface", "tourney_level"]

# Le suffixe fait partie du nom de colonne, et non d'une légende sous le tableau:
# ce ROI est calculé avec une stratégie de mise réglée sur ces mêmes matchs (cf.
# app/validation.py). Un tableau se lit et se recopie sans sa légende — la réserve
# doit voyager avec le chiffre.
WF_ROI_COL = "roi_% (in-sample)"


def slim_wf_result(result: dict) -> dict:
    """Allège un résultat de fenêtre avant de le garder en session: on conserve
    le jeu de test réduit aux colonnes du backtest (nécessaire pour recalculer
    roi/n_bets à la volée quand la stratégie change, cf. rebacktest_window),
    mais pas l'estimateur entraîné, les probabilités d'entraînement ni le
    journal des paris — inutilisés à l'affichage et coûteux en mémoire une fois
    multipliés par le nombre de fenêtres (cf. compact_result)."""
    if result is None:
        return None
    slim = {k: v for k, v in result.items() if k not in ("model", "train_probs", "bets_df")}
    tf = result.get("test_frame")
    if tf is not None:
        slim["test_frame"] = tf[[c for c in BACKTEST_COLS if c in tf.columns]].copy()
    return slim


def rebacktest_window(result: dict, strategy: dict) -> dict:
    """Rejoue le backtest d'une fenêtre déjà entraînée avec une AUTRE stratégie
    de mise, sans réentraîner: le modèle et ses probabilités prédites sont figés
    une fois la fenêtre calculée, donc changer de règle ROI ne peut déplacer que
    roi/n_bets/drawdown — les réentraîner coûterait plusieurs minutes pour un
    résultat identique sur AUC/Brier/LogLoss."""
    tf = result.get("test_frame")
    if tf is None or "model_prob_p1" not in tf:
        return result["backtest_metrics"]        # ancien résultat en session, avant ce mécanisme
    _, bt = run_backtest(tf["model_prob_p1"].to_numpy(), tf, strategy)
    return bt


def render_walk_forward_section(algo_key, params, features, strategy, key_prefix, odds_label,
                                 confirmation_start, confirmation_months, data_max, wf_dataset,
                                 strategy_source=None):
    """Réentraîne (algo/hyperparamètres/features/stratégie) sur plusieurs fenêtres
    glissantes + la réserve de confirmation finale. Utilisé à la fois par le
    constructeur manuel (config des widgets) et par l'onglet bruteforce (config
    figée d'un modèle déjà généré) — le comportement est identique, seule la
    provenance de la config change.

    `strategy_source` décrit d'où vient la stratégie de mise reçue (ex: règle ROI
    sélectionnée vs stratégie de référence du bruteforce): l'appelant peut faire
    varier `strategy` indépendamment du modèle, et sans cette mention rien à
    l'écran ne dirait laquelle des deux produit les colonnes roi_%/n_bets.

    `data_max` est la dernière date couverte par la source de cotes SUR CE
    PÉRIMÈTRE (cf. validation.usable_date_range), pas la fin du dataset brut:
    c'est elle qui borne la réserve de confirmation."""
    st.divider()
    st.markdown("#### 🔁 Validation glissante (walk-forward)")
    st.caption(
        "Réentraîne CETTE configuration (algo/hyperparamètres/features/stratégie ci-dessus) sur plusieurs "
        "fenêtres temporelles successives (entraînement extensible, test glissant), puis sur la tranche de "
        "confirmation finale définie plus haut (🔒 Réserve de confirmation finale). Un modèle qui n'est bon "
        "que sur une seule coupure train/test peut avoir simplement eu de la chance : voir la config tenir "
        "d'une fenêtre à l'autre est un signe de stabilité qu'un run unique ne donne pas."
    )
    st.warning(
        "⚠️ **Ce que les fenêtres ci-dessous ne prouvent pas.** Elles recouvrent la période sur laquelle cette "
        "config a été CHOISIE (le bruteforce y a trié ses modèles, l'optimisation ROI y a réglé son seuil "
        "d'edge et ses bornes de cote). Réentraîner le modèle par fenêtre ne défait pas ce choix : leur AUC/"
        "Brier est optimiste, et leur **ROI est in-sample** puisque la stratégie de mise a été ajustée sur ces "
        "mêmes matchs. Elles mesurent la STABILITÉ, pas la généralisation — pour ça, il n'y a que la "
        "confirmation finale, en bas de section."
    )
    if strategy_source:
        st.caption(
            f"Stratégie de mise appliquée ici : **{describe_strategy(strategy)}** — {strategy_source}. "
            "Elle ne change que les colonnes roi_%/n_bets/ROI de confirmation, recalculées immédiatement à "
            "chaque changement de règle (sans réentraînement). AUC/Brier/LogLoss et le score qualité, eux, ne "
            "dépendent pas de la mise : ils restent identiques quelle que soit la règle ROI sélectionnée."
        )
    wc1, wc2, wc3 = st.columns(3)
    wf_n_windows = wc1.slider("Nombre de fenêtres", 2, 8, 4, key=f"{key_prefix}_wf_n_windows")
    wf_test_months = wc2.slider("Durée de test par fenêtre (mois)", 1, 12, 6, key=f"{key_prefix}_wf_test_months")
    wf_min_train_months = wc3.slider(
        "Entraînement minimum (mois)", 6, 60, 24, key=f"{key_prefix}_wf_min_train",
        help="Une fenêtre dont la période d'entraînement est plus courte que ça n'est pas générée. Le compte "
             "part de la première date où la source de cotes choisie a des données (et non du premier match de "
             "la base, qui remonte à 1968 sans aucune cote) — demander un minimum trop élevé peut donc ne "
             "produire aucune fenêtre.",
    )
    st.caption(
        f"Confirmation finale : **{confirmation_start} → {data_max}** (réglée dans "
        "🔒 Réserve de confirmation finale ci-dessus, en haut de page — la même réserve que celle qui limite "
        "déjà l'exploration, pour garantir qu'elle est réellement jamais vue avant ce test)."
    )

    result_key = f"{key_prefix}_wf_result"
    if st.button("🔁 Lancer la validation glissante", key=f"{key_prefix}_wf_run_btn"):
        if not features:
            st.error("Sélectionne au moins une feature.")
        else:
            odds_w, odds_l = ODDS_CHOICES[odds_label]
            with st.spinner("Validation glissante en cours (plusieurs entraînements successifs)..."):
                try:
                    wf_windows, wf_holdout, wf_holdout_bounds = run_walk_forward(
                        wf_dataset, odds_w, odds_l, algo_key, params, expand_features(features), strategy,
                        n_windows=wf_n_windows, test_months=wf_test_months,
                        min_train_months=wf_min_train_months, holdout_months=confirmation_months,
                    )
                    wf_windows = [{"window": wr["window"], "result": slim_wf_result(wr["result"])}
                                  for wr in wf_windows]
                    st.session_state[result_key] = (wf_windows, slim_wf_result(wf_holdout),
                                                    wf_holdout_bounds)
                except ValueError as e:
                    st.error(str(e))
                    st.session_state.pop(result_key, None)

    if st.session_state.get(result_key):
        wf_windows, wf_holdout, wf_holdout_bounds = st.session_state[result_key]
        if not wf_windows:
            st.warning("Aucune fenêtre exploitable avec ces paramètres (pas assez de matchs avec cotes).")
        else:
            wf_rows = []
            for wr in wf_windows:
                w, r = wr["window"], wr["result"]
                # backtest rejoué avec la stratégie ACTUELLE (cf. rebacktest_window):
                # sans ça, changer de règle ROI laissait les roi_%/n_bets figés sur
                # la stratégie du dernier run.
                m, bt = r["metrics"], rebacktest_window(r, strategy)
                # borne haute exclusive (cf. validation.generate_walk_forward_windows):
                # afficher test_end telle quelle laisserait croire que le dernier jour
                # est évalué ici, alors qu'il appartient à la fenêtre suivante.
                last_day = w["test_end"] - pd.Timedelta(days=int(not w.get("test_end_inclusive", False)))
                wf_rows.append({
                    "Fenêtre test": f"{w['test_start'].date()} → {last_day.date()}",
                    "n_test": m["n_test"], "AUC": round(m["auc"], 3), "Brier": round(m["brier"], 4),
                    "LogLoss": round(m["logloss"], 4), WF_ROI_COL: round(bt["roi"] * 100, 2),
                    "n_bets": bt["n_bets"],
                })
            wf_df = pd.DataFrame(wf_rows)
            agg = wf_df[["AUC", "Brier", "LogLoss", WF_ROI_COL]].agg(["mean", "std"]).round(4)
            st.dataframe(wf_df, width="stretch", hide_index=True, key=f"{key_prefix}_wf_table")
            c1, c2 = st.columns(2)
            c1.metric("AUC moyen (± écart-type)", f"{agg.loc['mean','AUC']:.3f} ± {agg.loc['std','AUC']:.3f}")
            c2.metric("Brier moyen (± écart-type)", f"{agg.loc['mean','Brier']:.4f} ± {agg.loc['std','Brier']:.4f}")
            st.caption(
                f"ROI moyen sur les {len(wf_df)} fenêtres : {agg.loc['mean',WF_ROI_COL]:.1f}% "
                f"(écart-type {agg.loc['std',WF_ROI_COL]:.1f} pts) — à lire comme une mesure de STABILITÉ, pas "
                "de rendement attendu : la stratégie de mise a été réglée sur ces mêmes matchs, donc ce chiffre "
                "ne peut qu'être flatteur. Une forte variabilité d'une fenêtre à l'autre reste, elle, un signal "
                "négatif exploitable : même avantagé, le ROI ne tient pas."
            )

            st.write("")
            with st.container(border=True):
                st.markdown("### 🔒 Confirmation finale — jamais vue pendant la recherche")
                st.caption(
                    "Contrairement aux fenêtres ci-dessus (qui font partie du réglage de la config), cette "
                    "tranche est mise de côté AVANT tout entraînement : c'est le seul résultat qui n'est pas "
                    "exposé au biais de comparaisons multiples du bruteforce ni au réglage manuel des "
                    "hyperparamètres. Si un chiffre ici doit faire foi, c'est celui-là."
                )
                if wf_holdout is None:
                    st.warning("Pas assez de données sur la période de confirmation pour l'évaluer.")
                else:
                    hm, hbt = wf_holdout["metrics"], rebacktest_window(wf_holdout, strategy)
                    hq = compute_quality_score(hm, DEFAULT_WEIGHTS)
                    st.markdown(
                        f"**Période : {wf_holdout_bounds['test_start'].date()} → "
                        f"{wf_holdout_bounds['test_end'].date()}**"
                    )
                    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns(6)
                    hc1.metric("🏆 Score qualité", f"{hq['score']:.1f}/100" if hq["score"] == hq["score"] else "n/a")
                    hc2.metric("AUC", f"{hm['auc']:.3f}")
                    hc3.metric("Brier", f"{hm['brier']:.4f}")
                    hc4.metric("Log loss", f"{hm['logloss']:.4f}")
                    hc5.metric("ROI", f"{hbt['roi']*100:.1f}%")
                    hc6.metric("Paris placés", f"{hbt['n_bets']}")

                    # Le ROI des fenêtres n'est PAS une corroboration: réglé sur ces
                    # mêmes matchs, il est positif par construction ou presque. Il sert
                    # ici de point de comparaison pour mesurer la RETOMBÉE (l'écart
                    # entre in-sample et hors échantillon), pas à confirmer le holdout.
                    roi_windows = agg.loc["mean", WF_ROI_COL]
                    roi_holdout = hbt["roi"] * 100
                    drop = roi_windows - roi_holdout
                    if hbt["n_bets"] == 0:
                        st.info(
                            "Aucun pari déclenché sur la confirmation finale (seuil d'edge jamais atteint) — "
                            "rien à conclure, ni dans un sens ni dans l'autre."
                        )
                    elif hbt["n_bets"] < 30:
                        st.warning(
                            f"ROI de {roi_holdout:.1f}% sur seulement {hbt['n_bets']} paris — l'incertitude sur "
                            "ce chiffre dépasse largement son écart à zéro, quel que soit son signe. Allonge la "
                            "réserve de confirmation avant d'en tirer quoi que ce soit."
                        )
                    elif roi_holdout >= 0:
                        st.success(
                            f"✅ ROI positif sur du jamais-vu : **{roi_holdout:.1f}%** sur {hbt['n_bets']} paris. "
                            f"C'est le seul chiffre de cette page qui vaut comme preuve. (Retombée depuis les "
                            f"fenêtres in-sample : {roi_windows:.1f}% → {roi_holdout:.1f}%, soit {drop:.1f} pts — "
                            "une retombée modérée est normale, une retombée énorme signale du surapprentissage "
                            "de la stratégie de mise.)"
                        )
                    else:
                        st.warning(
                            f"⚠️ ROI négatif sur la confirmation finale : **{roi_holdout:.1f}%** sur "
                            f"{hbt['n_bets']} paris, contre {roi_windows:.1f}% sur les fenêtres in-sample "
                            f"({drop:.1f} pts de retombée). Le rendement observé pendant la recherche ne "
                            "survit pas à du jamais-vu — ne pas exploiter cette config."
                        )


def stored_scope(row) -> tuple:
    """Périmètre de données (surfaces, abandons) enregistré avec un modèle.
    Retourne (None, None) pour les modèles sauvegardés avant l'ajout de ces
    colonnes — l'appelant signale alors que le périmètre est inconnu."""
    raw = row.get("surfaces_json")
    if not raw or pd.isna(raw):
        return None, None
    return json.loads(raw), bool(row.get("exclude_retired"))


def render_saved_walk_forward(algo, params, features, strategy, strategy_source, key_prefix, odds_label,
                               surfaces=None, exclude_retired=None):
    """Validation glissante pour un modèle/stratégie RECHARGÉS depuis la base
    (💾 Modèles sauvegardés / 🎯 Stratégies), par opposition à render_walk_forward_section
    utilisé juste après un entraînement frais (🎲 Bruteforce / 🛠️ Construire un modèle).

    `surfaces`/`exclude_retired` sont le périmètre vu à l'entraînement, relu depuis
    la base (colonnes surfaces_json/exclude_retired, cf. store._MIGRATIONS). Ils
    doivent être réappliqués ici: valider un modèle terre battue sur l'historique
    complet le confronte à des surfaces qu'il n'a jamais apprises, et sa
    confirmation ne mesure alors plus sa qualité mais ce décalage de population.
    À None pour les modèles sauvegardés avant l'ajout de ces colonnes — on retombe
    sur l'historique complet, en le signalant à l'écran."""
    if odds_label is None:
        st.info(
            "Source de cotes introuvable pour ce modèle (sauvegardé avant l'ajout de son suivi) — impossible "
            "de reconstruire le jeu de données pour la validation glissante."
        )
        return

    wf_dataset = get_dataset()
    if surfaces:
        wf_dataset = wf_dataset[wf_dataset["surface"].isin(surfaces)]
        scope = f"surfaces {', '.join(surfaces)}"
        if exclude_retired:
            wf_dataset = wf_dataset[wf_dataset["match_comment"] != RETIRED_COMMENT]
            scope += ", abandons exclus"
        st.caption(f"Périmètre réappliqué depuis l'entraînement : **{scope}**.")
    else:
        st.caption(
            "⚠️ Périmètre d'entraînement (surfaces, abandons) non enregistré pour ce modèle — il a été "
            "sauvegardé avant le suivi de cette information. La validation ci-dessous porte sur l'historique "
            "complet, donc potentiellement sur des surfaces que ce modèle n'a jamais vues : réentraîne-le "
            "depuis l'onglet 🛠️ pour obtenir une confirmation sur la bonne population."
        )

    confirmation_months = st.slider(
        "🔒 Réserve de confirmation finale (mois)", min_value=1, max_value=12, value=6,
        key=f"{key_prefix}_wf_confirmation_months",
    )
    # bornes de la source de cotes SUR CE PÉRIMÈTRE, cohérentes avec celles que
    # run_walk_forward recalculera sur le même jeu (cf. render_general_settings)
    odds_w_col, odds_l_col = ODDS_CHOICES[odds_label]
    _, data_max = usable_date_range(wf_dataset, odds_w_col, odds_l_col)
    if data_max is None:
        st.warning(f"Aucun match avec des cotes « {odds_label} » sur ce périmètre — validation impossible.")
        return
    data_max = data_max.date()
    confirmation_start = (pd.Timestamp(data_max) - pd.DateOffset(months=confirmation_months)).date()
    render_walk_forward_section(
        algo, params, features, strategy, key_prefix=key_prefix, odds_label=odds_label,
        confirmation_start=confirmation_start, confirmation_months=confirmation_months,
        data_max=data_max, wf_dataset=wf_dataset, strategy_source=strategy_source,
    )


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
available_surfaces = [s for s in ["Hard", "Clay", "Grass", "Carpet"] if s in set(dataset["surface"].dropna().unique())]

tab_brute, tab_manual, tab_saved, tab_strategies, tab_upcoming, tab_bankroll, tab_data = st.tabs(
    ["🎲 Bruteforce", "🛠️ Construire un modèle", "💾 Modèles sauvegardés", "🎯 Stratégies",
     "📅 Prochains matchs", "💰 Bankroll", "🗄️ Données"]
)

# ----------------------------------------------------------------------
# Onglet Bruteforce
# ----------------------------------------------------------------------
with tab_brute:
    bf_settings = render_general_settings("bf", min_possible, available_surfaces)
    bf_odds_label = bf_settings["odds_label"]

    st.subheader("Génération aléatoire de modèles")
    c1, c2, c3 = st.columns([1, 1, 1])
    n_models = c1.number_input(
        "Nombre de modèles", min_value=1, max_value=10000, value=25, key="bf_n",
        help="Chaque modèle entraîne + backteste séparément — au-delà de quelques centaines, "
             "le run peut prendre plusieurs minutes.",
    )
    seed = c2.number_input("Seed", min_value=0, value=0, key="bf_seed")
    bf_surface = c3.selectbox(
        "Surface (pour ce bruteforce)", ["Toutes les surfaces sélectionnées"] + bf_settings["selected_surfaces"],
        key="bf_surface",
    )

    bf_train_frame, bf_test_frame = bf_settings["train_frame"], bf_settings["test_frame"]
    if bf_surface != "Toutes les surfaces sélectionnées":
        bf_train_frame = bf_train_frame[bf_train_frame["surface"] == bf_surface]
        bf_test_frame = bf_test_frame[bf_test_frame["surface"] == bf_surface]
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
                qscore = compute_quality_score(m, DEFAULT_WEIGHTS)["score"]
                rows.append({
                    "#": i, "score_qualité": qscore if qscore == qscore else None,
                    "algo": ALGOS[result["algo"]]["label"], "n_features": len(result["features"]),
                    "brier": round(m["brier"], 4) if m["brier"] == m["brier"] else None,
                    "logloss": round(m["logloss"], 4) if m["logloss"] == m["logloss"] else None,
                    "ece": round(m.get("ece", float("nan")), 4) if m.get("ece", float("nan")) == m.get("ece", float("nan")) else None,
                    "auc": round(m["auc"], 3) if m["auc"] == m["auc"] else None,
                    "calibré": m.get("calibrated", False),
                    "n_bets": bt["n_bets"], "roi_%": round(bt["roi"] * 100, 2),
                    "profit": round(bt["total_profit"], 1), "win_rate_%": round(bt["win_rate"] * 100, 1),
                    "max_drawdown_%": round(bt["max_drawdown"] * 100, 1),
                })
                results.append(compact_result(result))
            progress.empty()
            st.session_state["bf_rows"] = rows
            st.session_state["bf_results"] = results
            st.session_state["bf_n_requested"] = int(n_models)

    rows = st.session_state.get("bf_rows", [])
    results = st.session_state.get("bf_results", [])
    n_requested = st.session_state.get("bf_n_requested")
    if rows:
        if n_requested is not None and n_requested != int(n_models):
            st.warning(
                f"⚠️ Le tableau ci-dessous montre les résultats du dernier run ({n_requested} modèles "
                f"demandés), pas encore régénéré pour la valeur actuelle du champ ({int(n_models)}). "
                "Clique sur '🚀 Lancer le bruteforce' pour le mettre à jour."
            )
        elif n_requested is not None and len(rows) != n_requested:
            st.caption(
                f"{len(rows)} modèle(s) valide(s) sur {n_requested} demandé(s) "
                f"({n_requested - len(rows)} config(s) invalide(s), ex: trop peu de features utiles)."
            )
        st.caption(
            "Trié par défaut par score_qualité décroissant (Log loss 32% + Brier 26% + Calibration 21% + "
            "AUC 11% + Nombre de matchs 10%, cf. détail dans la vue d'un modèle) — ce bruteforce ne compare "
            "QUE la qualité prédictive des modèles, avec une stratégie de mise FIXE et identique pour toutes "
            "les lignes (edge ≥ 3%, mise plate) : roi_%/profit servent de référence commune, pas d'optimisation. "
            "Une fois un bon modèle identifié et sauvegardé, utilise le 🎯 Bruteforce ROI (onglet 💾 Modèles "
            "sauvegardés) pour chercher la meilleure stratégie de mise sur CE modèle précis. Clique sur une "
            "ligne pour voir le détail, ou sur un en-tête de colonne pour trier autrement (ex: roi_%)."
        )
        df_rows = pd.DataFrame(rows)
        sorted_df = df_rows.sort_values("score_qualité", ascending=False, na_position="last", kind="stable")
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
            # jeu de test et journal des paris reconstruits ici seulement: les
            # garder pour les 1000 modèles saturait la mémoire (cf. compact_result)
            selected = rehydrate_result(results[idx], bf_test_frame)
            render_result_detail(
                selected, key_prefix=f"bf_{idx}", odds_label=bf_odds_label,
                train_start=bf_settings["train_start"], train_end=bf_settings["train_end"],
                test_start=bf_settings["test_start"], test_end=bf_settings["test_end"],
                surfaces=bf_settings["selected_surfaces"], exclude_retired=bf_settings["exclude_retired"],
            )

            st.divider()
            st.markdown("#### 🎯 Bruteforce ROI — optimiser la stratégie de mise pour ce modèle")
            st.caption(
                "Le modèle ci-dessus reste figé (mêmes probabilités prédites) — seule la stratégie de mise "
                "(seuil d'edge, mode flat/kelly) est testée en masse pour trouver le meilleur ROI. Sauvegarde "
                "d'abord le modèle (bouton 💾 ci-dessus) si tu veux aussi garder une règle de sélection associée — "
                "l'onglet 💾 Modèles sauvegardés le permet. "
                "Sélectionner une règle ci-dessous la fait aussi servir de stratégie de mise à la validation "
                "glissante en bas de page ; sans sélection, celle-ci retombe sur la stratégie de référence du "
                "bruteforce (edge ≥ 3%, mise plate)."
            )
            bf_roi_test_frame = selected["test_frame"]
            # Par défaut la validation glissante plus bas utilise la stratégie de
            # référence du bruteforce modèle (edge 3% / mise plate, identique pour
            # toutes les lignes); sélectionner une règle ROI ci-dessous la remplace.
            bf_wf_strategy = results[idx]["strategy"]
            bf_wf_strategy_source = "stratégie de référence du bruteforce (aucune règle ROI sélectionnée ci-dessus)"
            bfr1, bfr2 = st.columns(2)
            bf_roi_n = bfr1.number_input("Nombre de règles à tester", min_value=10, max_value=2000,
                                         value=300, step=10, key=f"bf_roi_n_{idx}")
            bf_roi_seed = bfr2.number_input("Seed", min_value=0, value=0, key=f"bf_roi_seed_{idx}")

            if st.button("🚀 Lancer le bruteforce ROI", key=f"bf_roi_btn_{idx}"):
                probs = bf_roi_test_frame["model_prob_p1"].values
                bf_roi_rows, bf_roi_strategies = [], []
                progress = st.progress(0.0)
                for i, strat, bt_i, _ in run_roi_bruteforce(probs, bf_roi_test_frame, int(bf_roi_n), seed=int(bf_roi_seed)):
                    progress.progress((i + 1) / bf_roi_n)
                    rscore = compute_roi_score(bt_i)["score"]
                    bf_roi_rows.append({
                        "#": i, "score_roi": rscore if rscore == rscore else None,
                        "edge_threshold": round(strat["edge_threshold"], 4), "stake_mode": strat["stake_mode"],
                        "kelly_fraction": round(strat["kelly_fraction"], 3) if strat["stake_mode"] == "kelly" else None,
                        "min_odds": round(strat["min_odds"], 2) if strat.get("min_odds") is not None else None,
                        "max_odds": round(strat["max_odds"], 2) if strat.get("max_odds") is not None else None,
                        "n_bets": bt_i["n_bets"], "roi_%": round(bt_i["roi"] * 100, 2),
                        "profit": round(bt_i["total_profit"], 1), "win_rate_%": round(bt_i["win_rate"] * 100, 1),
                        "max_drawdown_%": round(bt_i["max_drawdown"] * 100, 1),
                    })
                    bf_roi_strategies.append(strat)
                progress.empty()
                st.session_state[f"bf_roi_rows_{idx}"] = bf_roi_rows
                st.session_state[f"bf_roi_strategies_{idx}"] = bf_roi_strategies

            bf_roi_rows = st.session_state.get(f"bf_roi_rows_{idx}", [])
            bf_roi_strategies = st.session_state.get(f"bf_roi_strategies_{idx}", [])
            if bf_roi_rows:
                st.caption(
                    "Trié par défaut par score_roi décroissant (ROI 50% + nombre de paris 25% + drawdown "
                    "maîtrisé 25%). Clique sur une ligne pour voir le détail."
                )
                bf_roi_df = pd.DataFrame(bf_roi_rows)
                bf_roi_sorted = bf_roi_df.sort_values("score_roi", ascending=False, na_position="last", kind="stable")
                bf_roi_order = bf_roi_sorted.index.to_numpy()
                bf_roi_df = bf_roi_sorted.reset_index(drop=True)
                bf_roi_strategies_sorted = [bf_roi_strategies[i] for i in bf_roi_order]
                bf_roi_event = st.dataframe(
                    bf_roi_df, width="stretch", hide_index=True,
                    on_select="rerun", selection_mode="single-row", key=f"bf_roi_table_{idx}",
                )
                bf_roi_sel = bf_roi_event.selection.rows if bf_roi_event and bf_roi_event.selection else []
                if bf_roi_sel:
                    bf_ridx = bf_roi_sel[0]
                    bf_strat_i = bf_roi_strategies_sorted[bf_ridx]
                    bf_wf_strategy = bf_strat_i
                    bf_wf_strategy_source = (
                        f"règle ROI #{bf_roi_df.iloc[bf_ridx]['#']} sélectionnée ci-dessus"
                    )
                    probs = bf_roi_test_frame["model_prob_p1"].values
                    bf_bets_df_i, bf_bt_i = run_backtest(probs, bf_roi_test_frame, bf_strat_i)
                    st.markdown(f"##### Détail — règle #{bf_roi_df.iloc[bf_ridx]['#']}")
                    render_roi_score(bf_bt_i, key_prefix=f"bf_roi_detail_{idx}_{bf_ridx}")
                    render_backtest_metrics(bf_bt_i)
                    render_equity_curve(bf_bets_df_i, f"bf_roi_{idx}_{bf_ridx}")
                    with st.expander("Règle testée"):
                        st.json(bf_strat_i)

                    saved_model_id = st.session_state.get(f"saved_model_id_bf_{idx}")
                    if saved_model_id is None:
                        st.info(
                            "Sauvegarde d'abord le modèle (bouton 💾 ci-dessus) pour pouvoir aussi "
                            "sauvegarder cette règle de sélection."
                        )
                    else:
                        bf_roi_strat_name = st.text_input(
                            "Nom de la règle de sélection", value=f"bruteforce_#{bf_roi_df.iloc[bf_ridx]['#']}",
                            key=f"bf_roi_name_{idx}_{bf_ridx}",
                        )
                        if st.button("💾 Sauvegarder cette règle", key=f"bf_roi_save_{idx}_{bf_ridx}"):
                            bf_rscore_i = compute_roi_score(bf_bt_i)["score"]
                            store.save_selection_rule(
                                saved_model_id, bf_roi_strat_name, bf_strat_i, bf_bt_i, bf_rscore_i, bf_bets_df_i
                            )
                            st.success("Règle de sélection sauvegardée — visible dans l'onglet 🎯 Stratégies.")

            render_walk_forward_section(
                results[idx]["algo"], results[idx]["params"], results[idx]["features"], bf_wf_strategy,
                key_prefix=f"bf_{idx}", odds_label=bf_odds_label,
                confirmation_start=bf_settings["confirmation_start"], confirmation_months=bf_settings["confirmation_months"],
                data_max=bf_settings["data_max"], wf_dataset=bf_settings["wf_dataset"],
                strategy_source=bf_wf_strategy_source,
            )


# ----------------------------------------------------------------------
# Onglet constructeur manuel
# ----------------------------------------------------------------------
with tab_manual:
    man_settings = render_general_settings("man", min_possible, available_surfaces)
    odds_label = man_settings["odds_label"]
    train_frame, test_frame = man_settings["train_frame"], man_settings["test_frame"]

    st.subheader("Construire un modèle sur-mesure")
    if st.session_state.get("_loaded_from_saved_name"):
        st.info(
            f"Configuration chargée depuis le modèle sauvegardé « "
            f"{st.session_state.pop('_loaded_from_saved_name')} » — ajuste si besoin puis relance "
            "l'entraînement ci-dessous (ça créera un nouveau modèle, l'original n'est pas modifié)."
        )
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
    c5, c6 = st.columns(2)
    min_odds_in = c5.number_input("Cote minimum (1.0 = pas de borne)", min_value=1.0, value=1.0,
                                   step=0.1, key="man_min_odds")
    max_odds_in = c6.number_input("Cote maximum (0 = pas de borne)", min_value=0.0, value=0.0,
                                   step=0.5, key="man_max_odds")
    strategy = {
        "edge_threshold": edge_threshold, "stake_mode": stake_mode,
        "flat_stake": 1.0, "kelly_fraction": kelly_fraction, "bankroll": bankroll,
        "min_odds": min_odds_in if min_odds_in > 1.0 else None,
        "max_odds": max_odds_in if max_odds_in > 0.0 else None,
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

    man_wf_strategy = strategy
    man_wf_strategy_source = None

    if st.session_state.get("manual_result"):
        st.divider()
        man_result = st.session_state["manual_result"]
        render_result_detail(
            man_result, key_prefix="manual", odds_label=odds_label,
            train_start=man_settings["train_start"], train_end=man_settings["train_end"],
            test_start=man_settings["test_start"], test_end=man_settings["test_end"],
            surfaces=man_settings["selected_surfaces"], exclude_retired=man_settings["exclude_retired"],
        )

        st.divider()
        st.markdown("#### 🎯 Bruteforce ROI — optimiser la stratégie de mise pour ce modèle")
        st.caption(
            "Le modèle ci-dessus reste figé (mêmes probabilités prédites) — seule la stratégie de mise "
            "(seuil d'edge, mode flat/kelly, cote min/max) est testée en masse pour trouver le meilleur ROI. "
            "Sauvegarde d'abord le modèle (bouton 💾 ci-dessus) si tu veux aussi garder une règle de sélection "
            "associée — l'onglet 💾 Modèles sauvegardés le permet. "
            "Sélectionner une règle ci-dessous la fait aussi servir de stratégie de mise à la validation "
            "glissante en bas de page ; sans sélection, celle-ci retombe sur la stratégie réglée dans les "
            "sliders ci-dessus."
        )
        man_roi_test_frame = man_result["test_frame"]
        # Par défaut la validation glissante plus bas utilise la stratégie réglée dans
        # les sliders ci-dessus; sélectionner une règle ROI ci-dessous la remplace.
        man_wf_strategy = man_result["strategy"]
        man_wf_strategy_source = "stratégie réglée dans les sliders ci-dessus (aucune règle ROI sélectionnée ci-dessous)"
        mrc1, mrc2 = st.columns(2)
        man_roi_n = mrc1.number_input("Nombre de règles à tester", min_value=10, max_value=2000,
                                      value=300, step=10, key="man_roi_n")
        man_roi_seed = mrc2.number_input("Seed", min_value=0, value=0, key="man_roi_seed")

        if st.button("🚀 Lancer le bruteforce ROI", key="man_roi_btn"):
            probs = man_roi_test_frame["model_prob_p1"].values
            man_roi_rows, man_roi_strategies = [], []
            progress = st.progress(0.0)
            for i, strat, bt_i, _ in run_roi_bruteforce(probs, man_roi_test_frame, int(man_roi_n), seed=int(man_roi_seed)):
                progress.progress((i + 1) / man_roi_n)
                rscore = compute_roi_score(bt_i)["score"]
                man_roi_rows.append({
                    "#": i, "score_roi": rscore if rscore == rscore else None,
                    "edge_threshold": round(strat["edge_threshold"], 4), "stake_mode": strat["stake_mode"],
                    "kelly_fraction": round(strat["kelly_fraction"], 3) if strat["stake_mode"] == "kelly" else None,
                    "min_odds": round(strat["min_odds"], 2) if strat.get("min_odds") is not None else None,
                    "max_odds": round(strat["max_odds"], 2) if strat.get("max_odds") is not None else None,
                    "n_bets": bt_i["n_bets"], "roi_%": round(bt_i["roi"] * 100, 2),
                    "profit": round(bt_i["total_profit"], 1), "win_rate_%": round(bt_i["win_rate"] * 100, 1),
                    "max_drawdown_%": round(bt_i["max_drawdown"] * 100, 1),
                })
                man_roi_strategies.append(strat)
            progress.empty()
            st.session_state["man_roi_rows"] = man_roi_rows
            st.session_state["man_roi_strategies"] = man_roi_strategies

        man_roi_rows = st.session_state.get("man_roi_rows", [])
        man_roi_strategies = st.session_state.get("man_roi_strategies", [])
        if man_roi_rows:
            st.caption(
                "Trié par défaut par score_roi décroissant (ROI 50% + nombre de paris 25% + drawdown "
                "maîtrisé 25%). Clique sur une ligne pour voir le détail."
            )
            man_roi_df = pd.DataFrame(man_roi_rows)
            man_roi_sorted = man_roi_df.sort_values("score_roi", ascending=False, na_position="last", kind="stable")
            man_roi_order = man_roi_sorted.index.to_numpy()
            man_roi_df = man_roi_sorted.reset_index(drop=True)
            man_roi_strategies_sorted = [man_roi_strategies[i] for i in man_roi_order]
            man_roi_event = st.dataframe(
                man_roi_df, width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row", key="man_roi_table",
            )
            man_roi_sel = man_roi_event.selection.rows if man_roi_event and man_roi_event.selection else []
            if man_roi_sel:
                man_ridx = man_roi_sel[0]
                man_strat_i = man_roi_strategies_sorted[man_ridx]
                man_wf_strategy = man_strat_i
                man_wf_strategy_source = f"règle ROI #{man_roi_df.iloc[man_ridx]['#']} sélectionnée ci-dessus"
                probs = man_roi_test_frame["model_prob_p1"].values
                man_bets_df_i, man_bt_i = run_backtest(probs, man_roi_test_frame, man_strat_i)
                st.markdown(f"##### Détail — règle #{man_roi_df.iloc[man_ridx]['#']}")
                render_roi_score(man_bt_i, key_prefix=f"man_roi_detail_{man_ridx}")
                render_backtest_metrics(man_bt_i)
                render_equity_curve(man_bets_df_i, f"man_roi_{man_ridx}")
                with st.expander("Règle testée"):
                    st.json(man_strat_i)

                saved_model_id = st.session_state.get("saved_model_id_manual")
                if saved_model_id is None:
                    st.info(
                        "Sauvegarde d'abord le modèle (bouton 💾 ci-dessus) pour pouvoir aussi "
                        "sauvegarder cette règle de sélection."
                    )
                else:
                    man_roi_strat_name = st.text_input(
                        "Nom de la règle de sélection", value=f"bruteforce_#{man_roi_df.iloc[man_ridx]['#']}",
                        key=f"man_roi_name_{man_ridx}",
                    )
                    if st.button("💾 Sauvegarder cette règle", key=f"man_roi_save_{man_ridx}"):
                        man_rscore_i = compute_roi_score(man_bt_i)["score"]
                        store.save_selection_rule(
                            saved_model_id, man_roi_strat_name, man_strat_i, man_bt_i, man_rscore_i, man_bets_df_i
                        )
                        st.success("Règle de sélection sauvegardée — visible dans l'onglet 🎯 Stratégies.")

    render_walk_forward_section(
        algo_key, params, features, man_wf_strategy, key_prefix="manual", odds_label=odds_label,
        confirmation_start=man_settings["confirmation_start"], confirmation_months=man_settings["confirmation_months"],
        data_max=man_settings["data_max"], wf_dataset=man_settings["wf_dataset"],
        strategy_source=man_wf_strategy_source,
    )


# ----------------------------------------------------------------------
# Onglet modèles sauvegardés
# ----------------------------------------------------------------------
with tab_saved:
    st.subheader("Modèles sauvegardés")
    saved = store.list_models()
    if saved.empty:
        st.info("Aucun modèle sauvegardé pour l'instant.")
    else:
        saved = saved.copy()
        saved["score_qualité"] = saved.apply(
            lambda r: compute_quality_score(
                {"logloss": r["logloss"], "brier": r["brier"], "auc": r["auc"],
                 "ece": r.get("ece"), "n_test": r["n_test"]},
                DEFAULT_WEIGHTS,
            )["score"],
            axis=1,
        )

        # Lit la sélection du tableau depuis SON état précédent (avant de le
        # redessiner plus bas) pour pouvoir afficher les actions de gestion
        # au-dessus de celui-ci plutôt qu'après.
        prev_table_state = st.session_state.get("saved_table")
        prev_sel = prev_table_state.selection.rows if prev_table_state and prev_table_state.selection else []
        if prev_sel:
            row = saved.iloc[prev_sel[0]]
            st.markdown(f"### {row['name']} — {row['algo']}")
            st.markdown("##### Gestion du modèle")
            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                new_name = st.text_input("Nouveau nom", value=row["name"], key=f"rename_input_{row['id']}")
                if st.button("✏️ Renommer", key=f"rename_btn_{row['id']}", disabled=(new_name == row["name"])):
                    store.rename_model(row["id"], new_name)
                    st.rerun()
            with gc2:
                dup_name = st.text_input("Nom de la copie", value=f"{row['name']}_copie", key=f"dup_input_{row['id']}")
                if st.button("📋 Dupliquer", key=f"dup_btn_{row['id']}"):
                    store.duplicate_model(row["id"], dup_name)
                    st.session_state.pop("saved_table", None)  # l'ordre/index des lignes change (nouvelle ligne en tête)
                    st.success("Modèle dupliqué — visible dans le tableau ci-dessous.")
                    st.rerun()
            with gc3:
                st.caption("Modifier hyperparamètres/features/stratégie")
                if st.button("🛠️ Éditer", key=f"edit_btn_{row['id']}",
                             help="Charge cette config dans l'onglet 🛠️ Construire un modèle pour l'ajuster et "
                                  "la réentraîner (crée un nouveau modèle, ne modifie pas celui-ci)."):
                    _load_model_into_manual_constructor(row)
                    st.rerun()
            with gc4:
                st.caption("Irréversible")
                if st.button("🗑️ Supprimer", key=f"del_{row['id']}"):
                    confirm_delete_model(row["id"], row["name"])
            st.divider()

        display_cols = ["id", "name", "algo", "score_qualité", "test_start", "test_end", "n_bets",
                         "roi", "total_profit", "win_rate", "auc", "max_drawdown", "created_at"]
        event = st.dataframe(
            saved[display_cols], width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key="saved_table",
        )
        sel = event.selection.rows if event and event.selection else []
        if sel:
            row = saved.iloc[sel[0]]
            st.divider()

            bt = dict(n_bets=int(row["n_bets"]), roi=row["roi"], total_profit=row["total_profit"],
                      win_rate=row["win_rate"], final_bankroll=row["final_bankroll"], max_drawdown=row["max_drawdown"])
            m = dict(auc=row["auc"], accuracy=row["accuracy"], logloss=row["logloss"], brier=row["brier"])
            render_metrics(m, bt)
            bets_df = store.load_model_bets(row["id"])
            render_equity_curve(bets_df, f"saved_{row['id']}")
            with st.expander("Détails de la configuration"):
                st.json({
                    "hyperparams": json.loads(row["params_json"]),
                    "features": json.loads(row["features_json"]),
                    "strategy": json.loads(row["strategy_json"]),
                })

            st.divider()
            st.markdown("#### 🎯 Optimisation ROI pour ce modèle")
            st.caption(
                "Le modèle reste figé (mêmes probabilités prédites) — seule la stratégie de mise (seuil "
                "d'edge, mode flat/kelly) varie ici. Contrairement au 🎲 Bruteforce (onglet dédié), qui ne "
                "compare que la qualité prédictive des modèles entre eux."
            )
            stored_odds_w, stored_odds_l = row.get("odds_w_col"), row.get("odds_l_col")
            has_stored_odds = bool(stored_odds_w) and pd.notna(stored_odds_w)
            odds_w_override, odds_l_override = None, None
            if not has_stored_odds:
                st.info(
                    "Ce modèle a été sauvegardé avant l'ajout du suivi de la source de cotes — indique "
                    "laquelle a été utilisée à l'entraînement pour pouvoir relancer le backtest."
                )
                odds_override_label = st.selectbox(
                    "Source de cotes utilisée à l'entraînement", list(ODDS_CHOICES.keys()),
                    key=f"roi_odds_{row['id']}",
                )
                odds_w_override, odds_l_override = ODDS_CHOICES[odds_override_label]

            roi_test_frame = st.session_state.get(f"roi_test_frame_{row['id']}")
            rl1, rl2 = st.columns([3, 1])
            rl1.caption(
                f"{'✅ Prédictions chargées (' + str(len(roi_test_frame)) + ' matchs de test).' if roi_test_frame is not None else '⏳ Prédictions pas encore chargées.'}"
            )
            if rl2.button("🔄 Charger/recharger", key=f"roi_reload_{row['id']}"):
                with st.spinner("Reconstruction des prédictions du modèle sur son jeu de test complet..."):
                    reloaded = store.load_test_frame_with_probs(row["id"], dataset, odds_w_override, odds_l_override)
                if reloaded is None:
                    st.error(
                        "Impossible de reconstruire le jeu de test pour ce modèle (source de cotes manquante, "
                        "ou features incompatibles avec la base actuelle)."
                    )
                else:
                    st.session_state[f"roi_test_frame_{row['id']}"] = reloaded[0]
                    st.rerun()

            roi_test_frame = st.session_state.get(f"roi_test_frame_{row['id']}")
            if roi_test_frame is None:
                st.info("Charge d'abord les prédictions du modèle (bouton ci-dessus).")
            else:
                roi_tab_manual, roi_tab_bf, roi_tab_strats = st.tabs(
                    ["🎛️ Simulation manuelle", "🎲 Bruteforce ROI", "💾 Règles de sélection sauvegardées"]
                )

                # ------------------------------------------------------------
                # Simulation ROI manuelle
                # ------------------------------------------------------------
                with roi_tab_manual:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    m_edge = mc1.slider("Seuil d'edge minimum", 0.0, 0.25,
                                        DEFAULT_STRATEGY["edge_threshold"], key=f"roi_man_edge_{row['id']}")
                    m_stake_mode = mc2.radio("Mode de mise", ["flat", "kelly"], key=f"roi_man_stake_{row['id']}")
                    m_kelly = mc3.slider("Fraction de Kelly", 0.05, 1.0,
                                         DEFAULT_STRATEGY["kelly_fraction"], key=f"roi_man_kelly_{row['id']}")
                    m_bankroll = mc4.number_input("Bankroll initiale", min_value=1.0, value=100.0,
                                                  key=f"roi_man_bankroll_{row['id']}")
                    mc5, mc6 = st.columns(2)
                    m_min_odds = mc5.number_input("Cote minimum (1.0 = pas de borne)", min_value=1.0, value=1.0,
                                                  step=0.1, key=f"roi_man_min_odds_{row['id']}")
                    m_max_odds = mc6.number_input("Cote maximum (0 = pas de borne)", min_value=0.0, value=0.0,
                                                  step=0.5, key=f"roi_man_max_odds_{row['id']}")
                    manual_strategy = {
                        "edge_threshold": m_edge, "stake_mode": m_stake_mode, "flat_stake": 1.0,
                        # None en mode flat (non lu par run_backtest) pour ne pas sauvegarder une valeur de
                        # slider non pertinente comme si elle avait un effet — cf. app.roi_bruteforce.sample_strategy.
                        "kelly_fraction": m_kelly if m_stake_mode == "kelly" else None,
                        "bankroll": m_bankroll,
                        "min_odds": m_min_odds if m_min_odds > 1.0 else None,
                        "max_odds": m_max_odds if m_max_odds > 0.0 else None,
                    }

                    if st.button("▶️ Lancer la simulation", key=f"roi_man_run_{row['id']}"):
                        probs = roi_test_frame["model_prob_p1"].values
                        bets_df_m, bt_m = run_backtest(probs, roi_test_frame, manual_strategy)
                        st.session_state[f"roi_man_result_{row['id']}"] = (manual_strategy, bt_m, bets_df_m)

                    man_result = st.session_state.get(f"roi_man_result_{row['id']}")
                    if man_result:
                        strat_m, bt_m, bets_df_m = man_result
                        render_roi_score(bt_m, key_prefix=f"roi_man_score_{row['id']}")
                        render_backtest_metrics(bt_m)
                        render_equity_curve(bets_df_m, f"roi_man_{row['id']}")

                        strat_name = st.text_input(
                            "Nom de la règle de sélection", value=f"manuelle_edge{m_edge:.2f}_{m_stake_mode}",
                            key=f"roi_man_name_{row['id']}",
                        )
                        if st.button("💾 Sauvegarder cette règle", key=f"roi_man_save_{row['id']}"):
                            rscore = compute_roi_score(bt_m)["score"]
                            store.save_selection_rule(row["id"], strat_name, strat_m, bt_m, rscore, bets_df_m)
                            st.success("Règle de sélection sauvegardée — visible dans l'onglet 🎯 Stratégies.")

                # ------------------------------------------------------------
                # Bruteforce ROI
                # ------------------------------------------------------------
                with roi_tab_bf:
                    rc1, rc2 = st.columns(2)
                    roi_n = rc1.number_input("Nombre de règles à tester", min_value=10, max_value=2000,
                                              value=300, step=10, key=f"roi_bf_n_{row['id']}")
                    roi_seed = rc2.number_input("Seed", min_value=0, value=0, key=f"roi_bf_seed_{row['id']}")

                    if st.button("🚀 Lancer le bruteforce ROI", key=f"roi_bf_btn_{row['id']}"):
                        probs = roi_test_frame["model_prob_p1"].values
                        roi_rows, roi_strategies = [], []
                        progress = st.progress(0.0)
                        for i, strat, bt_i, _ in run_roi_bruteforce(probs, roi_test_frame, int(roi_n), seed=int(roi_seed)):
                            progress.progress((i + 1) / roi_n)
                            rscore = compute_roi_score(bt_i)["score"]
                            roi_rows.append({
                                "#": i, "score_roi": rscore if rscore == rscore else None,
                                "edge_threshold": round(strat["edge_threshold"], 4), "stake_mode": strat["stake_mode"],
                                "kelly_fraction": round(strat["kelly_fraction"], 3) if strat["stake_mode"] == "kelly" else None,
                                "min_odds": round(strat["min_odds"], 2) if strat.get("min_odds") is not None else None,
                                "max_odds": round(strat["max_odds"], 2) if strat.get("max_odds") is not None else None,
                                "n_bets": bt_i["n_bets"], "roi_%": round(bt_i["roi"] * 100, 2),
                                "profit": round(bt_i["total_profit"], 1), "win_rate_%": round(bt_i["win_rate"] * 100, 1),
                                "max_drawdown_%": round(bt_i["max_drawdown"] * 100, 1),
                            })
                            roi_strategies.append(strat)
                        progress.empty()
                        st.session_state[f"roi_bf_rows_{row['id']}"] = roi_rows
                        st.session_state[f"roi_bf_strategies_{row['id']}"] = roi_strategies

                    roi_rows = st.session_state.get(f"roi_bf_rows_{row['id']}", [])
                    roi_strategies = st.session_state.get(f"roi_bf_strategies_{row['id']}", [])
                    if roi_rows:
                        st.caption(
                            "Trié par défaut par score_roi décroissant (ROI 50% + nombre de paris 25% + "
                            "drawdown maîtrisé 25%). Clique sur une ligne pour voir le détail."
                        )
                        roi_df = pd.DataFrame(roi_rows)
                        roi_sorted = roi_df.sort_values("score_roi", ascending=False, na_position="last", kind="stable")
                        order = roi_sorted.index.to_numpy()
                        roi_df = roi_sorted.reset_index(drop=True)
                        roi_strategies_sorted = [roi_strategies[i] for i in order]
                        roi_event = st.dataframe(
                            roi_df, width="stretch", hide_index=True,
                            on_select="rerun", selection_mode="single-row", key=f"roi_bf_table_{row['id']}",
                        )
                        roi_sel = roi_event.selection.rows if roi_event and roi_event.selection else []
                        if roi_sel:
                            ridx = roi_sel[0]
                            strat_i = roi_strategies_sorted[ridx]
                            probs = roi_test_frame["model_prob_p1"].values
                            bets_df_i, bt_i = run_backtest(probs, roi_test_frame, strat_i)
                            st.markdown(f"##### Détail — règle #{roi_df.iloc[ridx]['#']}")
                            render_roi_score(bt_i, key_prefix=f"roi_bf_detail_{row['id']}_{ridx}")
                            render_backtest_metrics(bt_i)
                            render_equity_curve(bets_df_i, f"roi_bf_{row['id']}_{ridx}")

                            bf_strat_name = st.text_input(
                                "Nom de la règle de sélection", value=f"bruteforce_#{roi_df.iloc[ridx]['#']}",
                                key=f"roi_bf_name_{row['id']}_{ridx}",
                            )
                            if st.button("💾 Sauvegarder cette règle", key=f"roi_bf_save_{row['id']}_{ridx}"):
                                rscore_i = compute_roi_score(bt_i)["score"]
                                store.save_selection_rule(row["id"], bf_strat_name, strat_i, bt_i, rscore_i, bets_df_i)
                                st.success("Règle de sélection sauvegardée — visible dans l'onglet 🎯 Stratégies.")

                # ------------------------------------------------------------
                # Règles de sélection sauvegardées pour ce modèle
                # ------------------------------------------------------------
                with roi_tab_strats:
                    strats_df = store.list_selection_rules(row["id"])
                    if strats_df.empty:
                        st.info("Aucune règle de sélection sauvegardée pour ce modèle pour l'instant.")
                    else:
                        strat_params = strats_df["strategy_json"].apply(json.loads)
                        strats_df = strats_df.assign(
                            edge_threshold=strat_params.apply(lambda s: s.get("edge_threshold")),
                            stake_mode=strat_params.apply(lambda s: s.get("stake_mode")),
                            kelly_fraction=strat_params.apply(
                                lambda s: s.get("kelly_fraction") if s.get("stake_mode") == "kelly" else None
                            ),
                            min_odds=strat_params.apply(lambda s: s.get("min_odds")),
                            max_odds=strat_params.apply(lambda s: s.get("max_odds")),
                            bankroll=strat_params.apply(lambda s: s.get("bankroll")),
                        )
                        strat_display_cols = ["id", "name", "roi_score", "roi", "n_bets", "total_profit",
                                               "win_rate", "max_drawdown", "edge_threshold", "stake_mode",
                                               "kelly_fraction", "min_odds", "max_odds", "bankroll", "created_at"]
                        strat_event = st.dataframe(
                            strats_df[strat_display_cols], width="stretch", hide_index=True,
                            on_select="rerun", selection_mode="single-row", key=f"roi_strats_table_{row['id']}",
                        )
                        strat_sel = strat_event.selection.rows if strat_event and strat_event.selection else []
                        if strat_sel:
                            srow = strats_df.iloc[strat_sel[0]]
                            st.markdown(f"##### {srow['name']}")
                            with st.expander("Règle de sélection", expanded=True):
                                st.json(json.loads(srow["strategy_json"]))

                            rscore = srow["roi_score"]
                            st.metric("🎯 Score ROI (règle)", f"{rscore:.1f}/100" if rscore == rscore else "n/a")
                            render_backtest_metrics(dict(
                                n_bets=int(srow["n_bets"]), roi=srow["roi"], total_profit=srow["total_profit"],
                                win_rate=srow["win_rate"], final_bankroll=srow["final_bankroll"],
                                max_drawdown=srow["max_drawdown"],
                            ))

                            strat_bets_df = store.load_selection_rule_bets(srow["id"])
                            render_equity_curve(strat_bets_df, f"roi_strat_view_{srow['id']}")
                            if st.button("🗑️ Supprimer cette règle", key=f"del_strat_{srow['id']}"):
                                confirm_delete_selection_rule(srow["id"], srow["name"])

                st.divider()
                st.markdown("#### 🔁 Validation glissante")
                st.caption(
                    "Réentraîne ce modèle sur plusieurs fenêtres temporelles glissantes + une réserve de "
                    "confirmation finale jamais vue pendant sa recherche — pour vérifier que sa performance "
                    "ne tient pas juste à la coupure train/test d'origine (même logique que dans les onglets "
                    "🎲 Bruteforce et 🛠️ Construire un modèle)."
                )
                saved_rules_df = store.list_selection_rules(row["id"])
                wf_rule_names = ["Stratégie d'origine du modèle"] + (
                    list(saved_rules_df["name"]) if not saved_rules_df.empty else []
                )
                wf_rule_choice = st.selectbox(
                    "Stratégie de mise à valider", wf_rule_names, key=f"saved_wf_rule_{row['id']}"
                )
                if wf_rule_choice == "Stratégie d'origine du modèle":
                    wf_strategy = json.loads(row["strategy_json"])
                    wf_strategy_source = "stratégie d'origine du modèle (utilisée à l'entraînement)"
                else:
                    wf_rule_row = saved_rules_df[saved_rules_df["name"] == wf_rule_choice].iloc[0]
                    wf_strategy = json.loads(wf_rule_row["strategy_json"])
                    wf_strategy_source = f"règle de sélection sauvegardée « {wf_rule_choice} »"

                wf_odds_label = (
                    {v: k for k, v in ODDS_CHOICES.items()}.get((stored_odds_w, stored_odds_l))
                    if has_stored_odds else odds_override_label
                )
                wf_surfaces, wf_exclude_retired = stored_scope(row)
                render_saved_walk_forward(
                    row["algo"], json.loads(row["params_json"]), json.loads(row["features_json"]),
                    wf_strategy, wf_strategy_source, key_prefix=f"saved_{row['id']}", odds_label=wf_odds_label,
                    surfaces=wf_surfaces, exclude_retired=wf_exclude_retired,
                )


# ----------------------------------------------------------------------
# Onglet Stratégies
# ----------------------------------------------------------------------
with tab_strategies:
    st.subheader("🎯 Stratégies")
    st.caption(
        "Une stratégie = une association que tu crées toi-même entre un modèle sauvegardé et une de "
        "ses règles de sélection sauvegardées (créées via la simulation manuelle ou le bruteforce ROI, "
        "onglet 💾 Modèles sauvegardés → 🎯 Optimisation ROI). Cette liste est vide tant que tu n'as "
        "rien ajouté ci-dessous."
    )

    strat_models = store.list_models().reset_index(drop=True)
    if strat_models.empty:
        st.info(
            "Aucun modèle sauvegardé pour l'instant — entraîne et sauvegarde un modèle dans l'onglet "
            "'🛠️ Construire un modèle'."
        )
    else:
        st.markdown("#### ➕ Ajouter une stratégie")
        new_strat_model_labels = [f"{r['name']} ({r['algo']}, AUC {r['auc']:.3f})" for _, r in strat_models.iterrows()]
        new_strat_model_idx = st.selectbox(
            "Modèle", range(len(new_strat_model_labels)), format_func=lambda i: new_strat_model_labels[i],
            key="new_strat_model_select",
        )
        new_strat_model_row = strat_models.iloc[new_strat_model_idx]
        new_strat_model_id = new_strat_model_row["id"]

        new_strat_rules = store.list_selection_rules(new_strat_model_id).reset_index(drop=True)
        if new_strat_rules.empty:
            st.warning(
                "Ce modèle n'a aucune règle de sélection sauvegardée — crée-en une d'abord dans "
                "l'onglet '💾 Modèles sauvegardés' → section '🎯 Optimisation ROI pour ce modèle'."
            )
        else:
            new_rule_labels = [
                f"{r['name']} (ROI {r['roi']*100:.1f}%, {r['n_bets']} paris)" for _, r in new_strat_rules.iterrows()
            ]
            new_rule_idx = st.selectbox(
                "Règle de sélection", range(len(new_rule_labels)), format_func=lambda i: new_rule_labels[i],
                key="new_strat_rule_select",
            )
            new_rule_row = new_strat_rules.iloc[new_rule_idx]
            new_strat_name = st.text_input(
                "Nom de la stratégie", value=f"{new_strat_model_row['name']} — {new_rule_row['name']}",
                key="new_strat_name",
            )
            if st.button("➕ Créer la stratégie", key="new_strat_create_btn", type="primary"):
                store.create_strategy(new_strat_name, new_strat_model_id, new_rule_row["id"])
                st.success("Stratégie créée.")
                st.rerun()

        st.divider()
        st.markdown("#### Mes stratégies")
        strategies_df = store.list_strategies()
        if strategies_df.empty:
            st.info("Aucune stratégie créée pour l'instant — utilise le formulaire ci-dessus.")
        else:
            strat_params = strategies_df["strategy_json"].apply(json.loads)
            strategies_df = strategies_df.assign(
                edge_threshold=strat_params.apply(lambda s: s.get("edge_threshold")),
                stake_mode=strat_params.apply(lambda s: s.get("stake_mode")),
                kelly_fraction=strat_params.apply(
                    lambda s: s.get("kelly_fraction") if s.get("stake_mode") == "kelly" else None
                ),
                min_odds=strat_params.apply(lambda s: s.get("min_odds")),
                max_odds=strat_params.apply(lambda s: s.get("max_odds")),
                bankroll=strat_params.apply(lambda s: s.get("bankroll")),
            )
            strat_display_cols = ["name", "model_name", "algo", "auc", "rule_name", "roi_score", "roi",
                                   "n_bets", "total_profit", "win_rate", "max_drawdown",
                                   "edge_threshold", "stake_mode", "kelly_fraction", "min_odds", "max_odds",
                                   "bankroll", "created_at"]
            strat_event = st.dataframe(
                strategies_df[strat_display_cols].rename(columns={
                    "name": "Stratégie", "model_name": "Modèle", "algo": "Algo", "rule_name": "Règle de sélection",
                }),
                width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row", key="strategies_table",
            )
            strat_sel = strat_event.selection.rows if strat_event and strat_event.selection else []
            if strat_sel:
                srow = strategies_df.iloc[strat_sel[0]]
                st.divider()
                st.markdown(f"##### {srow['name']}")
                with st.expander("Détail", expanded=True):
                    st.write(f"**Modèle**: {srow['model_name']} ({srow['algo']})")
                    st.write(f"**Règle de sélection**: {srow['rule_name']}")
                    st.json(json.loads(srow["strategy_json"]))
                render_backtest_metrics(dict(
                    n_bets=int(srow["n_bets"]), roi=srow["roi"], total_profit=srow["total_profit"],
                    win_rate=srow["win_rate"], final_bankroll=srow["final_bankroll"],
                    max_drawdown=srow["max_drawdown"],
                ))

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    ren_name = st.text_input("Nouveau nom", value=srow["name"], key=f"strat_ren_input_{srow['id']}")
                    if st.button("✏️ Renommer", key=f"strat_ren_btn_{srow['id']}",
                                 disabled=(ren_name == srow["name"])):
                        store.rename_strategy(srow["id"], ren_name)
                        st.rerun()
                with sc2:
                    st.caption("Changer le modèle/la règle")
                    if st.button("🛠️ Éditer", key=f"strat_edit_open_{srow['id']}"):
                        st.session_state["strat_editing_id"] = srow["id"]
                with sc3:
                    st.caption("Irréversible")
                    if st.button("🗑️ Supprimer", key=f"strat_del_btn_{srow['id']}"):
                        store.delete_strategy(srow["id"])
                        st.rerun()

                st.divider()
                st.markdown("#### 🔁 Validation glissante")
                st.caption(
                    "Réentraîne le modèle associé sur plusieurs fenêtres temporelles glissantes + une réserve "
                    "de confirmation finale jamais vue pendant sa recherche, avec CETTE règle de sélection — "
                    "pour vérifier que cette stratégie tient au-delà de la coupure train/test d'origine."
                )
                strat_model_row = strat_models[strat_models["id"] == srow["model_id"]].iloc[0]
                strat_odds_label = None
                strat_odds_w, strat_odds_l = strat_model_row.get("odds_w_col"), strat_model_row.get("odds_l_col")
                if strat_odds_w and pd.notna(strat_odds_w):
                    strat_odds_label = {v: k for k, v in ODDS_CHOICES.items()}.get((strat_odds_w, strat_odds_l))
                strat_surfaces, strat_exclude_retired = stored_scope(strat_model_row)
                render_saved_walk_forward(
                    strat_model_row["algo"], json.loads(strat_model_row["params_json"]),
                    json.loads(strat_model_row["features_json"]), json.loads(srow["strategy_json"]),
                    f"règle « {srow['rule_name']} » de la stratégie « {srow['name']} »",
                    key_prefix=f"strat_{srow['id']}", odds_label=strat_odds_label,
                    surfaces=strat_surfaces, exclude_retired=strat_exclude_retired,
                )

                if st.session_state.get("strat_editing_id") == srow["id"]:
                    with st.expander("🛠️ Éditer cette stratégie", expanded=True):
                        edit_model_labels = [
                            f"{r['name']} ({r['algo']})" for _, r in strat_models.iterrows()
                        ]
                        cur_model_pos = int(strat_models.index[strat_models["id"] == srow["model_id"]][0])
                        edit_model_idx = st.selectbox(
                            "Modèle", range(len(edit_model_labels)), index=cur_model_pos,
                            format_func=lambda i: edit_model_labels[i], key=f"strat_edit_model_{srow['id']}",
                        )
                        edit_model_id = strat_models.iloc[edit_model_idx]["id"]

                        edit_rules = store.list_selection_rules(edit_model_id).reset_index(drop=True)
                        if edit_rules.empty:
                            st.warning("Ce modèle n'a aucune règle de sélection sauvegardée.")
                        else:
                            edit_rule_labels = [
                                f"{r['name']} (ROI {r['roi']*100:.1f}%, {r['n_bets']} paris)"
                                for _, r in edit_rules.iterrows()
                            ]
                            if edit_model_id == srow["model_id"] and srow["rule_id"] in edit_rules["id"].values:
                                cur_rule_pos = int(edit_rules.index[edit_rules["id"] == srow["rule_id"]][0])
                            else:
                                cur_rule_pos = 0
                            edit_rule_idx = st.selectbox(
                                "Règle de sélection", range(len(edit_rule_labels)), index=cur_rule_pos,
                                format_func=lambda i: edit_rule_labels[i], key=f"strat_edit_rule_{srow['id']}",
                            )
                            edit_rule_id = edit_rules.iloc[edit_rule_idx]["id"]

                            ec1, ec2 = st.columns(2)
                            if ec1.button("💾 Enregistrer", key=f"strat_edit_save_{srow['id']}", type="primary"):
                                store.update_strategy(srow["id"], edit_model_id, edit_rule_id)
                                st.session_state.pop("strat_editing_id", None)
                                st.rerun()
                            if ec2.button("Annuler", key=f"strat_edit_cancel_{srow['id']}"):
                                st.session_state.pop("strat_editing_id", None)
                                st.rerun()


# ----------------------------------------------------------------------
# Onglet Prochains matchs
# ----------------------------------------------------------------------
with tab_upcoming:
    render_upcoming(dataset)


# ----------------------------------------------------------------------
# Onglet Bankroll
# ----------------------------------------------------------------------
with tab_bankroll:
    render_bankroll(dataset)


# ----------------------------------------------------------------------
# Onglet Données
# ----------------------------------------------------------------------
with tab_data:
    render_data_management()
    st.divider()
    render_review_queue()
    st.divider()
    render_db_explorer()
