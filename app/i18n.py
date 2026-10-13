"""Lightweight French/English translations for the Streamlit interface."""

from __future__ import annotations

from functools import wraps
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

_LANGUAGE_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "ui_preferences.json"

_TRANSLATIONS = {
    "Langue": "Language",
    "Tennis ML Lab": "Tennis ML Lab",
    "Régression logistique": "Logistic regression",
    "Arbre de décision": "Decision tree",
    "Forêt aléatoire": "Random forest",
    "Gradient Boosting": "Gradient boosting",
    "K plus proches voisins": "K-nearest neighbors",
    "Toutes": "All",
    "Tous": "All",
    "Toutes les surfaces sélectionnées": "All selected surfaces",
    "Comparer les deux": "Compare both",
    "Fenêtre": "Window",
    "Tournois": "Tournaments",
    "Raison": "Reason",
    "Similarité": "Similarity",
    "A: tournoi": "A: tournament",
    "A: tour": "A: round",
    "A: vainqueur": "A: winner",
    "A: perdant": "A: loser",
    "B: tournoi": "B: tournament",
    "B: tour": "B: round",
    "B: vainqueur": "B: winner",
    "B: perdant": "B: loser",
    "Valeur": "Value",
    "Proba": "Probability",
    "Significatif": "Significant",
    "Pariable live": "Live bettable",
    "Niveau": "Level",
    "Features (variables du modèle)": "Model features",
    "Hyperparamètres par algorithme": "Hyperparameters by algorithm",
    "Stratégie de mise (backtest)": "Betting strategy (backtest)",
    "Glossaire des paramètres": "Parameter glossary",
    "Définitions des variables, hyperparamètres et métriques utilisés dans l'application.": (
        "Definitions of the features, hyperparameters, and metrics used in the application."
    ),
    "🎲 Bruteforce": "🎲 Brute force",
    "🛠️ Construire un modèle": "🛠️ Build a model",
    "Construire un modèle sur-mesure": "Build a custom model",
    "💾 Modèles sauvegardés": "💾 Saved models",
    "🎯 Stratégies": "🎯 Strategies",
    "📅 Prochains matchs": "📅 Upcoming matches",
    "🗄️ Données": "🗄️ Data",
    "💰 Bankroll": "💰 Bankroll",
    "🎛️ Simulation manuelle": "🎛️ Manual simulation",
    "🎲 Bruteforce ROI": "🎲 ROI brute force",
    "💾 Règles de sélection sauvegardées": "💾 Saved selection rules",
    "Backtesting de modèles de prédiction sur data/tennis.db": (
        "Prediction model backtesting on data/tennis.db"
    ),
    "Lancer le bruteforce": "Run brute force",
    "🚀 Lancer le bruteforce": "🚀 Run brute force",
    "Paramètres généraux": "General settings",
    "⚙️ Paramètres généraux": "⚙️ General settings",
    "Moyenne du marché (Avg)": "Market average (Avg)",
    "Source de cotes (marché de référence)": "Odds source (reference market)",
    "Réserve de confirmation finale (mois)": "Final confirmation holdout (months)",
    "Surfaces à inclure": "Surfaces to include",
    "Exclure les matchs abandonnés en cours de jeu": "Exclude matches retired during play",
    "Début période d'entraînement": "Training period start",
    "Début période de test (backtest)": "Test period start (backtest)",
    "Fin période de test": "Test period end",
    "Algorithme": "Algorithm",
    "Algorithmes autorisés (tirés au hasard parmi ceux cochés)": (
        "Allowed algorithms (randomly selected from checked options)"
    ),
    "Features autorisées (tirées au hasard, en sous-ensemble, parmi celles cochées)": (
        "Allowed features (randomly selected as a subset of checked options)"
    ),
    "Features utilisées": "Features used",
    "Hyperparamètres": "Hyperparameters",
    "Nombre de modèles": "Number of models",
    "Nombre de fenêtres": "Number of windows",
    "Durée de test par fenêtre (mois)": "Test duration per window (months)",
    "Entraînement minimum (mois)": "Minimum training period (months)",
    "Nombre de règles à tester": "Number of rules to test",
    "Nombre de cas affichés à la fois": "Number of cases to display at a time",
    "Bankroll initiale": "Initial bankroll",
    "Bankroll de départ (unités)": "Starting bankroll (units)",
    "Bankroll finale": "Final bankroll",
    "Mode de mise": "Staking mode",
    "Fraction de Kelly": "Kelly fraction",
    "Cote minimum (1.0 = pas de borne)": "Minimum odds (1.0 = no limit)",
    "Cote maximum (0 = pas de borne)": "Maximum odds (0 = no limit)",
    "Seuil d'edge minimum": "Minimum edge threshold",
    "Seuil d'edge": "Edge threshold",
    "Nom du modèle": "Model name",
    "Nom de la copie": "Copy name",
    "Nom de la règle de sélection": "Selection rule name",
    "Nom de la stratégie": "Strategy name",
    "Nouveau nom": "New name",
    "Détails de la configuration": "Configuration details",
    "Détail de la configuration": "Configuration details",
    "Détail du score ROI": "ROI score details",
    "Détail du score de qualité": "Quality score details",
    "Détail": "Details",
    "Modèles sauvegardés": "Saved models",
    "Modèle": "Model",
    "Règle de sélection": "Selection rule",
    "Stratégie de mise": "Betting strategy",
    "Modifier hyperparamètres/features/stratégie": "Edit hyperparameters/features/strategy",
    "Changer le modèle/la règle": "Change model/rule",
    "Nom du modèle": "Model name",
    "Entraînement en cours...": "Training...",
    "Construction du jeu d'entraînement...": "Building the training dataset...",
    "Chargement des matchs et calcul des features (Elo, forme, H2H)...": (
        "Loading matches and calculating features (Elo, form, H2H)..."
    ),
    "Récupération du calendrier ATP (ESPN)...": "Fetching the ATP schedule (ESPN)...",
    "Calcul des stats courantes des joueurs...": "Calculating current player statistics...",
    "AUC": "AUC",
    "AUC moyen (± écart-type)": "Mean AUC (± standard deviation)",
    "Accuracy": "Accuracy",
    "Brier": "Brier",
    "Brier score": "Brier score",
    "Brier moyen (± écart-type)": "Mean Brier score (± standard deviation)",
    "Log loss": "Log loss",
    "ECE (calibration)": "ECE (calibration)",
    "Drawdown max": "Maximum drawdown",
    "ROI": "ROI",
    "Profit": "Profit",
    "Taux de réussite": "Win rate",
    "Paris placés": "Bets placed",
    "Paris réglés": "Settled bets",
    "Paris enregistrés": "Recorded bets",
    "En attente": "Pending",
    "Réglés": "Settled",
    "Répartition du profit": "Profit breakdown",
    "Rendement/capital": "Return on capital",
    "Rendement/risque (Calmar)": "Return/risk (Calmar)",
    "Cote de paiement": "Settlement odds",
    "Mise flat (unités)": "Flat stake (units)",
    "Mise": "Stake",
    "Cote moyenne": "Average odds",
    "Cote moy.": "Average odds",
    "Date du match": "Match date",
    "Date": "Date",
    "Tournoi": "Tournament",
    "Adversaire": "Opponent",
    "Proba modèle": "Model probability",
    "Cote signal": "Signal odds",
    "Cote exécution": "Execution odds",
    "Cote d'exécution (meilleur marché)": "Execution odds (best market)",
    "Cote du signal (bookmaker)": "Signal odds (bookmaker)",
    "Résultat": "Result",
    "Stat": "Statistic",
    "Surface": "Surface",
    "Dernier match en base": "Most recent match in database",
    "Dernière consolidation": "Last consolidation",
    "Matchs en base": "Matches in database",
    "Répartition par source (nombre de matchs et dernière date)": (
        "Breakdown by source (number of matches and latest date)"
    ),
    "Gestion des données": "Data management",
    "Mettre à jour la base": "Update the database",
    "Mettre à jour la base de données": "Update the database",
    "État des dépôts git sources": "Source Git repositories",
    "Explorateur de base de données": "Database explorer",
    "Cas ambigus à valider": "Ambiguous cases to review",
    "Étape": "Stage",
    "Statut": "Status",
    "Décision": "Decision",
    "Recherche (joueur ou tournoi)": "Search (player or tournament)",
    "Recherche texte (dans les colonnes texte affichées, ex: un nom de joueur)": (
        "Text search (in displayed text columns, e.g. a player's name)"
    ),
    "Cas affichés à la fois": "Cases displayed at a time",
    "Colonnes affichées": "Columns to display",
    "Nb de lignes max": "Maximum number of rows",
    "Table": "Table",
    "SQL": "SQL",
    "Requête SQL personnalisée (lecture seule)": "Custom SQL query (read-only)",
    "Lecture seule : aucune requête ci-dessous ne peut modifier data/tennis.db.": (
        "Read-only: none of the queries below can modify data/tennis.db."
    ),
    "▶️ Exécuter": "▶️ Run",
    "💾 Enregistrer les décisions": "💾 Save decisions",
    "🔁 Relancer la consolidation": "🔁 Run consolidation again",
    "🔄 Mettre à jour la base de données": "🔄 Update the database",
    "Rafraîchir la page": "Refresh page",
    "Aucun pari pour cette stratégie.": "No bets for this strategy.",
    "Aucun pari enregistré pour cette stratégie pour l'instant.": (
        "No bets have been recorded for this strategy yet."
    ),
    "Aucun modèle sauvegardé pour l'instant.": "No saved models yet.",
    "Aucune stratégie créée pour l'instant — utilise le formulaire ci-dessus.": (
        "No strategies created yet — use the form above."
    ),
    "Aucune règle de sélection sauvegardée pour ce modèle pour l'instant.": (
        "No selection rules have been saved for this model yet."
    ),
    "Aucun pari placé avec cette configuration (seuil d'edge jamais atteint).": (
        "No bets were placed with this configuration (the edge threshold was never reached)."
    ),
    "Aucune fenêtre exploitable avec ces paramètres (pas assez de matchs avec cotes).": (
        "No usable windows with these settings (not enough matches with odds)."
    ),
    "Annuler": "Cancel",
    "Enregistrer": "Save",
    "Éditer": "Edit",
    "Renommer": "Rename",
    "Supprimer": "Delete",
    "🗑️ Supprimer définitivement": "🗑️ Delete permanently",
    "💾 Sauvegarder ce modèle": "💾 Save this model",
    "💾 Sauvegarder cette règle": "💾 Save this rule",
    "▶️ Lancer la simulation": "▶️ Run simulation",
    "🚀 Lancer le bruteforce ROI": "🚀 Run ROI brute force",
    "🔄 Charger/recharger": "🔄 Load/reload",
    "⬇️ Exporter le journal (CSV)": "⬇️ Export log (CSV)",
    "🧾 Journal des paris de la stratégie (onglet 💰 Bankroll)": (
        "🧾 Strategy bet log (💰 Bankroll tab)"
    ),
    "📝 Journal des prix de marché (constitution de l'historique Polymarket)": (
        "📝 Market price log (building Polymarket history)"
    ),
    "Enregistrer les signaux comme paris flat": "Record signals as flat bets",
    "Enregistrer les prix des matchs affichés": "Record prices for displayed matches",
    "💰 Simulation de bankroll": "💰 Bankroll simulation",
    "### Mettre à jour la base": "### Update the database",
    "#### Paramètres de simulation": "#### Simulation settings",
    "#### Paris pris": "#### Bets placed",
    "#### Mes stratégies": "#### My strategies",
    "#### ➕ Ajouter une stratégie": "#### ➕ Add a strategy",
    "#### 🎯 Optimisation ROI pour ce modèle": "#### 🎯 ROI optimization for this model",
    "#### 🎯 Bruteforce ROI — optimiser la stratégie de mise pour ce modèle": (
        "#### 🎯 ROI brute force — optimize this model's betting strategy"
    ),
    "#### 🔁 Validation glissante": "#### 🔁 Walk-forward validation",
    "#### 🔁 Validation glissante (walk-forward)": "#### 🔁 Walk-forward validation",
    "### 🔒 Confirmation finale — jamais vue pendant la recherche": (
        "### 🔒 Final confirmation — never seen during search"
    ),
    "Construire un modèle sur-mesure": "Build a custom model",
    "Surface supposée :": "Assumed surface:",
    "Noms retrouvés dans la base :": "Names found in the database:",
    "Classement ATP": "ATP ranking",
    "Points ATP": "ATP points",
    "Âge (dernier connu)": "Age (most recent)",
    "Taille (cm)": "Height (cm)",
    "Main": "Hand",
    "Forme (10 derniers)": "Form (last 10)",
    "Matchs joués (historique)": "Matches played (history)",
    "Dernier match connu": "Most recent match",
    "Comparaison des sources": "Source comparison",
    "Stratégie / modèle appliqué aux matchs à venir": "Strategy/model applied to upcoming matches",
    "Prédiction d'un modèle sauvegardé": "Prediction from a saved model",
    "Données utilisées pour ce bruteforce :": "Data used for this brute force:",
    "#### ⚖️ Comparaison des sources": "#### ⚖️ Source comparison",
    "#### 🎯 Stratégie / modèle appliqué aux matchs à venir": (
        "#### 🎯 Strategy/model applied to upcoming matches"
    ),
    "#### 🤖 Prédiction d'un modèle sauvegardé": "#### 🤖 Prediction from a saved model",
    "##### Gestion du modèle": "##### Model management",
    "##### Détail — règle #": "##### Details — rule #",
    "### Détail — modèle #": "### Details — model #",
    "Données utilisées pour ce bruteforce :": "Data used for this brute force:",
    "Surface (pour ce bruteforce)": "Surface (for this brute force)",
    "Stratégie de mise à valider": "Betting strategy to validate",
    "Source de cotes utilisée à l'entraînement": "Odds source used for training",
    "Seed": "Seed",
    "Validation glissante en cours (plusieurs entraînements successifs)...": (
        "Walk-forward validation in progress (multiple sequential training runs)..."
    ),
    "Reconstruction des prédictions du modèle sur son jeu de test complet...": (
        "Rebuilding the model's predictions for its full test dataset..."
    ),
    "Tous les cas ont été tranchés — relance la consolidation ci-dessous pour appliquer les décisions.": (
        "All cases have been reviewed — run consolidation below to apply the decisions."
    ),
    "Relancer seulement la consolidation (rapide, sans re-télécharger les sources)": (
        "Run consolidation only (fast; does not download the sources again)"
    ),
    "Règle testée": "Tested rule",
    "Je confirme vouloir tout effacer": "I confirm that I want to delete everything",
    "🗑️ Effacer définitivement": "🗑️ Delete permanently",
    "🗑️ Effacer l'historique de cette stratégie": "🗑️ Clear this strategy's history",
    "🔄 Mettre à jour les résultats": "🔄 Update results",
    "Cote de paiement": "Settlement odds",
    "Comparaison des sources": "Source comparison",
}


_TRANSLATIONS.update({
    "Métriques affichées": "Displayed metrics",
    "C (régression logistique)": "C (logistic regression)",
    "max_depth (arbre, forêt, boosting)": "max_depth (tree, forest, boosting)",
    "min_samples_leaf": "min_samples_leaf",
    "n_estimators (forêt, boosting)": "n_estimators (forest, boosting)",
    "learning_rate (gradient boosting)": "learning_rate (gradient boosting)",
    "n_neighbors (KNN)": "n_neighbors (KNN)",
    "Seuil d'edge (edge_threshold)": "Edge threshold (edge_threshold)",
    "Mode de mise — flat": "Staking mode — flat",
    "Mode de mise — kelly": "Staking mode — Kelly",
    "Cote minimum / maximum (min_odds / max_odds)": "Minimum / maximum odds (min_odds / max_odds)",
    "Paris placés (n_bets)": "Bets placed (n_bets)",
    "Taux de réussite (win_rate)": "Win rate (win_rate)",
    "Probability": "Probability",
    "Elo global": "Overall Elo",
    "Elo global (pondéré récence)": "Overall Elo (recency-weighted)",
    "Classement ATP": "ATP ranking",
    "Points ATP": "ATP points",
    "Âge (dernier connu)": "Age (most recent)",
    "Taille (cm)": "Height (cm)",
    "Forme (10 derniers)": "Form (last 10)",
    "Matchs joués (historique)": "Matches played (history)",
    "Dernier match connu": "Most recent match",
    "Régression logistique": "Logistic regression",
    "Arbre de décision": "Decision tree",
    "Forêt aléatoire": "Random forest",
    "K plus proches voisins": "K-nearest neighbors",
    "Gagné": "Won",
    "Perdu": "Lost",
    "En attente": "Pending",
    "Aucune": "None",
    "Toutes les surfaces sélectionnées": "All selected surfaces",
    "Fournisseur de cotes bookmaker (en plus de Polymarket)": (
        "Bookmaker odds provider (in addition to Polymarket)"
    ),
    "Modèle pour les colonnes de prédiction": "Model for prediction columns",
    "Fenêtre (jours)": "Window (days)",
    "Masquer les 'TBD'": "Hide 'TBD' matches",
    "Rafraîchir le calendrier": "Refresh schedule",
    "Aucune source de cotes/probabilités disponible pour ce match pour l'instant.": (
        "No odds/probability sources are currently available for this match."
    ),
    "Détail des features utilisées par ce modèle": "Details of the features used by this model",
    "Prédiction impossible : données manquantes pour": "Prediction unavailable: missing data for",
    "Impossible de charger ce modèle sauvegardé (fichier introuvable).": (
        "Cannot load this saved model (file not found)."
    ),
    "Impossible de récupérer le calendrier ESPN pour la fenêtre demandée.": (
        "Cannot retrieve the ESPN schedule for the requested window."
    ),
    "Aucun match à venir trouvé (tournois ATP en pause, ou API indisponible).": (
        "No upcoming matches found (ATP tournaments may be paused, or the API is unavailable)."
    ),
    "Calendrier en direct via l'API publique ESPN — tournois ATP en cours ou à venir. Heures affichées en Europe/Paris. Le tableau d'un tournoi n'est publié par l'ATP que quelques jours avant son début : au-delà, les matchs existent déjà (bon nombre de 'Round 1', 'Round 2'...) mais les joueurs restent en 'TBD' tant que le tirage au sort n'est pas sorti. Sélectionne une ligne dans un tableau pour voir le détail du match.": (
        "Live schedule from the public ESPN API — current and upcoming ATP tournaments. Times are shown in Europe/Paris. The ATP only publishes a tournament draw a few days before it starts: matches may already be listed (such as 'Round 1' and 'Round 2'), but players remain 'TBD' until the draw is released. Select a row in a table to see match details."
    ),
    "📊 Polymarket (marché prédictif, sans bookmaker) — toujours affiché, couvre tour principal ET Challengers, mais un marché n'existe que si le tirage au sort du tour est déjà sorti. Probabilité implicite (%) + cote décimale équivalente (1/proba). ": (
        "📊 Polymarket (prediction market, no bookmaker) — always shown and covers both the main tour and Challengers, but a market is available only after the draw for that round is published. Implied probability (%) and equivalent decimal odds (1/probability). "
    ),
    "Aucun modèle sauvegardé — entraîne et sauvegarde un modèle dans l'onglet '🛠️ Construire un modèle' pour afficher ses prédictions ici.": (
        "No saved model — train and save one in the '🛠️ Build a model' tab to show its predictions here."
    ),
    "Aucune stratégie créée — associe un modèle à une règle de sélection dans l'onglet '🎯 Stratégies' pour faire apparaître ici les opportunités de pari.": (
        "No strategy created — link a model to a selection rule in the '🎯 Strategies' tab to see betting opportunities here."
    ),
    "Aucun modèle sauvegardé, ou aucun sélectionné — choisis-en un dans le menu déroulant '🤖 Modèle pour les colonnes de prédiction' au-dessus du tableau.": (
        "No saved model, or none selected — choose one from the '🤖 Model for prediction columns' menu above the table."
    ),
    "👉": "👉",
    "Match": "Match",
    "Roi": "King",
    "Joueur": "Player",
    "Joueurs": "Players",
    "Round": "Round",
})

_TRANSLATIONS.update({
    "Écart de rating Elo global entre p1 et p2, calculé match après match sur tout l'historique "
    "(comme aux échecs: un joueur gagne des points en battant un adversaire mieux classé que lui). "
    "Positif = p1 est jugé plus fort. Ex: +200 signifie un net avantage pour p1 (~76% de proba théorique).": (
        "Difference in overall Elo rating between p1 and p2, calculated match by match across the full history "
        "(as in chess: a player gains points by beating a higher-rated opponent). Positive means p1 is considered "
        "stronger. For example, +200 indicates a clear advantage for p1 (~76% theoretical win probability)."
    ),
    "Même principe que elo_diff, mais le rating est recalculé séparément par surface "
    "(un joueur peut être fort sur terre battue et faible sur gazon). Capture la spécialisation surface.": (
        "Same principle as elo_diff, but the rating is recalculated separately for each surface "
        "(a player may be strong on clay and weak on grass). Captures surface specialization."
    ),
    "Comme elo_diff, mais régressé vers la moyenne (1500) en fonction du temps écoulé depuis "
    "le dernier match connu du joueur (demi-vie 1 an): un rating vieux de plusieurs années pèse "
    "moins qu'un rating tout juste mis à jour. Utile pour ne pas surestimer un joueur inactif.": (
        "Like elo_diff, but regressed toward the mean (1500) based on the time since the player's last known match "
        "(one-year half-life): a rating from several years ago carries less weight than a recently updated one. "
        "Helps avoid overrating an inactive player."
    ),
    "Comme elo_surface_diff, mais avec la même régression temporelle appliquée à la "
    "dernière fois où le joueur a joué SUR CETTE SURFACE précise (un joueur peut être "
    "très actif en général mais n'avoir pas touché la terre battue depuis longtemps).": (
        "Like elo_surface_diff, with the same time-based regression applied since the player last competed on "
        "THIS specific surface (a player may be very active overall but have not played on clay for a long time)."
    ),
    "Développe automatiquement 3 features one-hot (surface_clay, surface_grass, surface_hard — Carpet "
    "en catégorie de référence implicite). Permet au modèle d'apprendre un ajustement direct par "
    "surface, en complément (pas en remplacement) de elo_surface_diff.": (
        "Automatically expands into three one-hot features (surface_clay, surface_grass, surface_hard; Carpet "
        "is the implicit reference category). Lets the model learn a direct surface adjustment, in addition to "
        "(not instead of) elo_surface_diff."
    ),
    "Écart de classement ATP officiel (rang p2 - rang p1). Positif = p1 mieux classé (rang plus petit). "
    "Ex: p1 classé 10, p2 classé 50 -> rank_diff = +40.": (
        "Difference in official ATP ranking (p2 rank minus p1 rank). Positive means p1 is ranked higher (has a "
        "lower rank number). For example, p1 ranked 10 and p2 ranked 50 gives rank_diff = +40."
    ),
    "Écart de points au classement ATP (p1 - p2). Plus fin que rank_diff car reflète l'écart de "
    "niveau réel (le rang seul écrase les différences en fin de classement).": (
        "Difference in ATP ranking points (p1 minus p2). More granular than rank_diff because it reflects the "
        "actual skill gap (rank alone compresses differences near the bottom of the rankings)."
    ),
    "Écart d'âge en années (p1 - p2). Peut capturer un effet 'expérience vs fraîcheur physique'.": (
        "Age difference in years (p1 minus p2). May capture an experience-versus-physical-freshness effect."
    ),
    "Écart de taille en cm (p1 - p2). Utile pour des styles de jeu où le service prime (ex: gazon).": (
        "Height difference in cm (p1 minus p2). Useful for playing styles where serving is important (e.g. on grass)."
    ),
    "Écart de forme récente: taux de victoire sur les 10 derniers matchs de chaque joueur, avant ce "
    "match (p1 - p2). Ex: p1 a gagné 8/10, p2 a gagné 4/10 -> form_diff = +0.4.": (
        "Recent form difference: each player's win rate across their last 10 matches before this match "
        "(p1 minus p2). For example, p1 won 8/10 and p2 won 4/10, giving form_diff = +0.4."
    ),
    "Écart de face-à-face historique: (victoires de p1 contre p2) - (victoires de p2 contre p1), "
    "avant ce match. 0 si c'est leur première confrontation.": (
        "Historical head-to-head difference: (p1 wins against p2) minus (p2 wins against p1), before this match. "
        "Zero if this is their first meeting."
    ),
    "Indique si p1 (+1) ou p2 (-1) est gaucher et pas l'autre (0 si les deux ont la même main). "
    "Un gaucher peut avoir un léger avantage stylistique face à des droitiers peu habitués.": (
        "Indicates whether p1 (+1) or p2 (-1) is left-handed while the other is not (0 if they have the same "
        "handedness). A left-hander may have a slight stylistic advantage against right-handers who are less used to it."
    ),
    "Probabilité de victoire de p1 implicite dans les cotes du marché (1/cote, normalisée pour "
    "retirer la marge du bookmaker). C'est l'estimation du marché — le modèle doit faire mieux "
    "que ça pour espérer être rentable en pariant contre.": (
        "The probability of p1 winning implied by market odds (1/odds, normalized to remove the bookmaker margin). "
        "This is the market estimate — the model must outperform it to have a chance of profiting by betting against it."
    ),
    "Écart du nombre de matchs déjà joués dans l'historique (p1 - p2), un proxy d'expérience "
    "sur le circuit.": (
        "Difference in the number of matches already played in the historical data (p1 minus p2), a proxy for "
        "tour-level experience."
    ),
    "Inverse de la régularisation. Petit C = modèle plus 'prudent'/simple (moins de "
    "surapprentissage), grand C = modèle plus flexible, colle plus aux données "
    "d'entraînement (risque de surapprentissage).": (
        "Inverse of regularization strength. A small C produces a simpler, more conservative model (less "
        "overfitting); a large C is more flexible and fits the training data more closely (with greater overfitting risk)."
    ),
    "Profondeur maximale d'un arbre de décision. Plus c'est profond, plus le "
    "modèle peut capturer des interactions complexes entre features, mais "
    "risque de surapprendre le bruit des données d'entraînement.": (
        "Maximum depth of a decision tree. Greater depth can capture more complex feature interactions, but also "
        "increases the risk of fitting noise in the training data."
    ),
    "Nombre minimum de matchs requis dans une feuille de l'arbre. Plus c'est grand, plus le "
    "modèle est régularisé (évite de créer des règles basées sur 2-3 matchs isolés).": (
        "Minimum number of matches required in a tree leaf. Larger values regularize the model and prevent rules "
        "from being based on only two or three isolated matches."
    ),
    "Nombre d'arbres combinés dans l'ensemble. Plus il y en a, plus le modèle "
    "est stable, mais plus l'entraînement est long.": (
        "Number of trees combined in the ensemble. More trees make the model more stable, but increase training time."
    ),
    "Vitesse d'apprentissage: à quel point chaque nouvel arbre corrige les "
    "erreurs des précédents. Petit = apprentissage plus lent mais souvent "
    "plus robuste.": (
        "Learning rate: how much each new tree corrects the errors of previous trees. A lower rate learns more "
        "slowly but is often more robust."
    ),
    "Nombre de matchs historiques les plus 'similaires' (en termes de features) utilisés "
    "pour prédire l'issue d'un nouveau match.": (
        "Number of the most 'similar' historical matches (in terms of features) used to predict the outcome of a new match."
    ),
    "Marge minimale exigée entre la probabilité du modèle et la probabilité "
    "implicite du marché avant de placer un pari. Ex: seuil de 0.03 -> on ne "
    "parie que si le modèle estime au moins 3 points de % de plus que le marché.": (
        "Minimum required gap between the model probability and the market-implied probability before placing a bet. "
        "For example, a threshold of 0.03 means betting only when the model estimate is at least three percentage "
        "points higher than the market."
    ),
    "Mise fixe identique à chaque pari (ex: toujours 1 unité), quel que soit l'edge estimé.": (
        "The same fixed stake on every bet (e.g. always one unit), regardless of the estimated edge."
    ),
    "Mise proportionnelle à l'edge et à la cote (critère de Kelly), pour maximiser la "
    "croissance de la bankroll sur le long terme. Plus agressif quand l'edge est grand.": (
        "Stake proportional to the edge and odds (the Kelly criterion), intended to maximize long-term bankroll "
        "growth. More aggressive when the edge is larger."
    ),
    "Bornes optionnelles sur la cote du côté parié, en plus du "
    "seuil d'edge. Exclut les grosses faveurs (une cote proche de 1 amplifie mécaniquement "
    "l'edge pour une petite erreur de calibration du modèle) et/ou les gros outsiders "
    "(variance très élevée pour un edge nominal identique). Laisser vide = pas de borne.": (
        "Optional limits on the odds for the selected side, in addition to the edge threshold. Excludes heavy "
        "favorites (odds near 1 mechanically amplify the edge from a small model-calibration error) and/or long "
        "shots (very high variance for the same nominal edge). Leave blank for no limit."
    ),
    "Fraction du Kelly 'plein' réellement misée (ex: 0.25 = un quart du Kelly théorique). "
    "Réduit le risque/la volatilité, car le Kelly plein est très agressif en pratique.": (
        "Fraction of full Kelly actually staked (e.g. 0.25 is one quarter of theoretical Kelly). Reduces risk and "
        "volatility, since full Kelly is very aggressive in practice."
    ),
    "Capital de départ simulé pour le backtest, utilisé pour calculer les mises en mode Kelly.": (
        "Starting capital for the backtest, used to calculate stakes in Kelly mode."
    ),
    "Retour sur investissement = profit total / total misé. Ex: ROI de 5% -> pour 100 unités misées au "
    "total sur la période, 5 unités de profit net.": (
        "Return on investment = total profit / total amount staked. For example, a 5% ROI means 5 units of net "
        "profit for every 100 units staked over the period."
    ),
    "Gain ou perte net en unités de mise sur toute la période de test.": (
        "Net gain or loss in stake units over the entire test period."
    ),
    "Nombre de matchs sur lesquels la stratégie a effectivement misé (edge suffisant).": (
        "Number of matches on which the strategy actually placed a bet (sufficient edge)."
    ),
    "Proportion de paris gagnés (attention: pas forcément corrélé au ROI, un "
    "modèle peut gagner souvent mais sur de petites cotes et perdre au global).": (
        "Proportion of bets won. Note that this is not necessarily correlated with ROI: a model may win often at "
        "short odds and still lose overall."
    ),
    "Pire perte relative depuis un sommet de bankroll pendant le backtest — indicateur de risque.": (
        "Largest relative loss from a bankroll peak during the backtest — a measure of risk."
    ),
    "Capacité du modèle à bien classer/ordonner les matchs (0.5 = hasard, 1.0 = parfait). Ne dépend pas "
    "d'un seuil de décision, contrairement à l'accuracy.": (
        "The model's ability to rank matches correctly (0.5 = random, 1.0 = perfect). Unlike accuracy, it does "
        "not depend on a decision threshold."
    ),
    "Proportion de matchs où le modèle prédit le bon vainqueur (probabilité >= 50%).": (
        "Proportion of matches for which the model predicts the correct winner (probability >= 50%)."
    ),
    "Pénalise fortement les prédictions confiantes mais fausses. Plus c'est bas, mieux c'est.": (
        "Heavily penalizes confident but incorrect predictions. Lower is better."
    ),
    "Erreur quadratique moyenne entre probabilité prédite et résultat réel (0/1). Plus c'est bas, "
    "mieux le modèle est calibré.": (
        "Mean squared error between the predicted probability and the actual outcome (0/1). Lower values indicate "
        "better model calibration."
    ),
})

_TRANSLATIONS.update({
    "Ouvre/ferme les définitions des paramètres dans le panneau latéral": (
        "Open/close the parameter definitions in the sidebar"
    ),
    "Génération aléatoire de modèles": "Random model generation",
    "Irréversible": "Cannot be undone",
    "Chaque modèle entraîne + backteste séparément — au-delà de quelques centaines, le run peut prendre plusieurs minutes.": (
        "Each model is trained and backtested separately — runs may take several minutes when testing more than a few hundred."
    ),
    "Contrairement aux fenêtres ci-dessus (qui font partie du réglage de la config), cette tranche est mise de côté AVANT tout entraînement : c'est le seul résultat qui n'est pas exposé au biais de comparaisons multiples du bruteforce ni au réglage manuel des hyperparamètres. Si un chiffre ici doit faire foi, c'est celui-là.": (
        "Unlike the windows above (which are part of configuration tuning), this period is held out BEFORE any training. It is the only result not exposed to brute-force multiple-comparison bias or manual hyperparameter tuning. This is the result to rely on."
    ),
    "Note composite pondérée: Log loss 32%, Brier 26%, Calibration 21%, AUC 11%, Nombre de matchs 10%. Ne dépend PAS du ROI/backtest: dans le bruteforce la stratégie de mise est tirée au hasard indépendamment du modèle, donc le ROI mesure la chance du tirage de stratégie autant que la qualité du modèle (cf. onglet Données) — à vérifier séparément, ROI/nombre de paris restent affichés à part. Chaque critère est ramené à une échelle 0-1 par rapport à une référence absolue, pas relative aux autres modèles du lot — donc comparable d'un run à l'autre.": (
        "Weighted composite score: Log loss 32%, Brier 26%, calibration 21%, AUC 11%, and match count 10%. It does NOT depend on ROI/backtesting: brute force samples the betting strategy independently of the model, so ROI reflects the luck of the strategy draw as much as model quality (see the Data tab). Evaluate it separately; ROI and bet count are shown apart. Each criterion is scaled from 0 to 1 against an absolute reference, not relative to other models in the batch, so scores can be compared across runs."
    ),
    "Note composite pour classer des RÈGLES DE SÉLECTION entre elles sur un modèle fixe: ROI 50%, nombre de paris 25% (confiance statistique), drawdown maîtrisé 25%. Sans rapport avec le 🏆 Score de qualité du modèle — ici le modèle ne change pas, seule la règle varie.": (
        "Composite score for comparing SELECTION RULES on a fixed model: ROI 50%, number of bets 25% (statistical confidence), and controlled drawdown 25%. This is separate from the 🏆 model quality score — the model stays fixed here; only the rule changes."
    ),
    "Profit / total misé. Pas comparable entre flat et kelly: kelly mise proportionnellement plus au fil du temps, ce qui dilue son ROI% sans refléter une moindre rentabilité. Voir 'Rendement/capital' ci-dessous.": (
        "Profit / total amount staked. Not comparable between flat and Kelly: Kelly stakes proportionally more over time, which dilutes ROI% without indicating lower profitability. See 'Return on capital' below."
    ),
    "Profit / bankroll de départ — comparable entre flat et kelly.": (
        "Profit / starting bankroll — comparable between flat and Kelly."
    ),
    "Rendement/capital divisé par le drawdown max. 'n/a' si le drawdown est quasi nul.": (
        "Return on capital divided by maximum drawdown. 'n/a' when drawdown is close to zero."
    ),
    "Rendement/capital divisé par le drawdown max. Résume en un seul chiffre la rentabilité RAPPORTÉE au risque de creux traversé pour l'obtenir: plus c'est haut, plus le rendement obtenu compense le risque pris. 'n/a' si le drawdown est quasi nul (rien à rapporter).": (
        "Return on capital divided by maximum drawdown. Summarizes profitability RELATIVE to the drawdown risk in a single number: the higher it is, the more the return compensates for the risk taken. 'n/a' if drawdown is close to zero."
    ),
    "Profit total / bankroll de départ. Contrairement au ROI, comparable directement entre stratégies flat et kelly: c'est le multiple réel de croissance de votre capital initial.": (
        "Total profit / starting bankroll. Unlike ROI, this can be compared directly between flat and Kelly strategies: it is the actual growth multiple of your starting capital."
    ),
    "Pas assez de données sur la période choisie (train ou test vide).": (
        "Not enough data for the selected period (training or test set is empty)."
    ),
    "Pas assez de données sur la période/surface choisies (train ou test vide).": (
        "Not enough data for the selected period/surfaces (training or test set is empty)."
    ),
    "Pas assez de données sur la période de confirmation pour l'évaluer.": (
        "Not enough data in the confirmation period to evaluate it."
    ),
    "Sélectionne au moins une feature.": "Select at least one feature.",
    "Modèle dupliqué — visible dans le tableau ci-dessous.": (
        "Model duplicated — see the table below."
    ),
    "Aucun pari réglé pour l'instant sur cette stratégie (": (
        "No settled bets for this strategy yet ("
    ),
    "en attente). La courbe apparaîtra dès que les matchs concernés auront été joués et la base mise à jour.": (
        "pending). The chart will appear once the relevant matches have been played and the database is updated."
    ),
    "pari(s) réglé(s) à l'instant.": "bet(s) settled just now.",
    "`merge` = même match (les deux lignes seront fusionnées) · `reject` = matchs différents (gardés séparés) · vide = pas encore tranché.": (
        "`merge` = same match (the two rows will be merged) · `reject` = different matches (kept separate) · blank = not decided yet."
    ),
    "cas ambigus à valider — utilise la section « Cas ambigus à valider » ci-dessous, puis relance la consolidation.": (
        "ambiguous cases to review — use the 'Ambiguous cases to review' section below, then run consolidation again."
    ),
    "décision(s) enregistrée(s) dans data/match_review.csv.": (
        "decision(s) saved to data/match_review.csv."
    ),
    "ligne(s) affichée(s) (clique un en-tête de colonne pour trier).": (
        "row(s) displayed (click a column header to sort)."
    ),
    "❌ Rejetés (reject)": "❌ Rejected (reject)",
})

_TRANSLATIONS.update({
    "Ouvre/ferme les définitions des paramètres dans le panneau latéral": (
        "Open/close parameter definitions in the sidebar"
    ),
    "Génération aléatoire de modèles": "Random model generation",
    "Irréversible": "Cannot be undone",
    "Chaque modèle entraîne + backteste séparément — au-delà de quelques centaines, le run peut prendre plusieurs minutes.": (
        "Each model is trained and backtested separately — runs may take several minutes when testing more than a few hundred."
    ),
    "Contrairement aux fenêtres ci-dessus (qui font partie du réglage de la config), cette tranche est mise de côté AVANT tout entraînement : c'est le seul résultat qui n'est pas exposé au biais de comparaisons multiples du bruteforce ni au réglage manuel des hyperparamètres. Si un chiffre ici doit faire foi, c'est celui-là.": (
        "Unlike the windows above (which are part of configuration tuning), this period is held out BEFORE any training. It is the only result not exposed to brute-force multiple-comparison bias or manual hyperparameter tuning. This is the result to rely on."
    ),
    "Note composite pondérée: Log loss 32%, Brier 26%, Calibration 21%, AUC 11%, Nombre de matchs 10%. Ne dépend PAS du ROI/backtest: dans le bruteforce la stratégie de mise est tirée au hasard indépendamment du modèle, donc le ROI mesure la chance du tirage de stratégie autant que la qualité du modèle (cf. onglet Données) — à vérifier séparément, ROI/nombre de paris restent affichés à part. Chaque critère est ramené à une échelle 0-1 par rapport à une référence absolue, pas relative aux autres modèles du lot — donc comparable d'un run à l'autre.": (
        "Weighted composite score: Log loss 32%, Brier 26%, calibration 21%, AUC 11%, and match count 10%. It does NOT depend on ROI/backtesting: brute force samples the betting strategy independently of the model, so ROI reflects the luck of the strategy draw as much as model quality (see the Data tab). Evaluate it separately; ROI and bet count are shown apart. Each criterion is scaled from 0 to 1 against an absolute reference, not relative to other models in the batch, so scores can be compared across runs."
    ),
    "Note composite pour classer des RÈGLES DE SÉLECTION entre elles sur un modèle fixe: ROI 50%, nombre de paris 25% (confiance statistique), drawdown maîtrisé 25%. Sans rapport avec le 🏆 Score de qualité du modèle — ici le modèle ne change pas, seule la règle varie.": (
        "Composite score for comparing SELECTION RULES on a fixed model: ROI 50%, number of bets 25% (statistical confidence), and controlled drawdown 25%. This is separate from the 🏆 model quality score — the model stays fixed; only the rule changes."
    ),
    "Profit / bankroll de départ — comparable entre flat et kelly.": (
        "Profit / starting bankroll — comparable between flat and Kelly."
    ),
    "Pas assez de données sur la période choisie (train ou test vide).": (
        "Not enough data for the selected period (training or test set is empty)."
    ),
    "Pas assez de données sur la période/surface choisies (train ou test vide).": (
        "Not enough data for the selected period/surfaces (training or test set is empty)."
    ),
    "Pas assez de données sur la période de confirmation pour l'évaluer.": (
        "Not enough data in the confirmation period to evaluate it."
    ),
    "Aucun pari réglé pour l'instant sur cette stratégie (": (
        "No settled bets for this strategy yet ("
    ),
    "en attente). La courbe apparaîtra dès que les matchs concernés auront été joués et la base mise à jour.": (
        "pending). The chart will appear once the relevant matches have been played and the database is updated."
    ),
    "pari(s) réglé(s) à l'instant.": "bet(s) settled just now.",
    "`merge` = même match (les deux lignes seront fusionnées) · `reject` = matchs différents (gardés séparés) · vide = pas encore tranché.": (
        "`merge` = same match (the two rows will be merged) · `reject` = different matches (kept separate) · blank = not decided yet."
    ),
    "cas ambigus à valider — utilise la section « Cas ambigus à valider » ci-dessous, puis relance la consolidation.": (
        "ambiguous cases to review — use the section below, then run consolidation again."
    ),
    "décision(s) enregistrée(s) dans data/match_review.csv.": (
        "decision(s) saved to data/match_review.csv."
    ),
    "ligne(s) affichée(s) (clique un en-tête de colonne pour trier).": (
        "row(s) displayed (click a column header to sort)."
    ),
    "Ce modèle a été sauvegardé avant l'ajout du suivi de la source de cotes — indique laquelle a été utilisée à l'entraînement pour pouvoir relancer le backtest.": (
        "This model was saved before odds-source tracking was added — select the source used for training to rerun the backtest."
    ),
    "Ce modèle n'a aucune règle de sélection sauvegardée — crée-en une d'abord dans l'onglet '💾 Modèles sauvegardés' → section '🎯 Optimisation ROI pour ce modèle'.": (
        "This model has no saved selection rules — first create one in the '💾 Saved models' tab → '🎯 ROI optimization for this model' section."
    ),
    "Charge d'abord les prédictions du modèle (bouton ci-dessus).": (
        "Load the model predictions first (button above)."
    ),
    "Charge cette config dans l'onglet 🛠️ Construire un modèle pour l'ajuster et la réentraîner (crée un nouveau modèle, ne modifie pas celui-ci).": (
        "Load this configuration in the 🛠️ Build a model tab to adjust and retrain it (this creates a new model and does not modify the saved one)."
    ),
    "Source de cotes introuvable pour ce modèle (sauvegardé avant l'ajout de son suivi) — impossible de reconstruire le jeu de données pour la validation glissante.": (
        "Odds source not found for this model (saved before source tracking was added) — cannot rebuild the dataset for walk-forward validation."
    ),
    "Aucune stratégie exploitable : soit aucun modèle associé n'est retrouvable sur le disque, soit `data/tennis.db` est indisponible.": (
        "No usable strategy: either no associated model can be found on disk, or `data/tennis.db` is unavailable."
    ),
    "Aucune stratégie sélectionnée ci-dessus (un modèle seul 🤖 ne produit pas de signal) — rien n'est enregistré.": (
        "No strategy selected above (a standalone 🤖 model does not generate signals) — nothing is recorded."
    ),
    "Un pari par match et par stratégie, au premier signal — les consultations suivantes ne réécrivent pas la cote enregistrée.": (
        "One bet per match and strategy, recorded at the first signal — later visits do not overwrite the saved odds."
    ),
    "Impossible de récupérer le calendrier ESPN pour la fenêtre demandée. Dernière erreur :": (
        "Could not retrieve the ESPN schedule for the requested window. Latest error:"
    ),
    "Aucune clé ODDS_API_IO_KEY configurée dans .streamlit/secrets.toml.": (
        "No ODDS_API_IO_KEY is configured in .streamlit/secrets.toml."
    ),
    "Aucune clé ODDS_API_KEY configurée dans .streamlit/secrets.toml.": (
        "No ODDS_API_KEY is configured in .streamlit/secrets.toml."
    ),
    "Requêtes The Odds API restantes ce mois-ci :": "The Odds API requests remaining this month:",
    "Journal vide pour l'instant — il se remplira à chaque consultation de cet onglet.": (
        "The log is empty for now — it will be populated each time you visit this tab."
    ),
    "Collecte démarrée le": "Collection started on",
    "instantané(s) ajouté(s) à l'instant.": "snapshot(s) added just now.",
    "nouveau(x) pari(s) enregistré(s) au journal.": "new bet(s) recorded in the log.",
    "matchs à venir sur la période sélectionnée": "upcoming matches in the selected period",
    "Écart du modèle sur la probabilité de victoire de": "Model difference in win probability for",
    "Calibration parfaite": "Perfect calibration",
    "Calibration (probabilité prédite vs taux de victoire réel)": (
        "Calibration (predicted probability vs actual win rate)"
    ),
    "Probabilité prédite": "Predicted probability",
    "Taux de victoire observé": "Observed win rate",
    "Importance des features": "Feature importance",
    "Coefficients du modèle": "Model coefficients",
    "Progression de la bankroll (paris réglés)": "Bankroll progression (settled bets)",
    "Bankroll de départ": "Starting bankroll",
    "Profit par surface (n = nombre de paris)": "Profit by surface (n = number of bets)",
    "Profit par tranche de cote (n = nombre de paris)": "Profit by odds range (n = number of bets)",
    "Profit (u)": "Profit (units)",
    "Joueur": "Player",
    "Score de qualité": "Quality score",
    "Score ROI": "ROI score",
    "ROI par niveau de tournoi": "ROI by tournament level",
    "Calibration": "Calibration",
    "Règle de référence": "Reference rule",
    "Aucun modèle sauvegardé pour l'instant — entraîne et sauvegarde un modèle dans l'onglet '🛠️ Construire un modèle'.": (
        "No saved models yet — train and save a model in the '🛠️ Build a model' tab."
    ),
    "🔄 Mettre à jour les paris (signaux) pour toutes les stratégies": (
        "🔄 Update bets (signals) for all strategies"
    ),
})

_TRANSLATIONS.update({
    "Le modèle reste figé (mêmes probabilités prédites) — seule la stratégie de mise (seuil d'edge, mode flat/kelly) varie ici. Contrairement au 🎲 Bruteforce (onglet dédié), qui ne compare que la qualité prédictive des modèles entre eux.": (
        "The model stays fixed (the same predicted probabilities) — only the betting strategy (edge threshold, flat/Kelly mode) changes here. Unlike 🎲 Brute force (its dedicated tab), which only compares model predictive quality."
    ),
    "Le modèle ci-dessus reste figé (mêmes probabilités prédites) — seule la stratégie de mise (seuil d'edge, mode flat/kelly) est testée en masse pour trouver le meilleur ROI. Sauvegarde d'abord le modèle (bouton 💾 ci-dessus) si tu veux aussi garder une règle de sélection associée — l'onglet 💾 Modèles sauvegardés le permet. Sélectionner une règle ci-dessous la fait aussi servir de stratégie de mise à la validation glissante en bas de page ; sans sélection, celle-ci retombe sur la stratégie de référence du bruteforce (edge ≥ 3%, mise plate).": (
        "The model above stays fixed (the same predicted probabilities) — only the betting strategy (edge threshold, flat/Kelly mode) is tested at scale to find the best ROI. Save the model first (💾 button above) if you also want to save an associated selection rule; you can do so in the 💾 Saved models tab. Selecting a rule below also uses it for walk-forward validation at the bottom of the page. Without a selection, validation uses the brute-force reference strategy (edge ≥ 3%, flat stake)."
    ),
    "Le modèle ci-dessus reste figé (mêmes probabilités prédites) — seule la stratégie de mise (seuil d'edge, mode flat/kelly, cote min/max) est testée en masse pour trouver le meilleur ROI. Sauvegarde d'abord le modèle (bouton 💾 ci-dessus) si tu veux aussi garder une règle de sélection associée — l'onglet 💾 Modèles sauvegardés le permet. Sélectionner une règle ci-dessous la fait aussi servir de stratégie de mise à la validation glissante en bas de page ; sans sélection, celle-ci retombe sur la stratégie réglée dans les sliders ci-dessus.": (
        "The model above stays fixed (the same predicted probabilities) — only the betting strategy (edge threshold, flat/Kelly mode, minimum/maximum odds) is tested at scale to find the best ROI. Save the model first (💾 button above) if you also want to save an associated selection rule; you can do so in the 💾 Saved models tab. Selecting a rule below also uses it for walk-forward validation at the bottom of the page. Without a selection, validation uses the strategy configured by the sliders above."
    ),
    ". Elle ne change que les colonnes roi_%/n_bets/ROI de confirmation, recalculées immédiatement à chaque changement de règle (sans réentraînement). AUC/Brier/LogLoss et le score qualité, eux, ne dépendent pas de la mise : ils restent identiques quelle que soit la règle ROI sélectionnée.": (
        ". It changes only the confirmation roi_%/n_bets/ROI columns, recalculated immediately whenever the rule changes (without retraining). AUC/Brier/LogLoss and the quality score do not depend on the stake and remain the same for every selected ROI rule."
    ),
    "Profit / total misé (turnover). Mesure la marge moyenne extraite par mise, mais pas comparable entre flat et kelly: kelly mise proportionnellement plus au fil du temps (les mises grossissent avec la bankroll), ce qui dilue son ROI% sans refléter une moindre rentabilité réelle. Pour comparer la rentabilité réelle entre stratégies, voir 'Rendement/capital' et 'Rendement/risque' ci-dessous.": (
        "Profit / total amount staked (turnover). Measures average margin per stake, but is not comparable between flat and Kelly: Kelly stakes proportionally more over time (stakes grow with the bankroll), diluting ROI% without indicating lower profitability. To compare actual profitability across strategies, see 'Return on capital' and 'Return/risk' below."
    ),
    "Règle de sélection sauvegardée — visible dans l'onglet 🎯 Stratégies.": (
        "Selection rule saved — available in the 🎯 Strategies tab."
    ),
    "Réentraîne CETTE configuration (algo/hyperparamètres/features/stratégie ci-dessus) sur plusieurs fenêtres temporelles successives (entraînement extensible, test glissant), puis sur la tranche de confirmation finale définie plus haut (🔒 Réserve de confirmation finale). Un modèle qui n'est bon que sur une seule coupure train/test peut avoir simplement eu de la chance : voir la config tenir d'une fenêtre à l'autre est un signe de stabilité qu'un run unique ne donne pas.": (
        "Retrains THIS configuration (algorithm/hyperparameters/features/strategy above) across successive time windows (expanding training set and rolling test set), then on the final confirmation period defined above (🔒 Final confirmation holdout). A model that performs well on only one train/test split may simply have been lucky; consistent results across windows indicate stability that a single run cannot demonstrate."
    ),
    "Réentraîne ce modèle sur plusieurs fenêtres temporelles glissantes + une réserve de confirmation finale jamais vue pendant sa recherche — pour vérifier que sa performance ne tient pas juste à la coupure train/test d'origine (même logique que dans les onglets 🎲 Bruteforce et 🛠️ Construire un modèle).": (
        "Retrains this model across multiple rolling time windows and a final confirmation period never seen during its search, to check that performance is not specific to the original train/test split (the same approach used in the 🎲 Brute force and 🛠️ Build a model tabs)."
    ),
    "Réentraîne le modèle associé sur plusieurs fenêtres temporelles glissantes + une réserve de confirmation finale jamais vue pendant sa recherche, avec CETTE règle de sélection — pour vérifier que cette stratégie tient au-delà de la coupure train/test d'origine.": (
        "Retrains the associated model across multiple rolling time windows and a final confirmation period never seen during its search, using THIS selection rule, to check whether the strategy holds beyond the original train/test split."
    ),
    "Périmètre réappliqué depuis l'entraînement : **": "Scope reapplied from training: **",
    "** (déduite de l'historique du tournoi). Noms retrouvés dans la base : **": (
        "** (inferred from tournament history). Names found in the database: **"
    ),
    "** (edge insuffisant, probabilité du modèle non calculable, ou aucune cote bookmaker disponible — le signal ne se prend jamais sur Polymarket seul).": (
        "** (insufficient edge, model probability unavailable, or no bookmaker odds available — a signal is never based on Polymarket alone)."
    ),
    "** sur la période : soit aucun match n'atteint le seuil d'edge, soit les cotes bookmaker/probabilités du modèle manquent encore (le marché n'ouvre que quelques jours avant le tournoi).": (
        "** during this period: either no match meets the edge threshold, or bookmaker odds/model probabilities are not available yet (markets open only a few days before the tournament)."
    ),
    "** sur la période — joueur à jouer surligné en vert dans les tableaux ci-dessous.": (
        "** during this period — the player to bet on is highlighted in green in the tables below."
    ),
    ".joblib` est introuvable : aucune prédiction ne peut être calculée. Le dossier `data/` n'étant pas versionné, les modèles entraînés sur une autre machine ou effacés depuis doivent être ré-entraînés (onglet 🛠️ Construire un modèle).": (
        ".joblib` was not found, so predictions cannot be calculated. The `data/` directory is not version-controlled; models trained on another machine or since deleted must be retrained (🛠️ Build a model tab)."
    ),
    "🎯 **Opportunité — miser sur": "🎯 **Opportunity — bet on",
    "**Mise conseillée :**": "**Recommended stake:**",
    "de pari** détectée": " betting opportunity** detected",
    "opportunité": "opportunity",
    "stratégie(s) rejouée(s) sur": "strategy/strategies replayed across",
    "nouveau(x) pari(s) enregistré(s) à l'instant pour **": "new bet(s) just recorded for **",
    "🔄 Mettre à jour les paris (signaux) pour toutes les stratégies": (
        "🔄 Update bets (signals) for all strategies"
    ),
})

_TRANSLATIONS.update({
    "% sur les Grand Chelems/Masters. En production vous ne pourriez miser que sur la seconde moitié. Le chantier prioritaire est alors la couverture des cotes (app/odds.py), pas le modèle.": (
        "% on Grand Slams/Masters. In production, only the second half could be bet on. Improving odds coverage (app/odds.py), not the model, should be the priority."
    ),
    "). Clique sur '🚀 Lancer le bruteforce' pour le mettre à jour.": (
        "). Click '🚀 Run brute force' to update it."
    ),
    "** (réglée dans 🔒 Réserve de confirmation finale ci-dessus, en haut de page — la même réserve que celle qui limite déjà l'exploration, pour garantir qu'elle est réellement jamais vue avant ce test).": (
        "** (set in the 🔒 Final confirmation holdout above — the same holdout that limits exploration, ensuring it remains unseen before this test)."
    ),
    "** ? Cette action est irréversible.": "**? This action cannot be undone.",
    "** ainsi que ses règles de sélection sauvegardées ? Cette action est irréversible.": (
        "** and its saved selection rules? This action cannot be undone."
    ),
    "Cotes « ": "Odds “",
    " » disponibles du **": "” available from **",
    "** au **": "** to **",
    "** sur ce périmètre. Zone réservée à la confirmation finale : **": (
        "** in this scope. Final confirmation holdout: **"
    ),
    " — non explorable ci-dessous, y compris dans le bruteforce et le constructeur manuel.": (
        " — not explorable below, including in brute force and the manual builder."
    ),
    "Configuration chargée depuis le modèle sauvegardé «": (
        "Configuration loaded from saved model “"
    ),
    "Stratégie de mise appliquée ici : **": "Betting strategy applied here: **",
    "Trié par défaut par score_roi décroissant (ROI 50% + nombre de paris 25% + drawdown maîtrisé 25%). Clique sur une ligne pour voir le détail.": (
        "Sorted by descending score_roi by default (ROI 50% + bet count 25% + controlled drawdown 25%). Click a row to see details."
    ),
    "Trié par défaut par score_qualité décroissant (Log loss 32% + Brier 26% + Calibration 21% + AUC 11% + Nombre de matchs 10%, cf. détail dans la vue d'un modèle) — ce bruteforce ne compare QUE la qualité prédictive des modèles, avec une stratégie de mise FIXE et identique pour toutes les lignes (edge ≥ 3%, mise plate) : roi_%/profit servent de référence commune, pas d'optimisation. Une fois un bon modèle identifié et sauvegardé, utilise le 🎯 Bruteforce ROI (onglet 💾 Modèles sauvegardés) pour chercher la meilleure stratégie de mise sur CE modèle précis. Clique sur une ligne pour voir le détail, ou sur un en-tête de colonne pour trier autrement (ex: roi_%).": (
        "Sorted by descending quality_score by default (Log loss 32% + Brier 26% + calibration 21% + AUC 11% + match count 10%; see the model details for the breakdown). This brute force compares ONLY predictive model quality, with the same FIXED betting strategy for every row (edge ≥ 3%, flat stake): roi_%/profit are a common reference, not an optimization. Once a good model is identified and saved, use 🎯 ROI brute force (in the 💾 Saved models tab) to find the best betting strategy for THAT model. Click a row to see details or a column header to sort differently (e.g. roi_%)."
    ),
    "Une fenêtre dont la période d'entraînement est plus courte que ça n'est pas générée. Le compte part de la première date où la source de cotes choisie a des données (et non du premier match de la base, qui remonte à 1968 sans aucune cote) — demander un minimum trop élevé peut donc ne produire aucune fenêtre.": (
        "Windows with shorter training periods are not generated. The count starts from the first date with data for the selected odds source (not the first match in the database, which dates to 1968 and has no odds). A minimum that is too high may therefore produce no windows."
    ),
    "Une stratégie = une association que tu crées toi-même entre un modèle sauvegardé et une de ses règles de sélection sauvegardées (créées via la simulation manuelle ou le bruteforce ROI, onglet 💾 Modèles sauvegardés → 🎯 Optimisation ROI). Cette liste est vide tant que tu n'as rien ajouté ci-dessous.": (
        "A strategy is a link you create between a saved model and one of its saved selection rules (created with manual simulation or ROI brute force in 💾 Saved models → 🎯 ROI optimization). This list remains empty until you add an item below."
    ),
    "demandé(s) (": "requested (",
    "matchs avec cotes sur la période/surfaces sélectionnées — Train:": (
        "matches with odds in the selected period/surfaces — Train:"
    ),
    "modèles demandés), pas encore régénéré pour la valeur actuelle du champ (": (
        "models requested), not yet regenerated for the current field value ("
    ),
    "n/a (métrique indisponible)": "n/a (metric unavailable)",
    "paris — l'incertitude sur ce chiffre dépasse largement son écart à zéro, quel que soit son signe. Allonge la réserve de confirmation avant d'en tirer quoi que ce soit.": (
        "bets — uncertainty in this figure is much larger than its distance from zero, regardless of its sign. Extend the confirmation holdout before drawing any conclusions."
    ),
    "paris. C'est le seul chiffre de cette page qui vaut comme preuve. (Retombée depuis les fenêtres in-sample :": (
        "bets. This is the only figure on this page that counts as evidence. (Drop from the in-sample windows:"
    ),
    "pts de retombée). Le rendement observé pendant la recherche ne survit pas à du jamais-vu — ne pas exploiter cette config.": (
        "points of decline). The returns seen during search did not hold up on unseen data — do not use this configuration."
    ),
    "pts — une retombée modérée est normale, une retombée énorme signale du surapprentissage de la stratégie de mise.)": (
        "points — a moderate decline is normal; a very large decline indicates betting-strategy overfitting.)"
    ),
    "pts) — à lire comme une mesure de STABILITÉ, pas de rendement attendu : la stratégie de mise a été réglée sur ces mêmes matchs, donc ce chiffre ne peut qu'être flatteur. Une forte variabilité d'une fenêtre à l'autre reste, elle, un signal négatif exploitable : même avantagé, le ROI ne tient pas.": (
        "points) — interpret this as a measure of STABILITY, not expected returns: the betting strategy was tuned on these same matches, so this figure is inevitably optimistic. High variability between windows is still a useful negative signal: even with an advantage, the ROI is not consistent."
    ),
    "» sur ce périmètre — change de source de cotes ou élargis les surfaces.": (
        "” in this scope — change the odds source or include more surfaces."
    ),
    "» sur ce périmètre — validation impossible.": "” in this scope — validation is not possible.",
    "» — ajuste si besoin puis relance l'entraînement ci-dessous (ça créera un nouveau modèle, l'original n'est pas modifié).": (
        "” — adjust it if needed, then retrain below (this creates a new model; the original is unchanged)."
    ),
    "⏳ Prédictions pas encore chargées.": "⏳ Predictions not loaded yet.",
    "** est enregistré mais son fichier `data/models/": (
        "** is saved but its file `data/models/"
    ),
    "⚠️ **Ce que les fenêtres ci-dessous ne prouvent pas.** Elles recouvrent la période sur laquelle cette config a été CHOISIE (le bruteforce y a trié ses modèles, l'optimisation ROI y a réglé son seuil d'edge et ses bornes de cote). Réentraîner le modèle par fenêtre ne défait pas ce choix : leur AUC/Brier est optimiste, et leur **ROI est in-sample** puisque la stratégie de mise a été ajustée sur ces mêmes matchs. Elles mesurent la STABILITÉ, pas la généralisation — pour ça, il n'y a que la confirmation finale, en bas de section.": (
        "⚠️ **What the windows below do not prove.** They cover the period used to CHOOSE this configuration (brute force ranked the models there, and ROI optimization tuned its edge threshold and odds limits there). Retraining the model in each window does not undo that selection: AUC/Brier are optimistic, and **ROI is in-sample** because the betting strategy was tuned on the same matches. These windows measure STABILITY, not generalization — only the final confirmation period at the bottom of this section can do that."
    ),
    "⚠️ Le ROI est porté par des niveaux SANS cotes live :": (
        "⚠️ ROI is driven by tournament levels WITHOUT live odds:"
    ),
    "⚠️ Périmètre d'entraînement (surfaces, abandons) non enregistré pour ce modèle — il a été sauvegardé avant le suivi de cette information. La validation ci-dessous porte sur l'historique complet, donc potentiellement sur des surfaces que ce modèle n'a jamais vues : réentraîne-le depuis l'onglet 🛠️ pour obtenir une confirmation sur la bonne population.": (
        "⚠️ The training scope (surfaces, retirements) was not saved for this model — it predates tracking this information. Validation below uses the full history, potentially including surfaces the model has never seen. Retrain it from the 🛠️ tab to confirm performance on the correct population."
    ),
    "⚠️ ROI négatif sur la confirmation finale : **": (
        "⚠️ Negative ROI in the final confirmation period: **"
    ),
    "⚠️ `implied_prob_p1` (probabilité du marché) comme feature d'entraînement: le modèle peut se contenter de reproduire le marché plutôt que de détecter un edge indépendant. Pour identifier de vraies divergences model vs marché, entraîne plutôt sans cette feature et compare après coup (c'est déjà ce que fait le backtest, indépendamment de ce choix). `surface` développe automatiquement 3 colonnes one-hot (Clay/Grass/Hard, Carpet en référence).": (
        "⚠️ Using `implied_prob_p1` (market probability) as a training feature may let the model simply reproduce the market instead of finding an independent edge. To identify genuine differences between the model and market, train without this feature and compare them afterward (the backtest already does this regardless). `surface` automatically expands into three one-hot columns (Clay/Grass/Hard, with Carpet as the reference)."
    ),
    "⚠️ `implied_prob_p1` (probabilité du marché) comme feature: risque que le modèle se contente de reproduire le marché plutôt que de détecter un edge indépendant — décoche-la pour des modèles vraiment indépendants du marché. `surface` développe 3 colonnes one-hot (Clay/Grass/Hard).": (
        "⚠️ Using `implied_prob_p1` (market probability) as a feature risks letting the model reproduce the market instead of finding an independent edge — uncheck it for models that are truly independent of the market. `surface` expands into three one-hot columns (Clay/Grass/Hard)."
    ),
    "✅ Prédictions chargées (": "✅ Predictions loaded (",
    "🧠 Entraîner et backtester": "🧠 Train and backtest",
    "**. **Décision** : un pari est signalé (case du joueur en vert) quand la probabilité du modèle dépasse d'au moins le seuil d'edge celle impliquée par la **cote bookmaker** — seule source contre laquelle la règle a été validée en backtest, Polymarket n'est jamais utilisé pour décider. **Exécution** : la colonne '🛒 Où miser' route ensuite le pari vers le marché offrant la meilleure cote pour ce joueur, souvent Polymarket (pas de marge bookmaker). Miser plus haut sur un pari déjà décidé ne peut qu'améliorer le résultat : le ROI du backtest reste un plancher.": (
        "**. **Decision**: a bet is signaled (the player's cell is highlighted in green) when the model probability exceeds the bookmaker-implied probability by at least the edge threshold. Bookmaker odds are the only source used to validate the rule; Polymarket is never used to make the decision. **Execution**: the '🛒 Where to bet' column routes the bet to the market offering the best odds for that player, often Polymarket (without bookmaker margin). Better odds can only improve a bet already selected, so backtest ROI remains a lower bound."
    ),
    "Rejoue le pipeline de signal pour CHAQUE stratégie enregistrée (pas seulement celle sélectionnée ci-dessus) sur les tournois affichés, et journalise les nouveaux paris — sans avoir à les sélectionner une par une.": (
        "Replays the signal pipeline for EVERY saved strategy (not only the one selected above) across the displayed tournaments and logs new bets — without selecting strategies one by one."
    ),
    "📊 Polymarket (marché prédictif, sans bookmaker) — toujours affiché, couvre tour principal ET Challengers, mais un marché n'existe que si le tirage au sort du tour est déjà sorti. Probabilité implicite (%) + cote décimale équivalente (1/proba).": (
        "📊 Polymarket (prediction market, no bookmaker) — always shown and covers both the main tour and Challengers, but a market is available only after the draw for that round is published. Implied probability (%) and equivalent decimal odds (1/probability)."
    ),
})

_PHRASES = {
    "data/tennis.db introuvable — lance une première mise à jour ci-dessous.": (
        "data/tennis.db was not found — run an initial update below."
    ),
    "data/tennis.db introuvable — lance d'abord une mise à jour ci-dessus.": (
        "data/tennis.db was not found — run an update above first."
    ),
    "data/tennis.db est vide (aucune table) — lance d'abord une mise à jour ci-dessus.": (
        "data/tennis.db is empty (no tables) — run an update above first."
    ),
    "Échec de la mise à jour :": "Update failed:",
    "Échec de la consolidation :": "Consolidation failed:",
    "Consolidation en cours...": "Consolidation in progress...",
    "Mise à jour en cours...": "Update in progress...",
    "Mise à jour terminée": "Update complete",
    "Consolidation terminée": "Consolidation complete",
    "Le dernier match en base date de": "The most recent match in the database was",
    "jours. Une mise à jour est recommandée.": "days ago. An update is recommended.",
    "Dernier commit local :": "Latest local commit:",
    "introuvable ou pas un dépôt git": "not found or not a Git repository",
    "Aucun cas ambigu en attente": "No ambiguous cases pending",
    "Tous les cas ont été tranchés": "All cases have been reviewed",
    "decision(s) enregistrée(s) dans data/match_review.csv.": (
        "decision(s) saved to data/match_review.csv."
    ),
    "Erreur de requête :": "Query error:",
    "ligne(s) affichée(s)": "row(s) displayed",
    "lignes au total dans": "total rows in",
    "cas correspondent aux filtres (sur": "cases match the filters (out of",
    "Aucun fournisseur de cotes bookmaker sélectionné": "No bookmaker odds provider selected",
    "Aucune opportunité détectée par la stratégie": "No opportunity detected by the strategy",
    "Aucun signal de pari sur ce match pour la stratégie": "No betting signal for this match under strategy",
    "Adversaires pas encore connus (tirage au sort pas encore sorti) — pas de stats possibles.": (
        "Opponents are not known yet (the draw has not been published) — statistics are unavailable."
    ),
    "Base de données indisponible — impossible de calculer les stats.": (
        "Database unavailable — cannot calculate statistics."
    ),
    "introuvable dans l'historique de data/tennis.db — pas de stats possibles.": (
        "was not found in data/tennis.db history — statistics are unavailable."
    ),
    "Aucun pari déclenché sur la confirmation finale (seuil d'edge jamais atteint) — rien à conclure, ni dans un sens ni dans l'autre.": (
        "No bets were triggered in the final confirmation period (the edge threshold was never reached) — no conclusion can be drawn either way."
    ),
    "Aucun pari placé avec cette configuration (seuil d'edge jamais atteint).": (
        "No bets were placed with this configuration (the edge threshold was never reached)."
    ),
    "Aucune fenêtre exploitable avec ces paramètres (pas assez de matchs avec cotes).": (
        "No usable windows with these settings (not enough matches with odds)."
    ),
    "Rapprochement des paris avec les résultats connus...": "Matching bets with known results...",
    "Mise à jour des résultats": "Update results",
    "Récupération du calendrier ATP": "Fetching the ATP schedule",
    "n'a aucune règle de sélection sauvegardée": "has no saved selection rules",
    "pour l'instant": "yet",
    "Clique sur une ligne pour voir le détail.": "Click a row to see details.",
    "Trié par défaut par score_roi décroissant": "Sorted by score_roi in descending order by default",
    "Erreur :": "Error:",
    "data/tennis.db suit une hiérarchie de sources : `tennis_atp` (Sackmann) est la colonne vertébrale — seule à définir un match et à porter l'identité joueur ; `TML-Database` enrichit sur clé exacte (`indoor`) et couvre les tournois que Sackmann n'a pas encore publiés ; `tennis-data.co` n'apporte que les cotes. Une source secondaire ne peut jamais créer de ligne, ce qui garantit qu'un match réel n'apparaît qu'une fois.": (
        "data/tennis.db follows a source hierarchy: `tennis_atp` (Sackmann) is the backbone — it alone defines matches and player identities; `TML-Database` enriches exact matches (`indoor`) and covers tournaments not yet published by Sackmann; `tennis-data.co` provides odds only. Secondary sources can never create rows, ensuring that each real match appears only once."
    ),
    "1) `git pull` sur les 2 dépôts sources  ·  2) téléchargement des derniers fichiers tennis-data.co (année en cours + précédente)  ·  3) relance de la consolidation (fusion, dédoublonnage, écriture de tennis.db). Peut prendre plusieurs minutes.": (
        "1) `git pull` the two source repositories · 2) download the latest tennis-data.co files (current and previous year) · 3) rerun consolidation (merge, deduplicate, and write tennis.db). This may take several minutes."
    ),
    "Applique les décisions enregistrées ci-dessus en refusionnant les données déjà présentes sur disque, sans repasser par git pull / téléchargement tennis-data.co.": (
        "Apply the decisions saved above by merging the data already on disk again, without running git pull or downloading tennis-data.co."
    ),
    "Performance des paris **réellement signalés** par une stratégie dans l'onglet '📅 Prochains matchs' : chaque signal y est enregistré comme un pari flat de 1 unité, à la cote du moment, avant que le résultat n'existe. C'est la seule mesure hors échantillon dont on dispose — le ROI de l'onglet 🎯 Stratégies vient, lui, d'un jeu de test que le bruteforce ROI a exploré des centaines de fois avant d'en retenir la meilleure règle, et il est donc optimiste par construction.": (
        "Performance of bets **actually signaled** by a strategy in the '📅 Upcoming matches' tab: each signal is recorded as a flat one-unit bet at the current odds, before the result is known. This is the only out-of-sample measure available — the ROI in the 🎯 Strategies tab comes from a test dataset that ROI brute force explored hundreds of times before selecting the best rule, so it is inherently optimistic."
    ),
    "Aucun pari enregistré pour l'instant. Sélectionne une stratégie (🎯) dans l'onglet '📅 Prochains matchs' : dès qu'elle signale une opportunité sur un match à venir, le pari est journalisé ici automatiquement.": (
        "No bets have been recorded yet. Select a strategy (🎯) in the '📅 Upcoming matches' tab: as soon as it signals an opportunity in an upcoming match, the bet is automatically logged here."
    ),
    "L'exécution est la meilleure cote trouvée pour le pari (souvent Polymarket, sans marge bookmaker) — c'est ce qu'on aurait réellement encaissé. La cote du signal donne une vue prudente, directement comparable au backtest.": (
        "Execution odds are the best odds found for the bet (often Polymarket, with no bookmaker margin) — the amount that would actually have been paid out. Signal odds provide a conservative view that can be compared directly with the backtest."
    ),
    "Le journal est le relevé de performance réelle de la stratégie : effacé, il ne peut pas être reconstitué (les cotes du moment du signal ne sont plus disponibles).": (
        "The log records the strategy's actual performance. Once deleted, it cannot be reconstructed because the odds available when the signal was generated are no longer available."
    ),
    "Paris signalés dont le match n'est pas encore joué, ou dont le résultat n'est pas encore dans data/tennis.db. Exclus de la courbe et des métriques.": (
        "Signaled bets whose match has not been played yet, or whose result is not yet in data/tennis.db. Excluded from the chart and metrics."
    ),
    "Un pari reste en attente tant que son match n'est pas retrouvé dans `data/tennis.db` — pense à mettre la base à jour (onglet 🗄️ Données) avant de recharger les résultats.": (
        "A bet remains pending until its match is found in `data/tennis.db` — remember to update the database (🗄️ Data tab) before reloading results."
    ),
    "Aucun signal de pari sur ce match pour la stratégie **": (
        "No betting signal for this match under strategy **"
    ),
    "Adversaires pas encore connus (tirage au sort pas encore sorti) — pas de stats possibles.": (
        "Opponents are not known yet (the draw has not been published) — statistics are unavailable."
    ),
    "Base de données indisponible — impossible de calculer les stats.": (
        "Database unavailable — cannot calculate statistics."
    ),
    "Aucun fournisseur de cotes bookmaker sélectionné : la décision de pari se prend uniquement sur les cotes bookmaker, aucun signal ne peut donc être calculé. Choisis 'The Odds API' ou 'odds-api.io' dans le menu 'Fournisseur de cotes' ci-dessus.": (
        "No bookmaker odds provider is selected. Betting decisions rely on bookmaker odds, so no signal can be calculated. Select 'The Odds API' or 'odds-api.io' from the 'Odds provider' menu above."
    ),
    "Chaque opportunité signalée ci-dessus est enregistrée comme un pari à mise flat de 1 unité, à la cote disponible au moment du signal, sur un match pas encore joué. Une fois le match dans `data/tennis.db`, le pari est réglé automatiquement et vient alimenter la courbe de bankroll de l'onglet 💰 Bankroll. C'est la seule mesure de performance **hors échantillon** : le ROI du backtest, lui, vient d'un jeu de test déjà exploré des centaines de fois par la recherche de règle.": (
        "Each opportunity signaled above is recorded as a flat one-unit bet at the odds available when the signal was generated, on a match that has not yet been played. Once the match appears in `data/tennis.db`, the bet is settled automatically and contributes to the bankroll chart in the 💰 Bankroll tab. This is the only **out-of-sample** performance measure: the backtest ROI comes from a test dataset that has already been explored hundreds of times during rule search."
    ),
    "Aucun historique Polymarket n'est disponible publiquement, alors que l'historique bookmaker couvre des années (data/tennis.db). C'est ce qui empêche aujourd'hui de valider un signal sur les matchs que **seul** Polymarket couvre : ils sont absents du backtest, qui ne retient que les matchs cotés par un bookmaker. Ce journal enregistre à chaque consultation le prix Polymarket **et** la cote bookmaker du moment, pour reconstituer cet historique. Une fois les résultats connus, les matchs couverts par les deux sources permettent de mesurer directement l'écart entre Polymarket et le consensus bookmaker dévigorisé, et les matchs Polymarket-only deviennent enfin backtestables.": (
        "No public Polymarket history is available, while bookmaker history spans years (data/tennis.db). This currently prevents validating signals on matches covered **only** by Polymarket: they are absent from the backtest, which only includes matches with bookmaker odds. This log records both the Polymarket price **and** bookmaker odds at each visit to build that history. Once results are known, matches covered by both sources make it possible to measure the gap between Polymarket and the de-vigged bookmaker consensus directly, and Polymarket-only matches can finally be backtested."
    ),
    "Le modèle ci-dessus reste figé (mêmes probabilités prédites) — seule la stratégie de mise (seuil d'edge, mode flat/kelly) varie ici. Contrairement au 🎲 Bruteforce (onglet dédié), qui ne compare que la qualité prédictive des modèles entre eux.": (
        "The model above remains fixed (the same predicted probabilities) — only the betting strategy (edge threshold, flat/Kelly mode) changes here. Unlike 🎲 Brute force (its dedicated tab), which only compares the models' predictive quality."
    ),
    "Les X derniers mois COUVERTS PAR LA SOURCE DE COTES choisie sont mis de côté ci-dessous: ni le bruteforce ni le constructeur manuel ne peuvent les voir (impossible de régler une date au-delà). Seule la validation glissante (section 🔁 ci-dessous) les évalue, en confirmation finale — ça garantit qu'aucun réglage manuel ni aucun tri de résultat bruteforce n'a pu être influencé par cette période avant qu'elle serve de test.": (
        "The last X months COVERED BY THE SELECTED ODDS SOURCE are held out below: neither brute force nor the manual builder can access them (dates cannot be set beyond this point). Only walk-forward validation (the 🔁 section below) evaluates this period as final confirmation — ensuring that neither manual tuning nor brute-force result sorting could be influenced by it before it is used as a test."
    ),
    "Un abandon désigne bien un vainqueur et les paris sont généralement réglés, ces matchs sont donc gardés par défaut. Mais leur issue dépend surtout de l'état physique du joueur, que le modèle ne voit pas: les exclure retire ce bruit d'étiquette, au prix d'environ 3 % des lignes. (Les forfaits, eux, sont toujours écartés: aucun match n'a été joué.)": (
        "A retirement still determines a winner and bets are generally settled, so these matches are included by default. However, the outcome depends largely on the player's physical condition, which the model cannot observe. Excluding them removes this label noise at the cost of about 3% of rows. Walkovers are always excluded because no match was played."
    ),
    "Pas assez de données sur la période choisie (train ou test vide).": (
        "Not enough data for the selected period (training or test set is empty)."
    ),
    "Pas assez de données sur la période/surface choisies (train ou test vide).": (
        "Not enough data for the selected period/surfaces (training or test set is empty)."
    ),
    "Pas assez de données sur la période de confirmation pour l'évaluer.": (
        "Not enough data in the confirmation period to evaluate it."
    ),
    "Sélectionne au moins une feature.": "Select at least one feature.",
    "Stratégie créée.": "Strategy created.",
    "Modèle dupliqué — visible dans le tableau ci-dessous.": (
        "Model duplicated — see the table below."
    ),
    "Aucun modèle sauvegardé pour l'instant — entraîne et sauvegarde un modèle dans l'onglet '🛠️ Construire un modèle'.": (
        "No saved models yet — train and save a model in the '🛠️ Build a model' tab."
    ),
}


_WORDS = {
    "ajouter": "add", "ajustement": "adjustment", "année": "year", "années": "years",
    "abandonné": "retired", "abandonnés": "retired", "abandon": "retirement",
    "abord": "first", "affichée": "displayed", "affichées": "displayed",
    "affichés": "displayed", "afficher": "display", "algorithme": "algorithm",
    "algorithmes": "algorithms", "appliqué": "applied", "appliquer": "apply",
    "après": "after", "aujourd'hui": "today", "aucun": "no", "aucune": "no",
    "avec": "with", "base": "database", "biais": "bias", "borne": "limit",
    "bookmaker": "bookmaker", "calcul": "calculation", "calculer": "calculate",
    "cas": "cases", "cette": "this", "ceux": "those", "choisie": "selected",
    "choisi": "selected", "choix": "selection", "clique": "click",
    "colonnes": "columns", "colonne": "column", "comparaison": "comparison",
    "configuration": "configuration", "connu": "known", "connus": "known",
    "confirmation": "confirmation", "construction": "building", "cote": "odds",
    "cotes": "odds", "côté": "side", "courbe": "curve", "créée": "created",
    "créer": "create", "date": "date", "données": "data", "donnée": "data",
    "début": "start", "décision": "decision", "décisions": "decisions",
    "définitivement": "permanently", "détail": "details", "détails": "details",
    "disponible": "available", "disponibles": "available", "écart": "difference",
    "écarts": "differences", "échec": "failure", "en": "in", "entraînement": "training",
    "erreur": "error", "exclure": "exclude", "exploitable": "usable",
    "exploiter": "use", "exporter": "export", "features": "features",
    "fenêtre": "window", "fenêtres": "windows", "fin": "end", "fusion": "merge",
    "fusionnés": "merged", "gagnés": "won", "gestion": "management",
    "glissante": "walk-forward", "glissantes": "walk-forward", "historique": "history",
    "hyperparamètres": "hyperparameters", "impossible": "unable", "inclus": "included",
    "inclure": "include", "indisponible": "unavailable", "instantané": "snapshot",
    "instantanés": "snapshots", "jeu": "dataset", "joueur": "player",
    "joueurs": "players", "jours": "days", "lancer": "run", "ligne": "row",
    "lignes": "rows", "match": "match", "matchs": "matches", "maximum": "maximum",
    "minimum": "minimum", "mises": "stakes", "mise": "stake", "modèle": "model",
    "modèles": "models", "modifier": "edit", "mois": "months", "moyen": "mean",
    "moyenne": "average", "moyennes": "averages", "moyens": "mean",
    "nombre": "number", "nouveau": "new", "nouvelle": "new", "obtenir": "get",
    "onglet": "tab", "onglets": "tabs", "optionnelle": "optional",
    "optimisation": "optimization", "optimiser": "optimize", "par": "by",
    "pari": "bet", "paris": "bets", "partie": "part", "période": "period",
    "périodes": "periods", "plus": "more", "point": "point", "points": "points",
    "pour": "for", "prédiction": "prediction", "prédictions": "predictions",
    "profit": "profit", "prochains": "upcoming", "prochaine": "next",
    "propre": "custom", "qualité": "quality", "quand": "when", "recherche": "search",
    "recharger": "reload", "règle": "rule", "règles": "rules", "résultat": "result",
    "résultats": "results", "retour": "return", "réserve": "holdout",
    "réussite": "win", "sauvegardé": "saved", "sauvegardée": "saved",
    "sauvegardées": "saved", "sauvegardés": "saved", "sauvegarder": "save",
    "sélection": "selection", "sélectionné": "selected", "sélectionnée": "selected",
    "sélectionnés": "selected", "simulation": "simulation", "source": "source",
    "sources": "sources", "statistiques": "statistics", "stratégie": "strategy",
    "stratégies": "strategies", "surface": "surface", "surfaces": "surfaces",
    "tableau": "table", "test": "test", "tous": "all", "tournoi": "tournament",
    "tournois": "tournaments", "tranche": "period", "trier": "sort", "utilise": "use",
    "validation": "validation", "vérifier": "check", "vide": "empty", "vue": "view",
}

_TEXT_ARGUMENTS = {
    "markdown": {"body"},
    "caption": {"body"},
    "title": {"body"},
    "header": {"body"},
    "subheader": {"body"},
    "text": {"body"},
    "info": {"body"},
    "warning": {"body"},
    "success": {"body"},
    "error": {"body"},
    "exception": {"exception"},
    "metric": {"label", "help"},
    "button": {"label", "help"},
    "download_button": {"label", "help"},
    "selectbox": {"label", "help", "placeholder"},
    "multiselect": {"label", "help", "placeholder"},
    "radio": {"label", "help"},
    "checkbox": {"label", "help"},
    "toggle": {"label", "help"},
    "text_input": {"label", "help", "placeholder"},
    "text_area": {"label", "help", "placeholder"},
    "number_input": {"label", "help"},
    "slider": {"label", "help"},
    "date_input": {"label", "help"},
    "time_input": {"label", "help"},
    "file_uploader": {"label", "help"},
    "select_slider": {"label", "help"},
    "expander": {"label"},
    "tabs": {"tabs"},
    "spinner": {"text"},
    "status": {"label"},
    "toast": {"body"},
    "popover": {"label"},
    "page_link": {"label", "help"},
    "dialog": {"title"},
}

_WIDGET_OPTIONS = {"selectbox", "multiselect", "radio", "select_slider"}
_PATCHED_MARKER = "__ml_app_i18n_wrapped__"
_CURRENT_LANGUAGE = "en"


def _load_saved_language() -> str | None:
    """Return the saved language, or None when no preference exists yet."""
    try:
        preferences = json.loads(_LANGUAGE_SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

    language = preferences.get("language") if isinstance(preferences, dict) else None
    if language not in {"en", "fr"}:
        raise ValueError(f"Invalid saved application language in {_LANGUAGE_SETTINGS_PATH}.")
    return language


def _save_language(language: str) -> None:
    """Persist the selected language in the local, git-ignored data directory."""
    if language not in {"en", "fr"}:
        raise ValueError(f"Unsupported application language: {language!r}")

    _LANGUAGE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LANGUAGE_SETTINGS_PATH.write_text(
        json.dumps({"language": language}, indent=2) + "\n",
        encoding="utf-8",
    )


def _language() -> str:
    return _CURRENT_LANGUAGE


def translate(value: Any, language: str | None = None) -> Any:
    """Translate visible text while leaving non-text values untouched."""
    if isinstance(value, str):
        if (language or _language()) != "en":
            return value
        if value in _TRANSLATIONS:
            return _TRANSLATIONS[value]

        translated = value
        for source, target in sorted(_PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
            translated = translated.replace(source, target)
        for source, target in sorted(_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
            translated = translated.replace(source, target)

        import re

        word_pattern = re.compile(
            r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?"
        )

        def replace_word(match: re.Match[str]) -> str:
            word = match.group(0)
            replacement = _WORDS.get(word.casefold())
            if replacement is None:
                return word
            if word.isupper():
                return replacement.upper()
            if word[:1].isupper():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        return word_pattern.sub(replace_word, translated)
    if isinstance(value, list):
        return [translate(item, language) for item in value]
    if isinstance(value, tuple):
        return tuple(translate(item, language) for item in value)
    return value


def _translate_dataframe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        translated = value.rename(columns=lambda column: translate(str(column))).copy()
        for column in translated.select_dtypes(include=["object", "string"]).columns:
            translated[column] = translated[column].map(
                lambda item: translate(item) if isinstance(item, str) else item
            )
        return translated
    return value


def _translate_figure(value: Any) -> Any:
    try:
        from copy import deepcopy
        import plotly.graph_objects as go

        if not isinstance(value, go.Figure):
            return value
        figure = deepcopy(value)
        layout = figure.layout
        if layout.title and layout.title.text:
            figure.update_layout(title=translate(layout.title.text))
        if layout.xaxis and layout.xaxis.title and layout.xaxis.title.text:
            figure.update_xaxes(title_text=translate(layout.xaxis.title.text))
        if layout.yaxis and layout.yaxis.title and layout.yaxis.title.text:
            figure.update_yaxes(title_text=translate(layout.yaxis.title.text))
        for annotation in layout.annotations or ():
            if annotation.text:
                annotation.text = translate(annotation.text)
        for trace in figure.data:
            if isinstance(trace.name, str):
                trace.name = translate(trace.name)
        return figure
    except ImportError:
        return value


def _translate_bound_arguments(bound: inspect.BoundArguments, name: str) -> None:
    if name == "write":
        for key, value in tuple(bound.arguments.items()):
            if key != "self":
                bound.arguments[key] = translate(value)
    else:
        for key in _TEXT_ARGUMENTS.get(name, ()):
            if key in bound.arguments:
                bound.arguments[key] = translate(bound.arguments[key])

    if name in _WIDGET_OPTIONS and "options" in bound.arguments:
        formatter = bound.arguments.get("format_func")

        def translated_option(option: Any) -> str:
            displayed = formatter(option) if formatter is not None else option
            return str(translate(displayed))

        bound.arguments["format_func"] = translated_option

    if name == "dataframe" and "data" in bound.arguments:
        bound.arguments["data"] = _translate_dataframe(bound.arguments["data"])
    if name == "data_editor" and isinstance(bound.arguments.get("column_config"), dict):
        for config in bound.arguments["column_config"].values():
            if isinstance(config, dict):
                for key in ("label", "help"):
                    if isinstance(config.get(key), str):
                        config[key] = translate(config[key])
    if name == "plotly_chart" and "figure_or_data" in bound.arguments:
        bound.arguments["figure_or_data"] = _translate_figure(
            bound.arguments["figure_or_data"]
        )


def _wrap_method(original: Any, name: str):
    signature = inspect.signature(original)

    @wraps(original)
    def translated(self: Any, *args: Any, **kwargs: Any):
        bound = signature.bind_partial(self, *args, **kwargs)
        _translate_bound_arguments(bound, name)
        return original(*bound.args, **bound.kwargs)

    setattr(translated, _PATCHED_MARKER, True)
    return translated


def _wrap_function(original: Any, name: str):
    signature = inspect.signature(original)

    @wraps(original)
    def translated(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        _translate_bound_arguments(bound, name)
        return original(*bound.args, **bound.kwargs)

    setattr(translated, _PATCHED_MARKER, True)
    return translated


def install_translations() -> None:
    """Translate Streamlit component text without changing widget values."""
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator

    method_names = set(_TEXT_ARGUMENTS) | {"write", "dataframe", "data_editor", "plotly_chart"}
    for name in method_names:
        method = getattr(DeltaGenerator, name, None)
        if method is not None and not getattr(method, _PATCHED_MARKER, False):
            setattr(DeltaGenerator, name, _wrap_method(method, name))
        public_function = getattr(st, name, None)
        if public_function is not None and not getattr(public_function, _PATCHED_MARKER, False):
            setattr(st, name, _wrap_function(public_function, name))

    original_dialog = st.dialog
    if not getattr(original_dialog, _PATCHED_MARKER, False):
        signature = inspect.signature(original_dialog)

        @wraps(original_dialog)
        def translated_dialog(*args: Any, **kwargs: Any):
            bound = signature.bind_partial(*args, **kwargs)
            if "title" in bound.arguments:
                bound.arguments["title"] = translate(bound.arguments["title"])
            return original_dialog(*bound.args, **bound.kwargs)

        setattr(translated_dialog, _PATCHED_MARKER, True)
        st.dialog = translated_dialog


def render_language_selector(st: Any) -> None:
    """Render the persistent language selector at the top of the page."""
    global _CURRENT_LANGUAGE

    if "app_language" not in st.session_state:
        saved_language = _load_saved_language()
        st.session_state["app_language"] = saved_language or "en"
        st.session_state["_app_saved_language"] = saved_language
    elif "_app_saved_language" not in st.session_state:
        st.session_state["_app_saved_language"] = _load_saved_language()
    if st.session_state["app_language"] not in {"fr", "en"}:
        st.session_state["app_language"] = "en"
    _, language_column = st.columns([6, 1])
    with language_column:
        st.selectbox(
            "Langue",
            options=("fr", "en"),
            format_func=lambda language: {"fr": "Français", "en": "English"}[language],
            key="app_language",
            label_visibility="collapsed",
        )
    selected_language = st.session_state["app_language"]
    if selected_language != st.session_state["_app_saved_language"]:
        _save_language(selected_language)
        st.session_state["_app_saved_language"] = selected_language
    _CURRENT_LANGUAGE = selected_language
