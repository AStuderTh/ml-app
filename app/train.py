"""Entraîne un modèle sur une période, backteste sur la période suivante,
et rassemble tout dans un seul résultat exploitable par l'UI/le stockage.

Calibration: l'estimateur brut sklearn n'est pas garanti bien calibré (les
arbres/forêts/KNN en particulier — leur predict_proba est une fréquence
empirique, pas une probabilité optimisée). On calibre donc systématiquement
via CalibratedClassifierCV, sur une tranche de fin de la période
d'entraînement jamais vue par le fit principal (train_frame est déjà trié
chronologiquement en amont — cf. app/features.py — donc prendre la queue du
tableau revient à réserver les matchs les plus récents pour la calibration,
sans mélanger le futur dans le passé)."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .backtest import run_backtest
from .modeling import build_estimator

CALIBRATION_FRACTION = 0.15
MIN_ROWS_FOR_CALIBRATION = 150  # en dessous, la calibration ferait plus de mal que de bien
MIN_CALIB_ROWS = 40
ISOTONIC_MIN_CALIB_ROWS = 1000  # isotonic a besoin de volume, sigmoid (Platt) sinon


def _fit_calibrated(algo, params, X_train, y_train):
    """Entraîne l'estimateur de base sur l'essentiel des données, puis
    calibre sur la tranche la plus récente réservée à cet effet. Retombe sur
    un fit classique (non calibré) si le volume est trop faible pour calibrer
    sans bruit."""
    n = len(X_train)
    if n < MIN_ROWS_FOR_CALIBRATION:
        est = build_estimator(algo, params)
        est.fit(X_train, y_train)
        return est, False

    n_calib = max(MIN_CALIB_ROWS, int(n * CALIBRATION_FRACTION))
    n_calib = min(n_calib, n - MIN_CALIB_ROWS)  # garde assez de lignes pour le fit principal
    split = n - n_calib

    base = build_estimator(algo, params)
    base.fit(X_train[:split], y_train[:split])

    y_calib = y_train[split:]
    if len(np.unique(y_calib)) < 2:  # calibration impossible si une seule classe présente
        return base, False

    method = "isotonic" if n_calib >= ISOTONIC_MIN_CALIB_ROWS else "sigmoid"
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    calibrated.fit(X_train[split:], y_calib)
    return calibrated, True


def train_and_evaluate(train_frame, test_frame, algo, params, features, strategy):
    # dropna seulement sur les features + cotes réellement utilisées par CE
    # modèle: une feature absente pour certaines lignes (âge, taille...) ne
    # doit pas faire perdre ces lignes aux modèles qui ne l'utilisent pas.
    needed = list(features) + ["label", "odds_p1", "odds_p2"]
    train_frame = train_frame.dropna(subset=needed)
    test_frame = test_frame.dropna(subset=needed)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Plus aucune ligne après filtrage des valeurs manquantes pour ces features.")

    X_train = train_frame[features].values
    y_train = train_frame["label"].values
    X_test = test_frame[features].values
    y_test = test_frame["label"].values

    est, calibrated = _fit_calibrated(algo, params, X_train, y_train)
    probs_test = est.predict_proba(X_test)[:, 1]
    probs_train = est.predict_proba(X_train)[:, 1]

    def safe_metrics(y, p):
        if len(np.unique(y)) < 2:
            return dict(auc=float("nan"), accuracy=float("nan"), logloss=float("nan"), brier=float("nan"))
        return dict(
            auc=float(roc_auc_score(y, p)),
            accuracy=float(accuracy_score(y, p >= 0.5)),
            logloss=float(log_loss(y, p, labels=[0, 1])),
            brier=float(brier_score_loss(y, p)),
        )

    metrics = safe_metrics(y_test, probs_test)
    metrics["n_train"] = len(train_frame)
    metrics["n_test"] = len(test_frame)
    metrics["calibrated"] = calibrated

    bets_df, bt_metrics = run_backtest(probs_test, test_frame, strategy)

    return {
        "model": est,
        "algo": algo,
        "params": params,
        "features": features,
        "strategy": strategy,
        "metrics": metrics,
        "backtest_metrics": bt_metrics,
        "bets_df": bets_df,
        "test_frame": test_frame.assign(model_prob_p1=probs_test),
        "train_probs": probs_train,
    }
