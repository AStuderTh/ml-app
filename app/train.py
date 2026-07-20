"""Entraîne un modèle sur une période, backteste sur la période suivante,
et rassemble tout dans un seul résultat exploitable par l'UI/le stockage."""
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .backtest import run_backtest
from .modeling import build_estimator


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

    est = build_estimator(algo, params)
    est.fit(X_train, y_train)
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
