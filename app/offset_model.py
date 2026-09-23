"""Régression logistique à offset: le marché est le point de départ, le modèle
n'apprend que la correction.

    logit(P_final) = logit(P_marché) + b0 + X·beta

Différence avec `implied_prob_p1` utilisée comme feature ordinaire: là, le
modèle doit RÉAPPRENDRE toute la probabilité, et il consacre l'essentiel de sa
capacité à reconstruire ce que la cote contenait déjà. Ici le coefficient sur
logit(P_marché) est FIXÉ à 1, donc chaque paramètre estimé ne sert qu'à
mesurer un écart au marché.

Ce que ça change en pratique:
  - la question « mes features apportent-elles quelque chose ? » devient
    directement lisible dans les coefficients: tous nuls = rien à ajouter;
  - le modèle ne peut plus paraître bon en recopiant la cote, puisqu'il part
    déjà de la cote — l'AUC affichée cesse d'être flattée par cet effet;
  - la correction est bornée par la régularisation, donc un edge s'exprime en
    points de log-odds plutôt qu'en probabilité reconstruite de zéro.

C'est la formulation GLM standard (`offset=` dans R). On l'implémente à la main
parce que LogisticRegression de sklearn n'expose pas d'offset, et que les
implémentations qui le permettent nativement (base_margin XGBoost, init_score
LightGBM) ne sont pas installées.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    # Forme par morceaux: exp() sur des logits très négatifs déborde en float64
    # et renvoie inf, ce qui propage des NaN dans le gradient et fait échouer
    # l'optimisation sans message clair.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class OffsetLogisticRegression(BaseEstimator, ClassifierMixin):
    """Correction linéaire apprise au-dessus d'une probabilité de référence.

    `offset_index` désigne la colonne de X qui porte la probabilité du marché
    (elle est retirée des features: elle sert d'offset, pas de variable). La
    passer par l'index plutôt que par son nom permet de rester compatible avec
    le pipeline existant, qui travaille sur des tableaux numpy nus.
    """

    def __init__(self, offset_index: int = 0, C: float = 1.0, fit_intercept: bool = True):
        self.offset_index = offset_index
        self.C = C
        self.fit_intercept = fit_intercept

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        offset = logit(X[:, self.offset_index])
        Z = np.delete(X, self.offset_index, axis=1)
        return offset, Z

    def fit(self, X, y):
        offset, Z = self._split(X)
        y = np.asarray(y, dtype=float)

        # Standardisation interne: sans elle, des features d'échelles très
        # différentes (elo_diff ~ centaines, h2h_diff ~ unités) rendent le
        # problème mal conditionné, et la pénalité L2 frapperait arbitrairement
        # plus fort les unes que les autres.
        self.mean_ = Z.mean(axis=0)
        self.scale_ = Z.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Zs = (Z - self.mean_) / self.scale_

        n, d = Zs.shape
        l2 = 1.0 / max(self.C, 1e-12)

        def objective(theta):
            beta = theta[:d]
            b0 = theta[d] if self.fit_intercept else 0.0
            z = offset + Zs @ beta + b0
            p = _sigmoid(z)
            pc = np.clip(p, EPS, 1 - EPS)
            nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
            loss = nll + 0.5 * l2 * np.dot(beta, beta) / n
            resid = (p - y) / n
            grad = np.empty_like(theta)
            grad[:d] = Zs.T @ resid + l2 * beta / n
            if self.fit_intercept:
                grad[d] = resid.sum()
            return loss, grad

        theta0 = np.zeros(d + 1)
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.coef_ = res.x[:d]
        self.intercept_ = float(res.x[d]) if self.fit_intercept else 0.0
        self.classes_ = np.array([0, 1])
        self.converged_ = bool(res.success)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        offset, Z = self._split(X)
        Zs = (Z - self.mean_) / self.scale_
        return offset + Zs @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def correction_report(self, feature_names):
        """Coefficients en log-odds par écart-type de feature.

        Directement interprétable: 'à cote de marché égale, +1 écart-type de
        elo_diff déplace le log-odds de +0.03'. Une liste de coefficients tous
        proches de zéro est la réponse — négative mais nette — à la question
        de savoir si ces features ajoutent quoi que ce soit à la cote.
        """
        names = [n for i, n in enumerate(feature_names) if i != self.offset_index]
        order = np.argsort(-np.abs(self.coef_))
        return [(names[i], float(self.coef_[i])) for i in order]
