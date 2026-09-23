"""Le modèle offset apporte-t-il quelque chose que la cote ne contient pas ?

Compare trois prédicteurs sur le MÊME jeu de test, en validation chronologique:

  1. le marché seul (probabilité implicite dévigorisée) — la référence;
  2. la cote comme feature ordinaire — ce que font vos modèles enregistrés;
  3. le modèle offset — logit(P_marché) fixé, seule la correction est apprise.

La comparaison décisive est (3) contre (1), en log-loss APPARIÉE match par
match: modèle et marché prédisent les mêmes rencontres, donc la difficulté
commune s'annule dans la différence et l'incertitude devient assez petite pour
trancher sur ~12 000 matchs.

Usage:
    py -3.13 scripts/offset_test.py
    py -3.13 scripts/offset_test.py --test-start 2023-01-01 --odds b365
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.data import ODDS_CHOICES, load_matches  # noqa: E402
from app.features import (build_training_frame, compute_chronological_features,  # noqa: E402
                          expand_features)
from app.offset_model import OffsetLogisticRegression  # noqa: E402

# implied_prob_p1 EN PREMIER: le modèle offset repère la colonne d'offset par
# son index, fixé à 0 ci-dessous.
FEATURES = [
    "implied_prob_p1",
    "elo_diff", "elo_surface_diff", "elo_diff_recent", "elo_surface_diff_recent",
    "age_diff", "form_diff", "experience_diff", "h2h_diff", "rest_days_diff",
    "rank_diff", "surface", "best_of_5",
]

ODDS_ALIASES = {"avg": "Moyenne du marché (Avg)", "b365": "Bet365",
                "ps": "Pinnacle", "max": "Meilleure cote (Max)"}


def _rowwise_ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(y, p_ref, p_test, label):
    """Écart apparié au marché, avec son incertitude."""
    d = _rowwise_ll(y, p_ref) - _rowwise_ll(y, p_test)
    gain, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    if gain > 2 * se:
        verdict = "BAT le marché"
    elif gain < -2 * se:
        verdict = "PERD contre le marché"
    else:
        verdict = "égalité (dans le bruit)"
    print(f"  {label:<34} LL={log_loss(y, np.clip(p_test, 1e-6, 1-1e-6), labels=[0,1]):.4f}  "
          f"AUC={roc_auc_score(y, p_test):.4f}  gain={gain:+.4f}±{2*se:.4f}  {verdict}")
    return gain, se


def walk_forward(full, feats, C, first_test_year, last_year):
    """Réestime la correction année par année, sur historique croissant.

    Un coefficient obtenu sur une seule période est une hypothèse: avec 12
    features et 28 000 lignes, une régression trouve TOUJOURS quelque chose.
    Seule sa reproduction d'une période à l'autre distingue un signal d'un
    artefact d'échantillon — et un coefficient qui change de SIGNE en dit plus
    long qu'une p-value.
    """
    names = [n for i, n in enumerate(feats) if i != 0]
    per_year, gains = {}, []

    for year in range(first_test_year, last_year + 1):
        tr = full[full["tourney_date"] < pd.Timestamp(f"{year}-01-01")]
        te = full[(full["tourney_date"] >= pd.Timestamp(f"{year}-01-01"))
                  & (full["tourney_date"] < pd.Timestamp(f"{year + 1}-01-01"))]
        if len(te) < 500 or len(tr) < 2000:
            continue
        m = OffsetLogisticRegression(offset_index=0, C=C).fit(tr[feats].values, tr["label"].values)
        y = te["label"].values
        d = _rowwise_ll(y, te["implied_prob_p1"].values) - _rowwise_ll(y, m.predict_proba(te[feats].values)[:, 1])
        gains.append({"annee": year, "n": len(te), "gain": d.mean(),
                      "se": d.std(ddof=1) / np.sqrt(len(d))})
        per_year[year] = m.coef_

    if not per_year:
        print("  Pas assez de données pour un walk-forward.")
        return

    print(f"\n{'=' * 92}\n  STABILITÉ DE LA CORRECTION (réestimée chaque année)\n{'=' * 92}")
    years = sorted(per_year)
    print(f"  {'feature':<32}" + "".join(f"{y:>8}" for y in years) + f"{'signe':>10}")
    print(f"  {'-' * (32 + 8 * len(years) + 10)}")
    coefs = np.array([per_year[y] for y in years])
    order = np.argsort(-np.abs(coefs).mean(axis=0))
    for i in order[:10]:
        serie = coefs[:, i]
        stable = "stable" if np.all(np.sign(serie) == np.sign(serie[0])) else "S'INVERSE"
        print(f"  {names[i]:<32}" + "".join(f"{v:>+8.3f}" for v in serie) + f"{stable:>10}")

    print(f"\n  {'-' * 74}\n  Gain annuel vs marché")
    for g in gains:
        mark = "+" if g["gain"] > 2 * g["se"] else ("-" if g["gain"] < -2 * g["se"] else "=")
        print(f"    {g['annee']}  n={g['n']:>5}  gain={g['gain']:+.4f}±{2*g['se']:.4f}  [{mark}]")
    tot_n = sum(g["n"] for g in gains)
    tot = sum(g["gain"] * g["n"] for g in gains) / tot_n
    print(f"    {'TOTAL':<5} n={tot_n:>5}  gain={tot:+.4f}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--odds", default="avg", choices=sorted(ODDS_ALIASES))
    ap.add_argument("--C", type=float, default=1.0, help="inverse de la régularisation L2")
    ap.add_argument("--walk-forward", action="store_true",
                    help="réestime la correction chaque année et teste sa stabilité")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    odds_w, odds_l = ODDS_CHOICES[ODDS_ALIASES[args.odds]]
    print("Chargement + features chronologiques... [~1 min]")
    full = build_training_frame(compute_chronological_features(load_matches()), odds_w, odds_l)

    feats = expand_features(FEATURES)
    full = full.dropna(subset=feats + ["label"])
    split = pd.Timestamp(args.test_start)
    train, test = full[full["tourney_date"] < split], full[full["tourney_date"] >= split]
    print(f"  train={len(train)}  test={len(test)}  features={len(feats)}")

    Xtr, ytr = train[feats].values, train["label"].values
    Xte, yte = test[feats].values, test["label"].values
    p_market = test["implied_prob_p1"].values

    print(f"\n{'=' * 92}\n  COMPARAISON AU MARCHÉ (test >= {args.test_start}, n={len(test)})\n{'=' * 92}")
    print(f"  {'référence':<34} LL={log_loss(yte, np.clip(p_market, 1e-6, 1-1e-6), labels=[0,1]):.4f}  "
          f"AUC={roc_auc_score(yte, p_market):.4f}   (marché seul)")

    # (2) la cote comme feature ordinaire, standardisée comme dans l'app.
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=args.C).fit(sc.transform(Xtr), ytr)
    compare(yte, p_market, lr.predict_proba(sc.transform(Xte))[:, 1], "cote comme feature ordinaire")

    # (3) le modèle offset.
    off = OffsetLogisticRegression(offset_index=0, C=args.C).fit(Xtr, ytr)
    compare(yte, p_market, off.predict_proba(Xte)[:, 1], "modèle OFFSET")
    if not off.converged_:
        print("  (!) L-BFGS n'a pas convergé — résultat à prendre avec prudence.")

    print(f"\n{'-' * 92}")
    print("  Correction apprise (log-odds par écart-type de feature, à cote égale)")
    print(f"  intercept: {off.intercept_:+.4f}"
          "   <- biais systématique du marché (favori/outsider)")
    for name, coef in off.correction_report(feats)[:12]:
        barre = "#" * min(int(abs(coef) * 200), 40)
        print(f"    {name:<32} {coef:+.4f}  {barre}")

    print(f"\n{'-' * 92}")
    ampl = float(np.abs(off.coef_).max())
    if ampl < 0.02:
        print("  Tous les coefficients sont quasi nuls: à cote donnée, aucune de ces")
        print("  features ne porte d'information que le marché n'aurait déjà intégrée.")
        print("  Conclusion: le gisement n'est pas dans le modèle mais dans les données")
        print("  (sources absentes du prix) ou dans le marché visé (lignes moins")
        print("  travaillées, cotes d'ouverture).")
    else:
        print(f"  Coefficient le plus fort: {ampl:.4f} en log-odds/écart-type.")
        print("  Vérifiez sa STABILITÉ en walk-forward avant d'en tirer une stratégie:")
        print("  un coefficient significatif sur une seule période reste une hypothèse.")
    print()

    if args.walk_forward:
        walk_forward(full, feats, args.C, int(args.test_start[:4]),
                     int(full["tourney_date"].max().year))


if __name__ == "__main__":
    main()
