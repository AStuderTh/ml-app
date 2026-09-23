"""À partir de combien de matchs d'historique le modèle devient-il fiable ?

Motivation: app/live_match.py attribue DEFAULT_ELO (1500) à tout joueur absent
de la base, silencieusement. Le modèle produit alors une probabilité d'aspect
normal qui reflète surtout cette valeur par défaut — et l'écart avec une vraie
cote se lit comme un edge. Ce script cherche le seuil sous lequel il faut
couper le signal.

Le critère n'est PAS l'AUC brute. Les matchs entre joueurs peu documentés sont
intrinsèquement moins prévisibles, pour le modèle comme pour le bookmaker: une
AUC qui baisse peut simplement signaler des matchs plus serrés, pas un modèle
défaillant. La quantité qui décide est l'écart de log-loss AU MARCHÉ dans le
même bucket. Le seuil utile est là où cet écart se retourne — où le modèle
cesse d'apporter quelque chose que la cote ne contient pas déjà.

Usage:
    py -3.13 scripts/experience_threshold.py
    py -3.13 scripts/experience_threshold.py --test-start 2022-01-01 --odds b365
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

from app.data import ODDS_CHOICES, load_matches  # noqa: E402
from app.features import (build_training_frame, compute_chronological_features,  # noqa: E402
                          expand_features)
from app.train import train_and_evaluate  # noqa: E402

# Bornes de bucket sur min_played. Resserrées en bas: c'est là que l'Elo bouge
# vite (chaque match déplace fortement une valeur encore proche de 1500), donc
# c'est là que le seuil se joue.
BUCKETS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250, np.inf]

DEFAULT_FEATURES = [
    "elo_diff", "elo_surface_diff", "rank_diff", "form_diff", "h2h_diff",
    "rest_days_diff", "age_diff", "surface", "best_of_5",
]

ODDS_ALIASES = {"avg": "Moyenne du marché (Avg)", "b365": "Bet365",
                "ps": "Pinnacle", "max": "Meilleure cote (Max)"}


def _bucket_label(lo, hi):
    return f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+"


def _rowwise_ll(y, p):
    """Log-loss match par match (log_loss de sklearn n'en rend que la moyenne)."""
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def analyse(test_frame: pd.DataFrame, min_rows: int):
    """Compare, bucket par bucket, le modèle au marché sur la MÊME métrique.

    La log-loss du marché se calcule sur la probabilité implicite dévigorisée
    (implied_prob_p1): c'est la meilleure estimation que le bookmaker publie,
    et donc la seule référence honnête pour un modèle probabiliste. Ne pas la
    confondre avec le prix brut, qui sert à décider un pari (cf. backtest.py).
    """
    rows = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        sub = test_frame[(test_frame["min_played"] >= lo) & (test_frame["min_played"] < hi)]
        if len(sub) < min_rows or sub["label"].nunique() < 2:
            rows.append({"bucket": _bucket_label(lo, hi), "n": len(sub)})
            continue

        y = sub["label"].values
        p_model = sub["model_prob_p1"].values
        p_market = sub["implied_prob_p1"].values

        pm = np.clip(p_model, 1e-6, 1 - 1e-6)
        pk = np.clip(p_market, 1e-6, 1 - 1e-6)
        ll_model = log_loss(y, pm, labels=[0, 1])
        ll_market = log_loss(y, pk, labels=[0, 1])

        # Écart APPARIÉ, match par match: modèle et marché prédisent les mêmes
        # rencontres, donc la variance commune (certains matchs sont
        # imprévisibles pour tout le monde) s'annule dans la différence. Sans
        # cet appariement, l'incertitude serait tellement large qu'aucun bucket
        # ne se distinguerait, et on lirait du bruit comme un signal.
        d = _rowwise_ll(y, pk) - _rowwise_ll(y, pm)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan

        rows.append({
            "bucket": _bucket_label(lo, hi),
            "n": len(sub),
            "auc_model": roc_auc_score(y, p_model),
            "auc_market": roc_auc_score(y, p_market),
            "ll_model": ll_model,
            "ll_market": ll_market,
            # Positif = le modèle bat le marché sur ce bucket.
            "gain": ll_market - ll_model,
            "gain_se": se,
            "favori_gagne": float((y == (p_market >= 0.5)).mean()),
        })
    return pd.DataFrame(rows)


def print_report(res: pd.DataFrame, n_test: int):
    print(f"\n{'=' * 78}")
    print("  FIABILITÉ DU MODÈLE PAR PROFONDEUR D'HISTORIQUE")
    print(f"  (min_played = nb de matchs connus du joueur le moins documenté)")
    print(f"{'=' * 78}")
    print(f"  {'bucket':<10} {'n':>6}  {'AUC mod':>8} {'AUC mkt':>8}  "
          f"{'LL mod':>7} {'LL mkt':>7}  {'gain':>18}  verdict")
    print(f"  {'-' * 78}")

    for _, r in res.iterrows():
        if "auc_model" not in r or pd.isna(r.get("auc_model")):
            print(f"  {r['bucket']:<10} {int(r['n']):>6}   (échantillon trop faible)")
            continue
        # Un gain n'est lisible qu'au regard de son incertitude: sans ce test,
        # un +0.0014 sur un bucket se lit comme une victoire du modèle.
        if pd.notna(r["gain_se"]) and abs(r["gain"]) > 2 * r["gain_se"]:
            verdict = "modèle > marché" if r["gain"] > 0 else "marché > modèle"
        else:
            verdict = "égalité (dans le bruit)"
        gain_txt = f"{r['gain']:+.4f}±{2 * r['gain_se']:.4f}"
        print(f"  {r['bucket']:<10} {int(r['n']):>6}  {r['auc_model']:>8.3f} {r['auc_market']:>8.3f}  "
              f"{r['ll_model']:>7.4f} {r['ll_market']:>7.4f}  {gain_txt:>18}  {verdict}")

    valid = res.dropna(subset=["gain"]) if "gain" in res.columns else pd.DataFrame()
    print(f"\n{'-' * 78}")
    if valid.empty:
        print("  Aucun bucket exploitable. Élargissez la fenêtre de test.")
        return

    print(f"  Jeu de test: {n_test} matchs.")
    print(f"  Gain global vs marché: {valid['gain'].mul(valid['n']).sum() / valid['n'].sum():+.4f} "
          f"de log-loss (positif = le modèle apporte).")

    # Seuls les buckets SIGNIFICATIVEMENT positifs comptent: un gain inférieur à
    # son propre bruit ne justifie pas de recommander un seuil de coupure.
    positifs = valid[valid["gain"] > 2 * valid["gain_se"]]

    if positifs.empty:
        # Cas le plus fréquent, et celui qu'il ne faut surtout pas déguiser en
        # seuil: si le modèle est battu PARTOUT, il n'existe aucun bucket où
        # couper le signal suffirait. Le garde-fou ne réglerait rien.
        ecart = valid["gain"].max() - valid["gain"].min()
        print("\n  VERDICT: le modèle ne bat le marché sur AUCUN bucket.")
        print("  La question du seuil d'historique ne se pose donc pas: couper les")
        print("  joueurs peu documentés ne rendrait pas le reste rentable.")
        print(f"\n  Amplitude du gain entre buckets: {ecart:.4f} de log-loss.")
        if ecart < 0.03:
            print("  Écart faible d'un bucket à l'autre -> la profondeur d'historique")
            print("  n'est PAS le facteur limitant. Le déficit est uniforme, donc")
            print("  structurel: features ou calibration, pas périmètre.")
        else:
            print("  Écart notable -> la profondeur joue, mais elle ne suffit pas à")
            print("  expliquer le déficit global.")
        print("\n  Priorité: réduire l'écart au marché (features, calibration, ou")
        print("  modèle offset sur logit(P_marché)) avant tout travail de filtrage.")
    else:
        negatifs_bas = valid[(valid["gain"] < -2 * valid["gain_se"])
                             & (valid.index < positifs.index.min())]
        if not negatifs_bas.empty:
            seuil = BUCKETS[int(negatifs_bas.index.max()) + 1]
            print(f"\n  SEUIL SUGGÉRÉ: couper le signal sous min_played = {int(seuil)}")
            print("  En dessous, le modèle fait moins bien que la cote seule — donc")
            print("  tout 'edge' qu'il y trouve est du bruit, pas de l'information.")
        else:
            print("\n  Les buckets déficitaires ne sont pas les plus bas: le problème")
            print("  n'est pas la profondeur d'historique. Regardez plutôt le découpage")
            print("  par niveau de tournoi (backtest.segment_metrics).")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--odds", default="avg", choices=sorted(ODDS_ALIASES))
    ap.add_argument("--algo", default="random_forest")
    ap.add_argument("--min-rows", type=int, default=150,
                    help="taille minimale d'un bucket pour être évalué (défaut 150)")
    ap.add_argument("--features", default=None,
                    help="liste séparée par des virgules; défaut = jeu de base. "
                         "Le verdict dépend du jeu de features: testez celui de "
                         "votre meilleur modèle enregistré, pas seulement le défaut.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    odds_w, odds_l = ODDS_CHOICES[ODDS_ALIASES[args.odds]]

    print("Chargement des matchs...")
    matches = load_matches()
    print(f"  {len(matches)} matchs")
    print("Calcul des features chronologiques (Elo, forme, H2H)... [~1 min]")
    chrono = compute_chronological_features(matches)
    full = build_training_frame(chrono, odds_w, odds_l)
    print(f"  {len(full)} matchs cotés exploitables ({args.odds})")

    split = pd.Timestamp(args.test_start)
    train = full[full["tourney_date"] < split]
    test = full[full["tourney_date"] >= split]
    if train.empty or test.empty:
        sys.exit(f"Découpage vide à {args.test_start}: train={len(train)} test={len(test)}")
    print(f"Entraînement sur {len(train)} matchs (< {args.test_start}), "
          f"test sur {len(test)}...")

    # 'surface' est une pseudo-feature: elle doit être développée en colonnes
    # one-hot avant d'indexer le DataFrame (cf. features.expand_features).
    base_feats = ([f.strip() for f in args.features.split(",") if f.strip()]
                  if args.features else DEFAULT_FEATURES)
    print(f"  features: {', '.join(base_feats)}")
    feats = expand_features(base_feats)
    result = train_and_evaluate(train, test, args.algo, {}, feats, {})
    tf = result["test_frame"]
    print(f"  AUC globale: {result['metrics']['auc']:.4f}")

    print_report(analyse(tf, args.min_rows), len(tf))


if __name__ == "__main__":
    main()
