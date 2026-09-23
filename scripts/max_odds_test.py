"""Estimer la probabilité sur le consensus, parier au prix d'un book réel.

Idée testée: la correction apprise par le modèle offset vaut ~2,5 points de
probabilité, soit l'ordre de grandeur de la marge sur les cotes moyennes (~5%
d'overround, ~2,5 points par côté). Contre un book à marge plus faible, elle
pourrait passer devant.

Trois stratégies sur exactement le même jeu de matchs:

  A. consensus au prix moyen  -> contrôle négatif; doit perdre, c'est la marge
  B. consensus au prix de pari -> line shopping pur, SANS modèle
  C. offset au prix de pari    -> line shopping + correction apprise

La comparaison qui compte est C contre B. Si elles se valent, la correction
n'apporte rien et tout gain éventuel vient du magasinage de cotes — conclusion
utile, mais très différente.

CHOIX DU PRIX (--price), et c'est le point critique de ce script: la colonne
`max` de tennis-data.co N'EST PAS un prix cotable. MaxW et MaxL sont chacun un
maximum pris sur tous les bookmakers ET sur toute la période pré-match; 27% des
paires ainsi formées affichent un overround NÉGATIF, c'est-à-dire un arbitrage
garanti, ce qui ne survit pas quelques secondes dans un marché réel. Backtester
dessus produit un ROI fictif (~+8%) qui disparaît intégralement au prix d'un
book réel. Le défaut est donc `ps` (Pinnacle), dont l'overround mesuré (~2,6%)
est plausible.

Usage:  py -3.13 scripts/max_odds_test.py [--price ps|b365|max]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest import run_backtest, segment_metrics  # noqa: E402
from app.data import ODDS_CHOICES, load_matches  # noqa: E402
from app.features import (build_training_frame, compute_chronological_features,  # noqa: E402
                          expand_features)
from app.offset_model import OffsetLogisticRegression  # noqa: E402
from scripts.offset_test import FEATURES  # noqa: E402


PRICE_SOURCES = {
    # Prix de PARI candidats. 'max' est conservé pour montrer pourquoi il est
    # inutilisable: 27% de ses paires affichent un overround NÉGATIF, c'est-à-dire
    # un arbitrage garanti. Une telle paire n'a jamais été cotable simultanément
    # — MaxW et MaxL sont chacun un maximum pris sur TOUS les bookmakers ET sur
    # toute la période pré-match. Backtester dessus, c'est parier à des prix qui
    # n'ont pas existé.
    "max": ("odds_w_max", "odds_l_max"),
    "ps": ("odds_w_ps", "odds_l_ps"),        # Pinnacle: book réel, marge faible
    "b365": ("odds_w_b365", "odds_l_b365"),
}


def attach_price(frame: pd.DataFrame, matches: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ajoute les cotes Max en respectant l'attribution p1/p2 DÉJÀ tirée.

    On ne reconstruit surtout pas une seconde frame avec odds_w_max: le tirage
    aléatoire gagnant/perdant -> p1/p2 dépend du nombre de lignes retenues, et
    Avg et Max n'ont pas exactement la même couverture. Deux frames auraient
    donc des attributions DIFFÉRENTES, et joindre l'une sur l'autre inverserait
    silencieusement les cotes d'une partie des matchs — le pire bug possible
    ici, puisqu'il ressemblerait à un edge.

    `label` porte l'information nécessaire: label==1 signifie que p1 est le
    vainqueur, donc que odds_w_max est la cote de p1.
    """
    w_col, l_col = PRICE_SOURCES[source]
    m = matches.set_index("match_id")[[w_col, l_col]]
    out = frame.join(m, on="match_id")
    p1_won = out["label"].values == 1
    out["odds_p1_bet"] = np.where(p1_won, out[w_col], out[l_col])
    out["odds_p2_bet"] = np.where(p1_won, out[l_col], out[w_col])
    out = out.dropna(subset=["odds_p1_bet", "odds_p2_bet"])
    # tennis-data.co encode une cote manquante par 0 sur certains books
    # (Pinnacle notamment). Une cote décimale est nécessairement > 1: sans ce
    # filtre, 1/0 = inf fait passer ces lignes pour des paris à edge infini.
    return out[(out["odds_p1_bet"] > 1.0) & (out["odds_p2_bet"] > 1.0)]


def run(frame, probs, odds_p1_col, odds_p2_col, strategy):
    df = frame.copy()
    df["odds_p1"] = df[odds_p1_col]
    df["odds_p2"] = df[odds_p2_col]
    return run_backtest(probs, df, strategy)


def show(label, bets, metrics, overround=None):
    roi = metrics["roi"]
    n = metrics["n_bets"]
    if n == 0:
        print(f"  {label:<36} aucun pari")
        return
    # Erreur-type du ROI: sans elle, +4% sur 300 paris se lit comme un succès.
    se = bets["profit"].std(ddof=1) / np.sqrt(n) / bets["stake"].mean()
    flag = "significatif" if abs(roi) > 2 * se else "dans le bruit"
    extra = f"  overround={overround:.1%}" if overround is not None else ""
    print(f"  {label:<36} n={n:>5}  ROI={roi*100:+6.2f}% ±{2*se*100:4.2f}  "
          f"profit={metrics['total_profit']:+8.1f}  [{flag}]{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--price", default="ps", choices=sorted(PRICE_SOURCES),
                    help="prix auquel on parie (défaut ps=Pinnacle, un book RÉEL; "
                         "'max' est un prix synthétique, cf. PRICE_SOURCES)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    strategy = {"edge_threshold": args.edge, "stake_mode": "flat", "flat_stake": 1.0}

    print("Chargement + features chronologiques... [~1 min]")
    matches = load_matches()
    chrono = compute_chronological_features(matches)
    avg_w, avg_l = ODDS_CHOICES["Moyenne du marché (Avg)"]
    full = build_training_frame(chrono, avg_w, avg_l)

    feats = expand_features(FEATURES)
    full = full.dropna(subset=feats + ["label"])
    full = attach_price(full, matches, args.price)

    split = pd.Timestamp(args.test_start)
    train, test = full[full["tourney_date"] < split], full[full["tourney_date"] >= split]
    print(f"  train={len(train)}  test={len(test)}  (matchs avec Avg ET {args.price})")

    over_avg = (1 / test["odds_p1"] + 1 / test["odds_p2"]).mean() - 1
    ov_bet = 1 / test["odds_p1_bet"] + 1 / test["odds_p2_bet"] - 1
    over_bet = ov_bet.mean()
    print(f"  overround moyen: Avg={over_avg:.2%}   {args.price}={over_bet:.2%}   "
          f"-> {(over_avg - over_bet) * 100 / 2:.2f} pts de marge récupérés par côté")
    # Contrôle de plausibilité du prix: un overround négatif est un arbitrage
    # garanti. S'il s'en trouve une part notable, la paire de cotes n'a jamais
    # été cotable simultanément et tout ROI qui en découle est fictif.
    part_neg = float((ov_bet < 0).mean())
    if part_neg > 0.02:
        print(f"  (!) {part_neg:.1%} des paires sont en ARBITRAGE (overround < 0):")
        print(f"      prix synthétique, non cotable. Les ROI ci-dessous sont fictifs.")

    model = OffsetLogisticRegression(offset_index=0, C=args.C)
    model.fit(train[feats].values, train["label"].values)
    p_offset = model.predict_proba(test[feats].values)[:, 1]
    p_consensus = test["implied_prob_p1"].values   # Avg dévigorisé

    print(f"\n{'=' * 94}\n  BACKTEST (test >= {args.test_start}, seuil d'edge {args.edge:.0%}, "
          f"mise fixe)\n{'=' * 94}")
    res = {}
    for label, probs, o1, o2, ov in [
        ("A. consensus @ prix moyen", p_consensus, "odds_p1", "odds_p2", over_avg),
        (f"B. consensus @ {args.price}", p_consensus, "odds_p1_bet", "odds_p2_bet", over_bet),
        (f"C. OFFSET @ {args.price}", p_offset, "odds_p1_bet", "odds_p2_bet", over_bet),
    ]:
        bets, met = run(test, probs, o1, o2, strategy)
        show(label, bets, met, ov)
        res[label[0]] = (bets, met)

    bets_b, met_b = res["B"]
    bets_c, met_c = res["C"]
    print(f"\n{'-' * 94}")
    if met_b["n_bets"] and met_c["n_bets"]:
        delta = (met_c["roi"] - met_b["roi"]) * 100
        print(f"  Apport de la correction (C - B): {delta:+.2f} points de ROI")
        if abs(delta) < 1.0:
            print("  -> La correction n'ajoute rien de lisible au line shopping seul.")
        else:
            print("  -> Écart notable; à confirmer année par année avant d'y croire.")

    if met_c["n_bets"]:
        print(f"\n  Décomposition de C par niveau de tournoi")
        seg = segment_metrics(bets_c)
        for lvl, r in seg.iterrows():
            print(f"    {str(lvl):<6} n={int(r['n_bets']):>5}  ROI={r['roi']*100:+6.2f}% "
                  f"±{r['roi_se']*200:4.2f}  {'pariable live' if r['pariable_live'] else ''}")

        print(f"\n  Décomposition de C par année")
        bc = bets_c.copy()
        bc["annee"] = pd.to_datetime(bc["date"]).dt.year
        for year, g in bc.groupby("annee"):
            roi = g["profit"].sum() / g["stake"].sum()
            se = g["profit"].std(ddof=1) / np.sqrt(len(g)) / g["stake"].mean()
            print(f"    {year}  n={len(g):>5}  ROI={roi*100:+6.2f}% ±{2*se*100:4.2f}")

    print(f"\n{'=' * 94}")
    if args.price == "max":
        print("  RAPPEL: 'max' n'est PAS un prix cotable (27% de paires en arbitrage).")
        print("  Tout ROI ci-dessus est fictif. Relancez avec --price ps.\n")
    else:
        print(f"  Prix de pari: {args.price}, book réel, overround {over_bet:.2%} — plausible.")
        print("  Reste non modélisé: limites de mise et fermeture de compte, qui")
        print("  frappent précisément les parieurs gagnants.\n")


if __name__ == "__main__":
    main()
