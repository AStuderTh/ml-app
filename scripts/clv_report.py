"""Closing Line Value: le modèle prend-il des prix que le marché resserre ensuite ?

Pourquoi cette mesure plutôt que le ROI: le ROI d'une stratégie de paris a une
variance énorme (une cote à 3.0 gagnée ou perdue déplace le résultat de
plusieurs points), il lui faut des milliers de paris pour sortir du bruit. Le
CLV compare deux PRIX au lieu de comparer un prix à un résultat binaire: il
élimine la variance du résultat sportif et devient lisible en quelques
centaines d'observations.

Interprétation:
  - CLV moyen positif  -> vous achetez systématiquement moins cher que le prix
    d'équilibre final. C'est la signature d'un edge réel, AVANT tout gain.
  - CLV moyen négatif  -> le marché se déplace CONTRE vos sélections. Aucun
    réglage de modèle ne compensera cela; le problème est en amont.

Le CLV mesure la qualité du TIMING et de la SÉLECTION, pas la rentabilité
finale: il ignore la marge du bookmaker. Un CLV de +1% ne couvre pas un
overround de 5%. Il indique une direction, pas un profit.

Usage:  py -3.13 scripts/clv_report.py [--edge 0.03]
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "odds_log.db")

MIN_MOVE_MINUTES = 30  # en deçà, ouverture et clôture sont le même état de marché


def load_snapshots(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        sys.exit(f"Base introuvable: {db_path}")
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(
            """SELECT captured_at, match_key, tournoi, joueur_1, joueur_2,
                      book_odds_j1, book_odds_j2, book_source, model_prob_j1
               FROM odds_snapshots""",
            con,
        )
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    for c in ("book_odds_j1", "book_odds_j2", "model_prob_j1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["captured_at", "match_key"]).sort_values("captured_at")


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par match: prix d'ouverture, prix de clôture, avis du modèle.

    Les cotes et la probabilité du modèle ne sont PAS toujours renseignées sur
    la même capture (l'appel bookmaker et l'inférence n'échouent pas ensemble).
    On les récupère donc indépendamment plutôt que d'exiger une ligne complète,
    ce qui ferait tomber l'échantillon exploitable de moitié.
    """
    has_odds = df.dropna(subset=["book_odds_j1", "book_odds_j2"])
    has_prob = df.dropna(subset=["model_prob_j1"])
    if has_odds.empty:
        sys.exit("Aucune capture avec des cotes bookmaker.")

    first = has_odds.groupby("match_key").first()
    last = has_odds.groupby("match_key").last()

    pairs = pd.DataFrame({
        "tournoi": first["tournoi"],
        "joueur_1": first["joueur_1"],
        "joueur_2": first["joueur_2"],
        "t_open": first["captured_at"],
        "t_close": last["captured_at"],
        "open_j1": first["book_odds_j1"], "open_j2": first["book_odds_j2"],
        "close_j1": last["book_odds_j1"], "close_j2": last["book_odds_j2"],
        "src_open": first["book_source"], "src_close": last["book_source"],
    })
    # Probabilité modèle la plus ancienne disponible: c'est celle dont on
    # disposait au moment d'engager le pari. Prendre la plus récente ferait
    # entrer de l'information postérieure à la décision.
    pairs["model_prob_j1"] = has_prob.groupby("match_key")["model_prob_j1"].first()

    span = (pairs["t_close"] - pairs["t_open"]).dt.total_seconds() / 60.0
    pairs["span_min"] = span
    return pairs


def report(pairs: pd.DataFrame, edge_threshold: float):
    n_total = len(pairs)
    usable = pairs[(pairs["span_min"] >= MIN_MOVE_MINUTES) & pairs["model_prob_j1"].notna()].copy()

    print(f"\n{'='*66}\n  RAPPORT CLV\n{'='*66}")
    print(f"  matchs avec cotes                       : {n_total}")
    print(f"  dont mouvement observable (>{MIN_MOVE_MINUTES} min)      : {int((pairs['span_min'] >= MIN_MOVE_MINUTES).sum())}")
    print(f"  dont probabilité modèle disponible      : {len(usable)}")

    mixed = usable[usable["src_open"] != usable["src_close"]]
    if not mixed.empty:
        print(f"  ⚠ {len(mixed)} matchs comparent deux bookmakers différents "
              f"(ouverture/clôture) — CLV contaminé, exclus.")
        usable = usable[usable["src_open"] == usable["src_close"]]

    if usable.empty:
        print("\n  Échantillon vide. Laissez le journal se remplir puis relancez.\n")
        return

    # Côté que la stratégie aurait joué, à l'OUVERTURE, avec les mêmes règles
    # que app/backtest.py: edge contre le prix BRUT (1/cote), marge comprise.
    edge1 = usable["model_prob_j1"] - 1.0 / usable["open_j1"]
    edge2 = (1.0 - usable["model_prob_j1"]) - 1.0 / usable["open_j2"]
    on_j1 = edge1 >= edge2
    best_edge = np.maximum(edge1, edge2)

    taken = np.where(on_j1, usable["open_j1"], usable["open_j2"])
    closed = np.where(on_j1, usable["close_j1"], usable["close_j2"])
    usable["clv"] = taken / closed - 1.0
    usable["edge"] = best_edge
    usable["side"] = np.where(on_j1, usable["joueur_1"], usable["joueur_2"])

    sel = usable[usable["edge"] >= edge_threshold]

    def _block(title, sub):
        if sub.empty:
            print(f"\n  {title}\n    (aucun match)")
            return
        clv = sub["clv"]
        se = clv.std(ddof=1) / np.sqrt(len(clv)) if len(clv) > 1 else np.nan
        # Une large part des lignes ne bouge pas du tout entre deux captures.
        # Rapporter le % de CLV positif sur l'ensemble ferait passer ces
        # ex aequo pour des échecs: 50% d'immobiles plafonnent mécaniquement le
        # ratio à ~25%, ce qui se lit comme un désastre alors que ce n'en est
        # pas un. Le taux qui a un sens se calcule sur les lignes qui bougent.
        moved = clv[clv.abs() > 1e-9]
        pct_flat = 1.0 - len(moved) / len(clv)
        print(f"\n  {title}")
        print(f"    n                : {len(clv)}  (dont {pct_flat:.0%} sans mouvement)")
        print(f"    CLV moyen        : {clv.mean():+.2%}" + (f"  (± {2*se:.2%} à 2 σ)" if se == se else ""))
        print(f"    CLV médian       : {clv.median():+.2%}")
        if len(moved):
            print(f"    % positif (parmi les lignes qui bougent) : {(moved > 0).mean():.1%}"
                  f"  [n={len(moved)}]")
        if se == se and abs(clv.mean()) > 2 * se:
            verdict = "SIGNIFICATIF (positif)" if clv.mean() > 0 else "SIGNIFICATIF (négatif)"
        else:
            verdict = "indiscernable du hasard à ce volume"
        print(f"    verdict          : {verdict}")

    _block(f"Sélections du modèle (edge >= {edge_threshold:.0%})", sel)
    _block("Tous les matchs (référence: dérive du marché)", usable)

    if not sel.empty:
        print(f"\n  {'-'*62}\n  Détail des 10 plus gros écarts de prix")
        top = sel.reindex(sel["clv"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top.iterrows():
            side_taken = r["open_j1"] if r["side"] == r["joueur_1"] else r["open_j2"]
            side_close = r["close_j1"] if r["side"] == r["joueur_1"] else r["close_j2"]
            print(f"    {r['clv']:+7.2%}  {str(r['side'])[:24]:<24} "
                  f"{side_taken:.2f} -> {side_close:.2f}  ({str(r['tournoi'])[:22]})")

    print(f"\n{'='*66}")
    if len(sel) < 100:
        print("  ⚠ Moins de 100 sélections: résultat INDICATIF. Le CLV demande\n"
              "    quelques centaines d'observations pour trancher. Relancez ce\n"
              "    script à mesure que data/odds_log.db se remplit.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03,
                    help="seuil d'edge (défaut 0.03, comme DEFAULT_STRATEGY)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    # La console Windows est en cp1252 par défaut: sans cela le premier accent
    # ou symbole du rapport lève UnicodeEncodeError et tue le script.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    report(build_pairs(load_snapshots(args.db)), args.edge)


if __name__ == "__main__":
    main()
