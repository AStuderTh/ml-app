"""Colonne vertébrale de la base: Sackmann fait foi, TML enrichit.

Principe de conception, opposé à celui de l'ancien `pool_schema.py`:

    UNE LIGNE DE LA BASE = UN MATCH SACKMANN.

Une comparaison incertaine ne peut jamais créer de ligne — au pire elle laisse
une colonne à NULL. C'est ce qui élimine par construction les doublons: dans
l'ancien schéma, un rapprochement raté matérialisait les DEUX côtés en lignes
distinctes, et le même match réel se retrouvait deux fois en base (3 390 cas
mesurés), comptant double dans l'Elo, la forme et le H2H.

TML est un quasi-clone de Sackmann (même schéma à `indoor` près, volumes à
±64 matchs près). Il n'a donc pas à être "fusionné": il est joint sur clé
EXACTE uniquement, ce qui suffit à 90 % des lignes, et sert à deux choses:

  1. enrichir (`indoor`, et combler les trous de classement/surface/stats);
  2. couvrir la saison en cours, où TML a ~2 mois d'avance sur Sackmann —
     mais seulement à la maille du TOURNOI ENTIER absent de Sackmann, jamais
     match par match: un recouvrement partiel est précisément ce qui fabrique
     des doublons.
"""
import pandas as pd

from . import normalize as nm

# Colonnes que TML peut combler quand Sackmann ne les porte pas. Sackmann reste
# prioritaire partout: en cas de désaccord de valeur c'est lui qui gagne, TML
# n'intervient que sur une case vide.
FILLABLE_COLS = [
    "surface", "draw_size", "best_of", "minutes", "score",
    "winner_seed", "winner_entry", "winner_hand", "winner_ht", "winner_ioc",
    "winner_age", "winner_rank", "winner_rank_points",
    "loser_seed", "loser_entry", "loser_hand", "loser_ht", "loser_ioc",
    "loser_age", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
]

# Seuil de recouvrement de plateau au-delà duquel un tournoi TML est considéré
# comme DÉJÀ présent dans Sackmann sous une autre clé (et n'a donc pas le droit
# de créer des lignes). Volontairement bas: le coût d'un faux "absent" est un
# tournoi entier en double, celui d'un faux "présent" est un tournoi manquant
# le temps que Sackmann publie.
SAME_TOURNEY_OVERLAP = 0.30
SAME_TOURNEY_DAYS = 10


def _keys(df: pd.DataFrame):
    """Les deux clés EXACTES de rapprochement, construites en une passe
    vectorisée (`iterrows` sur 184 000 lignes coûte des dizaines de secondes).

    Les clés sont de simples CHAÎNES, et non des tuples de frozensets: à cette
    volumétrie (4 index de 184 000 entrées) la différence d'empreinte mémoire
    se compte en gigaoctets, et la machine finit par ne plus pouvoir allouer.

    La clé de repli par date sert quand le tourney_id diverge entre les deux
    sources (Sydney est 'M001' chez Sackmann et '338' chez TML). Elle reste
    EXACTE sur les noms: on change de clé de tournoi, pas de niveau d'exigence
    sur l'identité des joueurs.
    """
    wn, ln = df["wn"].to_numpy(), df["ln"].to_numpy()
    # paire de joueurs ordonnée: le vainqueur peut différer entre sources
    pair = [a + "\x00" + b if a <= b else b + "\x00" + a for a, b in zip(wn, ln)]
    rnd = df["round"].astype(str).to_numpy()
    tid = df["tid_norm"].astype(str).to_numpy()
    day = pd.to_datetime(df["tourney_date"]).dt.strftime("%Y%m%d").to_numpy()
    exact = [t + "\x01" + r + "\x01" + p for t, r, p in zip(tid, rnd, pair)]
    dated = [d + "\x01" + r + "\x01" + p for d, r, p in zip(day, rnd, pair)]
    return exact, dated


def _tml_index(tml: pd.DataFrame):
    exact, dated = _keys(tml)
    by_exact, by_date = {}, {}
    for j, ke, kd in zip(tml.index, exact, dated):
        by_exact.setdefault(ke, []).append(j)
        by_date.setdefault(kd, []).append(j)
    return by_exact, by_date


def _player_set(df):
    return set(df["wn"]) | set(df["ln"])


def build_spine(sack: pd.DataFrame, tml: pd.DataFrame, log=print):
    """Retourne (spine_df, stats). `spine_df` a une ligne par match retenu,
    avec `sources` indiquant les sources ayant contribué."""
    by_exact, by_date = _tml_index(tml)

    matched_tml = set()
    link = {}       # index sackmann -> index tml
    n_exact = n_date = 0
    s_exact, s_dated = _keys(sack)
    for i, ke, kd in zip(sack.index, s_exact, s_dated):
        cands = by_exact.get(ke)
        kind = "exact"
        if not cands:
            cands = by_date.get(kd)
            kind = "date"
        if not cands:
            continue
        j = next((c for c in cands if c not in matched_tml), None)
        if j is None:
            continue
        matched_tml.add(j)
        link[i] = j
        if kind == "exact":
            n_exact += 1
        else:
            n_date += 1

    log(f"  jointure TML sur clé exacte: {n_exact} + {n_date} par date "
        f"= {len(link)}/{len(sack)} ({len(link)/max(len(sack),1):.1%})")

    # --- lignes Sackmann, enrichies par TML ---
    rows = sack.copy()
    rows["indoor"] = None
    rows["winner_id_tml"] = None
    rows["loser_id_tml"] = None
    rows["sources"] = "sackmann"

    if link:
        s_idx = list(link.keys())
        t_idx = [link[i] for i in s_idx]
        # vue restreinte aux colonnes réellement lues: copier les 50 colonnes
        # de TML pour 176 000 lignes coûtait plusieurs centaines de Mo pour rien.
        wanted = [c for c in (FILLABLE_COLS + ["indoor", "winner_id", "loser_id"])
                  if c in tml.columns]
        tsub = tml.loc[t_idx, wanted]
        rows.loc[s_idx, "indoor"] = tsub["indoor"].to_numpy()
        rows.loc[s_idx, "winner_id_tml"] = tsub["winner_id"].to_numpy()
        rows.loc[s_idx, "loser_id_tml"] = tsub["loser_id"].to_numpy()
        rows.loc[s_idx, "sources"] = "sackmann,tml"

        filled = 0
        fillable = [c for c in FILLABLE_COLS if c in tsub.columns]
        gap_all = rows[fillable].isna()
        s_pos = rows.index.get_indexer(s_idx)   # une seule fois, pas par colonne
        for col in fillable:
            gap = gap_all[col].to_numpy()
            donor = tsub[col].to_numpy()
            take = gap[s_pos] & pd.notna(donor)
            if not take.any():
                continue
            labels = [s_idx[k] for k in take.nonzero()[0]]
            values = donor[take]
            try:
                rows.loc[labels, col] = values
            except (TypeError, ValueError):
                # les deux sources ne s'accordent pas toujours sur le dtype
                # d'une même colonne (Sackmann code `winner_seed` en numérique,
                # TML y écrit aussi 'Q' pour les qualifiés). On ne promeut en
                # object que la poignée de colonnes concernées, plutôt que
                # toutes.
                rows[col] = rows[col].astype(object)
                rows.loc[labels, col] = values
            filled += int(take.sum())
        log(f"  valeurs manquantes comblées par TML: {filled}")

    # --- tournois TML entièrement absents de Sackmann ---
    tml_left = tml.loc[[j for j in tml.index if j not in matched_tml]]
    added_idx = []
    added_tourneys = []
    if len(tml_left):
        # taille totale de chaque tournoi TML, précalculée: la recalculer dans
        # la boucle en scannant `tml` la rendait quadratique (~370 M
        # comparaisons, plusieurs minutes).
        tml_sizes = tml.groupby("tid_norm").size()

        # tournois Sackmann indexés par jour de début, pour ne confronter les
        # plateaux qu'aux quelques tournois de la même semaine.
        sack_by_day = {}
        for tid, g in sack.groupby("tid_norm"):
            start = g["tourney_date"].min()
            if pd.isna(start):
                continue
            sack_by_day.setdefault(start.normalize(), []).append(_player_set(g))

        one_day = pd.Timedelta(days=1)
        for tid, g in tml_left.groupby("tid_norm"):
            # un tournoi TML dont NE SERAIT-CE QU'UN match a rejoint Sackmann
            # est déjà couvert: y ajouter le reste créerait du recouvrement
            # partiel, donc des doublons.
            if tml_sizes.get(tid, 0) != len(g):
                continue
            t_start = g["tourney_date"].min()
            if pd.isna(t_start):
                continue
            t_players = _player_set(g)
            twin = False
            base = t_start.normalize()
            for delta in range(-SAME_TOURNEY_DAYS, SAME_TOURNEY_DAYS + 1):
                for s_players in sack_by_day.get(base + delta * one_day, ()):
                    union = len(t_players | s_players)
                    if union and len(t_players & s_players) / union >= SAME_TOURNEY_OVERLAP:
                        twin = True
                        break
                if twin:
                    break
            if twin:
                continue
            added_idx.extend(g.index.tolist())
            added_tourneys.append((g["tourney_name"].iloc[0], t_start))

    rows["match_date_src"] = pd.NaT
    if added_idx:
        extra = tml.loc[added_idx].copy()
        extra["winner_id_tml"] = extra["winner_id"]
        extra["loser_id_tml"] = extra["loser_id"]
        extra["winner_id"] = None
        extra["loser_id"] = None
        extra["sources"] = "tml"
        # Sur la SAISON EN COURS, TML est alimenté en direct et date chaque
        # match individuellement, alors que les saisons closes (comme
        # Sackmann) portent la date de DÉBUT du tournoi. Sans ce recalage, ces
        # lignes réintroduiraient précisément le mélange de sémantiques que le
        # schéma final cherche à éliminer: `tourney_date` cesserait d'être
        # unique par édition, et tout découpage train/test temporel ou tri
        # chronologique redeviendrait ambigu.
        extra["match_date_src"] = extra["tourney_date"]
        extra["tourney_date"] = extra.groupby("tid_norm")["tourney_date"].transform("min")
        rows = pd.concat([rows, extra], ignore_index=True, sort=False)
        log(f"  tournois absents de Sackmann repris depuis TML: "
            f"{len(added_tourneys)} ({len(added_idx)} matchs)")
        for name, start in sorted(added_tourneys, key=lambda t: str(t[1]))[-6:]:
            log(f"      + {name} ({str(start)[:10]})")

    skipped = len(tml_left) - len(added_idx)
    log(f"  lignes TML sans contrepartie Sackmann, volontairement ignorées: {skipped}")

    rows = rows.rename(columns={"winner_id": "winner_id_sackmann",
                                "loser_id": "loser_id_sackmann"})
    rows["round_rank"] = rows["round"].map(nm.atp_round_rank)
    stats = {
        "n_sackmann": len(sack),
        "n_linked_tml": len(link),
        "n_added_from_tml": len(added_idx),
        "n_tml_ignored": skipped,
    }
    return rows.reset_index(drop=True), stats
