"""Appariement des matchs entre sources.

Deux étapes, chacune produisant des paires "auto" (fusionnées directement) et
des paires "à valider" (ajoutées à la file de validation manuelle, voir
review_io.py) :

  1. Sackmann <-> TML          (clé quasi exacte: tid_norm + round + noms)
  2. (Sackmann+TML) <-> tennis-data.co   (aucun id commun: tournoi/round/noms approx.)

Rien n'est jamais silencieusement supprimé: ce qui n'est pas apparié reste
comme enregistrement autonome (source unique) dans la sortie finale.
"""
from dataclasses import dataclass, field

import pandas as pd

from . import normalize as nm

AUTO_THRESHOLD = 0.75
REVIEW_FLOOR = 0.45


def greedy_bipartite_match(scored_pairs):
    """scored_pairs: iterable de (i, j, score). Retourne la liste des paires
    retenues (i, j, score), triées par score décroissant, chaque i et j
    utilisé au plus une fois (appariement glouton, suffisant vu la petite
    taille des groupes: quelques matchs par tournoi/round)."""
    used_a, used_b = set(), set()
    chosen = []
    for i, j, score in sorted(scored_pairs, key=lambda t: -t[2]):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        chosen.append((i, j, score))
    return chosen


@dataclass
class MatchResult:
    auto_pairs: list = field(default_factory=list)      # (idx_a, idx_b, score, reason)
    review_pairs: list = field(default_factory=list)    # (idx_a, idx_b, score, reason)
    unmatched_a: set = field(default_factory=set)
    unmatched_b: set = field(default_factory=set)


def match_sackmann_tml(sack: pd.DataFrame, tml: pd.DataFrame) -> MatchResult:
    result = MatchResult(unmatched_a=set(sack.index), unmatched_b=set(tml.index))

    # --- tier 1: clé exacte (tid_norm, round, {noms complets normalisés}) ---
    def key_of(row):
        return (row["tid_norm"], row["round"], frozenset([row["wn"], row["ln"]]))

    tml_by_key = {}
    for j, row in tml.iterrows():
        tml_by_key.setdefault(key_of(row), []).append(j)

    for i, row in sack.iterrows():
        cands = tml_by_key.get(key_of(row))
        if cands:
            j = cands.pop(0)
            result.auto_pairs.append((i, j, 1.0, "exact_key"))
            result.unmatched_a.discard(i)
            result.unmatched_b.discard(j)

    # --- tier 2: appariement flou par groupe (tid_norm, round) ---
    def _fuzzy_pass(group_cols, reason_auto, reason_review, name_only=False):
        remaining_sack = sack.loc[list(result.unmatched_a)]
        remaining_tml = tml.loc[list(result.unmatched_b)]
        if remaining_sack.empty or remaining_tml.empty:
            return
        groups_a = remaining_sack.groupby(group_cols).groups
        groups_b = remaining_tml.groupby(group_cols).groups

        for gkey, idx_a in groups_a.items():
            idx_b = groups_b.get(gkey)
            if idx_b is None or len(idx_b) == 0:
                continue
            pairs = []
            for i in idx_a:
                ra = sack.loc[i]
                for j in idx_b:
                    rb = tml.loc[j]
                    wscore = nm.best_surname_score(ra["wn"], rb["wn"])
                    lscore = nm.best_surname_score(ra["ln"], rb["ln"])
                    score = (wscore + lscore) / 2
                    if name_only:
                        # pas de tid en commun ici: on exige en plus une
                        # cohérence minimale du nom de tournoi pour éviter
                        # de confondre deux tournois disputés la même semaine
                        tsim = nm.tourney_name_similarity(ra["tourney_name"], rb["tourney_name"])
                        score = score * (0.5 + 0.5 * tsim)
                    if score >= REVIEW_FLOOR:
                        pairs.append((i, j, score))
            for i, j, score in greedy_bipartite_match(pairs):
                if score >= AUTO_THRESHOLD:
                    result.auto_pairs.append((i, j, score, reason_auto))
                else:
                    result.review_pairs.append((i, j, score, reason_review))
                result.unmatched_a.discard(i)
                result.unmatched_b.discard(j)

    _fuzzy_pass(["tid_norm", "round"], "fuzzy_name", "fuzzy_name_low_confidence")
    # --- tier 2b: repli quand le tourney_id diverge carrément entre sources
    # (arrive pour certains tournois, ex. Sydney = 'M001' chez Sackmann vs
    # '338' chez TML) mais que la date de la semaine de tournoi, elle,
    # concorde presque toujours entre les deux sources.
    _fuzzy_pass(["tourney_date", "round"], "fuzzy_name_date_fallback",
                "fuzzy_name_date_fallback_low_confidence", name_only=True)

    return result


def match_with_tennisdata(fused: pd.DataFrame, td: pd.DataFrame) -> MatchResult:
    """fused: pool Sackmann+TML déjà consolidé (une ligne par match), avec
    colonnes w_surname/l_surname/w_ginit/l_ginit/tourney_name/tourney_date/round.
    td: lignes tennis-data.co avec w_surname/l_surname/w_initials/l_initials/
    tourney_name_norm/year/round_rank/is_rr."""
    result = MatchResult(unmatched_a=set(fused.index), unmatched_b=set(td.index))

    fused = fused.copy()
    fused["tourney_name_norm"] = fused["tourney_name"].map(nm.norm_tourney_name)
    fused["year"] = pd.to_datetime(fused["tourney_date"]).dt.year

    # --- association des instances de tournoi (année + nom normalisé) ---
    td_tourneys = td.groupby(["year", "tourney_name_norm"]).size().reset_index()[["year", "tourney_name_norm"]]
    fused_tourneys = fused.groupby(["year", "tourney_name_norm"]).size().reset_index()[["year", "tourney_name_norm"]]

    # Empreintes par instance de tournoi: date de début + plateau engagé. Le
    # recouvrement des joueurs est un signal bien plus robuste que le nom: il
    # survit à n'importe quel changement de sponsor (sans quoi TOURNEY_ALIASES
    # doit être complété à la main pour chaque nouveau nom commercial), alors
    # que deux tournois distincts ont des plateaux disjoints.
    f_start = fused.groupby(["year", "tourney_name_norm"])["tourney_date"].min()
    t_start = td.groupby(["year", "tourney_name_norm"])["Date"].min()
    f_players = fused.groupby(["year", "tourney_name_norm"]).apply(
        lambda g: set(g["wn"].map(nm.surname_guess)) | set(g["ln"].map(nm.surname_guess))
    )
    t_players = td.groupby(["year", "tourney_name_norm"]).apply(
        lambda g: set(g["w_surname"]) | set(g["l_surname"])
    )

    def _player_overlap(fkey, tkey):
        """Similarité de plateau (Jaccard) entre deux instances de tournoi.

        Jaccard et non coefficient de recouvrement: ce dernier vaut ~1 dès
        qu'un petit tournoi est inclus dans un gros, or les ATP 250 de la
        semaine précédant un Grand Chelem ont justement un plateau
        entièrement réengagé ensuite (Nice/Genève avant Roland-Garros) — ils
        écrasaient donc la marge exigée sur le 2e meilleur candidat."""
        a, b = f_players.get(fkey), t_players.get(tkey)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    tourney_link = {}  # (year, td_name_norm) -> (year, fused_name_norm)
    fused_exact = set(map(tuple, fused_tourneys.values.tolist()))
    for year, td_name in td_tourneys.values.tolist():
        if (year, td_name) in fused_exact:
            tourney_link[(year, td_name)] = (year, td_name)
            continue
        # repli flou: meilleure correspondance parmi les tournois de la même
        # année. Seuil volontairement strict + marge sur le 2e meilleur:
        # deux tournois différents joués la même semaine (ex. 'Australian
        # Open' vs 'Australian Hardcourt Championships') ne doivent jamais
        # se faire relier sur un score ambigu.
        same_year = fused_tourneys[fused_tourneys.year == year]
        tkey = (year, td_name)
        td_start = t_start.get(tkey)
        scored = []
        for cand in same_year.tourney_name_norm:
            fkey = (year, cand)
            score = nm.tourney_name_similarity(td_name, cand)
            # le plateau n'est confronté que si les deux instances tombent la
            # même semaine: sans cette garde, deux étapes successives d'une
            # même tournée (plateaux très proches) pourraient se relier.
            f_st = f_start.get(fkey)
            if td_start is not None and f_st is not None and abs((f_st - td_start).days) <= 10:
                score = max(score, _player_overlap(fkey, tkey))
            scored.append((score, cand))
        scored.sort(reverse=True)
        if not scored:
            continue
        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - runner_up
        # deux régimes d'acceptation. Le nom seul reste exigeant (0.72). La
        # preuve par le plateau tolère un score plus bas — les deux sources
        # ne couvrent pas exactement les mêmes tours, ce qui plafonne le
        # Jaccard d'un vrai couple autour de 0.7 — mais réclame en échange
        # une marge nettement plus large sur le 2e candidat, qui lui reste
        # sous 0.2 dès qu'il s'agit d'un autre tournoi.
        if (best_score >= 0.72 and margin >= 0.12) or (best_score >= 0.5 and margin >= 0.30):
            tourney_link[(year, td_name)] = (year, best_name)

    fused_groups = fused.groupby(["year", "tourney_name_norm"]).groups
    td_groups = td.groupby(["year", "tourney_name_norm"]).groups

    def pair_score(fr, tr):
        w = nm.best_surname_score(fr["wn"], tr["w_surname"])
        if not nm.initials_compatible(tr["w_initials"], fr["w_ginit"]):
            w *= 0.4
        l = nm.best_surname_score(fr["ln"], tr["l_surname"])
        if not nm.initials_compatible(tr["l_initials"], fr["l_ginit"]):
            l *= 0.4
        w_sw = nm.best_surname_score(fr["wn"], tr["l_surname"])
        if not nm.initials_compatible(tr["l_initials"], fr["w_ginit"]):
            w_sw *= 0.4
        l_sw = nm.best_surname_score(fr["ln"], tr["w_surname"])
        if not nm.initials_compatible(tr["w_initials"], fr["l_ginit"]):
            l_sw *= 0.4
        return (w + l) / 2, (w_sw + l_sw) / 2

    for (td_year, td_name), td_idx in td_groups.items():
        fkey = tourney_link.get((td_year, td_name))
        if fkey is None:
            continue
        f_idx = fused_groups.get(fkey)
        if f_idx is None or len(f_idx) == 0:
            continue
        # plusieurs variantes de nom côté tennis-data.co peuvent se relier au
        # même tournoi fusionné (ex. faute de frappe / graphie différente
        # une année donnée) : sans ce filtre, le même match Sackmann+TML
        # serait retraité (et donc dupliqué) à chaque variante rencontrée.
        f_idx = [x for x in f_idx if x in result.unmatched_a]
        if not f_idx:
            continue
        sub_fused, sub_td = fused.loc[f_idx], td.loc[td_idx]

        # sous-groupes par round: rang-depuis-la-finale pour les rounds à
        # élimination directe, poule "Round Robin" traitée comme un seul bloc
        elim_f = sub_fused[sub_fused["round"] != "RR"]
        elim_td = sub_td[~sub_td["is_rr"]]
        rr_f = sub_fused[sub_fused["round"] == "RR"]
        rr_td = sub_td[sub_td["is_rr"]]

        subgroups = []
        for rnk, ga in elim_f.groupby("round_rank").groups.items():
            gb = elim_td[elim_td.round_rank == rnk].index
            subgroups.append((ga, gb))
        if len(rr_f) and len(rr_td):
            subgroups.append((rr_f.index, rr_td.index))

        for idx_a, idx_b in subgroups:
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            pairs, swapped_flag = [], {}
            for i in idx_a:
                fr = fused.loc[i]
                for j in idx_b:
                    tr = td.loc[j]
                    score, score_sw = pair_score(fr, tr)
                    best = max(score, score_sw)
                    if best >= REVIEW_FLOOR:
                        pairs.append((i, j, best))
                        swapped_flag[(i, j)] = score_sw > score
            for i, j, score in greedy_bipartite_match(pairs):
                reason = "swapped_winner_loser" if swapped_flag.get((i, j)) else "fuzzy_name"
                if score >= AUTO_THRESHOLD and not swapped_flag.get((i, j)):
                    result.auto_pairs.append((i, j, score, reason))
                else:
                    result.review_pairs.append((i, j, score, reason))
                result.unmatched_a.discard(i)
                result.unmatched_b.discard(j)

    return result
