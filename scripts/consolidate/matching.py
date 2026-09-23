"""Appariement approximatif de la colonne vertébrale avec tennis-data.co.

C'est le SEUL appariement flou du pipeline, et il ne sert qu'à attacher des
cotes: tennis-data.co ne partage aucun identifiant avec les autres sources
(tournoi/tour/noms abrégés uniquement). Le rapprochement Sackmann <-> TML,
lui, se fait sur clé exacte dans `spine.py` — deux référentiels quasi
identiques n'ont pas besoin de logique floue, et lui en donner produisait
surtout des doublons.

Un couple non apparié ne coûte qu'une cote manquante (cf. odds_attach.py):
rien n'est supprimé, et rien n'est dupliqué.
"""
from dataclasses import dataclass, field

import pandas as pd

from . import normalize as nm

AUTO_THRESHOLD = 0.75
REVIEW_FLOOR = 0.45

# Recouvrement de plateau (Jaccard) au-dessus duquel on considère tenir une
# preuve DIRECTE que deux instances de tournoi sont la même. Fixé sous le
# plafond réaliste (~0.7) d'un vrai couple: les deux sources ne couvrent pas
# exactement les mêmes tours, donc le Jaccard d'un vrai couple n'atteint
# jamais 1.
ROSTER_EVIDENCE_FLOOR = 0.50


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


def match_with_tennisdata(fused: pd.DataFrame, td: pd.DataFrame) -> MatchResult:
    """fused: pool Sackmann+TML déjà consolidé (une ligne par match), avec
    colonnes w_surname/l_surname/w_ginit/l_ginit/tourney_name/tourney_date/round.
    td: lignes tennis-data.co avec w_surname/l_surname/w_initials/l_initials/
    tourney_name_norm/year/round_rank/is_rr."""
    result = MatchResult(unmatched_a=set(fused.index), unmatched_b=set(td.index))

    # frame de travail réduit aux seules colonnes d'appariement: le `fused`
    # reçu porte tout le schéma final (plusieurs dizaines de colonnes, blocs
    # fragmentés), le copier intégralement coûte des centaines de Mo et ralentit
    # chaque `.loc` de sous-groupe.
    fused = pd.DataFrame({
        "wn": fused["wn"].to_numpy(),
        "ln": fused["ln"].to_numpy(),
        "w_ginit": fused["w_ginit"].to_numpy(),
        "l_ginit": fused["l_ginit"].to_numpy(),
        "round": fused["round"].to_numpy(),
        "round_rank": fused["round_rank"].to_numpy(),
        "tourney_date": pd.to_datetime(fused["tourney_date"]).to_numpy(),
        "tourney_name_norm": fused["tourney_name"].map(nm.norm_tourney_name).to_numpy(),
    }, index=fused.index)
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
            # Deux NIVEAUX DE PREUVE, et le plateau prime sur le nom.
            #
            # Un plateau de joueurs partagé est une preuve directe: deux
            # tournois distincts ont des plateaux disjoints. Un alias de
            # sponsor n'est qu'une preuve indirecte, adossée à une table tenue
            # à la main qui vieillit à chaque renommage commercial.
            #
            # Sans cette hiérarchie, un alias erroné ne se contente pas de se
            # tromper: il fabrique un 2e candidat à 0.95 qui DÉTRUIT LA MARGE
            # du vrai tournoi (pourtant à 1.0) et fait rejeter le bon
            # rapprochement. C'est ce qui privait ~60 éditions de leurs cotes.
            # Le plateau n'est confronté que si les deux instances tombent la
            # même semaine: sans cette garde, deux étapes successives d'une
            # même tournée (plateaux très proches) pourraient se relier.
            tier = 0
            f_st = f_start.get(fkey)
            if td_start is not None and f_st is not None and abs((f_st - td_start).days) <= 10:
                ov = _player_overlap(fkey, tkey)
                if ov >= ROSTER_EVIDENCE_FLOOR:
                    tier, score = 1, max(score, ov)
                else:
                    score = max(score, ov)
            scored.append((tier, score, cand))
        scored.sort(reverse=True)
        if not scored:
            continue
        best_tier, best_score, best_name = scored[0]
        if len(scored) > 1:
            runner_tier, runner_score = scored[1][0], scored[1][1]
        else:
            runner_tier, runner_score = 0, 0.0
        # la marge ne se calcule qu'entre candidats de MÊME niveau de preuve:
        # un candidat "nom seul" ne peut pas opposer son veto à un candidat
        # attesté par le plateau.
        margin = best_score - runner_score if runner_tier == best_tier else 1.0
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

    # Colonnes extraites une fois en tableaux numpy, et accès par POSITION dans
    # la boucle d'appariement. Un `fused.loc[i]` y construit une Series de
    # plusieurs dizaines d'éléments à chaque candidat: sur ~1 M de couples
    # évalués, cet accès par label domine à lui seul le temps de consolidation
    # (plus de 10 minutes contre quelques dizaines de secondes ici).
    f_wn, f_ln = fused["wn"].to_numpy(), fused["ln"].to_numpy()
    f_wg, f_lg = fused["w_ginit"].to_numpy(), fused["l_ginit"].to_numpy()
    t_ws, t_ls = td["w_surname"].to_numpy(), td["l_surname"].to_numpy()
    t_wi, t_li = td["w_initials"].to_numpy(), td["l_initials"].to_numpy()
    f_pos = {lbl: p for p, lbl in enumerate(fused.index)}
    t_pos = {lbl: p for p, lbl in enumerate(td.index)}

    def pair_score(a, b):
        """a/b: positions (et non labels) dans `fused` / `td`."""
        ws, ls, wi, li = t_ws[b], t_ls[b], t_wi[b], t_li[b]
        wn, ln, wg, lg = f_wn[a], f_ln[a], f_wg[a], f_lg[a]
        w = nm.best_surname_score(wn, ws)
        if not nm.initials_compatible(wi, wg):
            w *= 0.4
        l = nm.best_surname_score(ln, ls)
        if not nm.initials_compatible(li, lg):
            l *= 0.4
        w_sw = nm.best_surname_score(wn, ls)
        if not nm.initials_compatible(li, wg):
            w_sw *= 0.4
        l_sw = nm.best_surname_score(ln, ws)
        if not nm.initials_compatible(wi, lg):
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
            pos_b = [(j, t_pos[j]) for j in idx_b]
            for i in idx_a:
                a = f_pos[i]
                for j, b in pos_b:
                    score, score_sw = pair_score(a, b)
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
