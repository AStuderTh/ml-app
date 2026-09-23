"""Stats 'en direct' pour un match pas encore joué: Elo/forme/H2H courants
des 2 joueurs, et construction du vecteur de features attendu par un modèle
sauvegardé pour en tirer une probabilité de victoire.

Nuance importante: selon la source d'origine d'un match dans data/tennis.db,
un même joueur peut apparaître sous plusieurs formats de nom (ex: 'Alex
Molcan' pour les matchs Sackmann/TML, mais 'Molcan A.' pour ceux connus
seulement via tennis-data.co — cf. scripts/consolidate/normalize.py). Pour
ne pas fausser l'Elo/la forme en ignorant une partie de l'historique d'un
joueur, ce module regroupe toutes les variantes de nom (même nom de famille
normalisé + même initiale) sous un nom canonique avant de calculer l'état
courant. Ce regroupement est fait uniquement pour CET affichage — il ne
touche pas au calcul utilisé pour l'entraînement des modèles
(app/features.py), qui reste inchangé, pour ne pas modifier rétroactivement
le comportement des modèles déjà sauvegardés."""
import os
import sys
from collections import defaultdict, deque

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.features import (DEFAULT_ELO, TOUR_AVG_RETURN, TOUR_AVG_SERVE, _adjust,
                           _RateTracker, elo_decay)
from scripts.consolidate.normalize import given_initials, norm_name_full, parse_abbrev_name, surname_guess

FEATURE_LABELS = {
    "elo_diff": "Écart Elo global",
    "elo_surface_diff": "Écart Elo surface",
    "elo_diff_recent": "Écart Elo global (pondéré récence)",
    "elo_surface_diff_recent": "Écart Elo surface (pondéré récence)",
    "rank_diff": "Écart classement ATP",
    "rank_points_diff": "Écart points ATP",
    "age_diff": "Écart d'âge",
    "ht_diff": "Écart de taille",
    "form_diff": "Écart de forme (10 derniers matchs)",
    "h2h_diff": "Écart face-à-face",
    "hand_advantage": "Avantage main gauche",
    "implied_prob_p1": "Probabilité implicite (cotes)",
    "experience_diff": "Écart d'expérience (matchs joués)",
    "surface_clay": "Sur terre battue",
    "surface_grass": "Sur gazon",
    "surface_hard": "Sur dur",
    "points_dominance_diff": "Écart de domination aux points (service + retour)",
    "points_dominance_surface_diff": "Écart de domination aux points (par surface)",
    "rest_days_diff": "Écart de jours de repos",
    "minutes_14d_diff": "Écart de minutes jouées (14 j)",
    "matches_14d_diff": "Écart de matchs joués (14 j)",
    "long_match_prev_diff": "Match précédent long (> 3 h)",
    "is_indoor": "Match en salle",
    "level_grandslam": "Grand Chelem",
    "level_masters": "Masters 1000",
    "level_finals": "Masters de fin d'année",
    "best_of_5": "Format en 5 sets",
}


def _name_key(name: str):
    """(nom de famille normalisé, initiale) — détecte automatiquement le
    format du nom: 'Prénom Nom' complet (Sackmann/TML) vs 'Nom I.' abrégé
    façon tennis-data.co (ces deux formats ont un ordre des mots opposé,
    donc pas le même parseur — cf. scripts/consolidate/normalize.py)."""
    raw = str(name).strip()
    tokens = raw.split()
    last_token = tokens[-1] if tokens else ""
    # format abrégé ('Molcan A.', 'Del Potro J.M.'): le dernier token est une
    # courte suite d'initiales avec points. On exige <=3 lettres pour éviter
    # de traiter un prénom composé complet comme 'J.J. Wolf' (initiales
    # seulement dans le PREMIER token, ici) comme un nom abrégé.
    if "." in last_token and len(last_token.replace(".", "")) <= 3:
        surname, initials = parse_abbrev_name(raw)
        return surname, initials[:1]
    n = norm_name_full(raw)
    return surname_guess(n), given_initials(n)


def _canonical_name_maps(names: pd.Series):
    """Regroupe les variantes de nom par (nom de famille normalisé,
    initiale). Retourne (variante -> canonique, (surname, initiale) -> canonique)."""
    groups = defaultdict(list)
    for name in names.dropna().unique():
        key = _name_key(name)
        if not key[0]:
            continue
        groups[key].append(name)

    variant_to_canon, key_to_canon = {}, {}
    for key, variants in groups.items():
        # préfère la variante la plus longue sans point (nom complet plutôt
        # que 'Surname X.' abrégé)
        best = max(variants, key=lambda v: ("." not in str(v), len(str(v))))
        key_to_canon[key] = best
        for v in variants:
            variant_to_canon[v] = best
    return variant_to_canon, key_to_canon


@st.cache_data(show_spinner="Calcul des stats courantes des joueurs...")
def compute_current_player_state(matches: pd.DataFrame, elo_k: float = 32.0, form_window: int = 10) -> dict:
    """État final (Elo, Elo surface, forme, H2H, expérience, dernier
    classement/âge/taille/main connus) de chaque joueur après tout
    l'historique disponible — c'est-à-dire son état 'actuel'."""
    df = matches.sort_values("tourney_date")
    variant_to_canon, key_to_canon = _canonical_name_maps(pd.concat([df["winner_name"], df["loser_name"]]))

    elo = defaultdict(lambda: DEFAULT_ELO)
    elo_surf = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
    recent = defaultdict(lambda: deque(maxlen=form_window))
    h2h = defaultdict(lambda: defaultdict(int))
    played = defaultdict(int)
    last_seen = {}
    last_seen_surface = defaultdict(dict)  # surface -> {joueur: date}

    # Domination aux points (service/retour): mêmes accumulateurs bayésiens
    # décroissants que app.features.compute_chronological_features (réutilisés
    # via _RateTracker/_adjust pour rester cohérents avec l'entraînement), mais
    # ici on ne garde que l'ÉTAT FINAL (comme pour l'Elo ci-dessus) et non une
    # valeur pré-match par ligne.
    srv = _RateTracker(TOUR_AVG_SERVE)
    ret = _RateTracker(TOUR_AVG_RETURN)
    srv_surf = defaultdict(lambda: _RateTracker(TOUR_AVG_SERVE))
    ret_surf = defaultdict(lambda: _RateTracker(TOUR_AVG_RETURN))

    cols = ["winner_name", "loser_name", "surface",
            "winner_rank", "winner_rank_points", "winner_age", "winner_ht", "winner_hand",
            "loser_rank", "loser_rank_points", "loser_age", "loser_ht", "loser_hand",
            "w_svpt", "l_svpt", "w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon",
            "tourney_date"]
    for row in df[cols].itertuples(index=False):
        w = variant_to_canon.get(row.winner_name)
        l = variant_to_canon.get(row.loser_name)
        if w is None or l is None:
            continue
        surf = row.surface or "Hard"
        ew, el = elo[w], elo[l]
        esw, esl = elo_surf[surf][w], elo_surf[surf][l]

        exp_w = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
        elo[w] = ew + elo_k * (1 - exp_w)
        elo[l] = el + elo_k * (0 - (1 - exp_w))
        exp_sw = 1.0 / (1.0 + 10 ** ((esl - esw) / 400.0))
        elo_surf[surf][w] = esw + elo_k * (1 - exp_sw)
        elo_surf[surf][l] = esl + elo_k * (0 - (1 - exp_sw))

        recent[w].append(1)
        recent[l].append(0)
        h2h[w][l] += 1
        played[w] += 1
        played[l] += 1

        # rang/points: toujours pris sur la ligne la plus récente (variable dans
        # le temps). âge/taille/main: source tennis-data.co seule ne les fournit
        # pas — on ne les écrase donc que si la nouvelle valeur est renseignée,
        # pour ne pas perdre une info connue via une ligne plus ancienne mieux
        # renseignée (Sackmann/TML).
        for name, rank, pts, age, ht, hand in (
            (w, row.winner_rank, row.winner_rank_points, row.winner_age, row.winner_ht, row.winner_hand),
            (l, row.loser_rank, row.loser_rank_points, row.loser_age, row.loser_ht, row.loser_hand),
        ):
            prev = last_seen.get(name, {})
            last_seen[name] = dict(
                rank=rank, rank_points=pts, date=row.tourney_date,
                age=age if pd.notna(age) else prev.get("age"),
                ht=ht if pd.notna(ht) else prev.get("ht"),
                hand=hand if pd.notna(hand) else prev.get("hand"),
            )
            last_seen_surface[surf][name] = row.tourney_date

        # points de service/retour: seuls certains matchs (source Sackmann/TML)
        # en disposent — cf. app.features.compute_chronological_features, même
        # garde-fou (wp/lp > 0, ww/lw non NaN).
        wp, lp = row.w_svpt, row.l_svpt
        ww = row.w_1stWon + row.w_2ndWon if pd.notna(row.w_1stWon) and pd.notna(row.w_2ndWon) else float("nan")
        lw = row.l_1stWon + row.l_2ndWon if pd.notna(row.l_1stWon) and pd.notna(row.l_2ndWon) else float("nan")
        if pd.notna(wp) and pd.notna(lp) and wp > 0 and lp > 0 and pd.notna(ww) and pd.notna(lw):
            srv.update(w, ww, wp)
            srv.update(l, lw, lp)
            ret.update(w, lp - lw, lp)
            ret.update(l, wp - ww, wp)
            srv_surf[surf].update(w, ww, wp)
            srv_surf[surf].update(l, lw, lp)
            ret_surf[surf].update(w, lp - lw, lp)
            ret_surf[surf].update(l, wp - ww, wp)

    return {
        "elo": dict(elo),
        "elo_surf": {s: dict(d) for s, d in elo_surf.items()},
        "form": {k: (sum(v) / len(v) if v else 0.5) for k, v in recent.items()},
        "played": dict(played),
        "h2h": {k: dict(v) for k, v in h2h.items()},
        "last_seen": last_seen,
        "last_seen_surface": dict(last_seen_surface),
        "key_to_canon": key_to_canon,
        "srv": srv,
        "ret": ret,
        "srv_surf": dict(srv_surf),
        "ret_surf": dict(ret_surf),
    }


def resolve_player(query_name: str, state: dict) -> str | None:
    """Retrouve le nom canonique d'un joueur dans l'historique à partir d'un
    nom externe (ex: format ESPN 'Prénom Nom')."""
    key = _name_key(query_name)
    if not key[0]:
        return None
    return state["key_to_canon"].get(key)


def _num_diff(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def player_snapshot(state: dict, canon_name: str, surface: str, as_of=None) -> dict:
    """as_of: date par rapport à laquelle on mesure l'inactivité (par défaut
    aujourd'hui) — sert à calculer elo_recent/elo_surface_recent (régression
    vers la moyenne selon le temps écoulé depuis le dernier match connu du
    joueur, globalement et sur cette surface précise)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
    ls = state["last_seen"].get(canon_name, {})
    elo = state["elo"].get(canon_name, DEFAULT_ELO)
    elo_surface = state["elo_surf"].get(surface or "Hard", {}).get(canon_name, DEFAULT_ELO)
    last_date = ls.get("date")
    last_surf_date = state.get("last_seen_surface", {}).get(surface or "Hard", {}).get(canon_name)
    days_inactive = (as_of - last_date).days if pd.notna(last_date) else None
    days_inactive_surf = (as_of - last_surf_date).days if pd.notna(last_surf_date) else None
    srv_surf_t = state.get("srv_surf", {}).get(surface or "Hard")
    ret_surf_t = state.get("ret_surf", {}).get(surface or "Hard")
    return {
        "elo": elo,
        "elo_surface": elo_surface,
        "elo_recent": elo_decay(elo, days_inactive),
        "elo_surface_recent": elo_decay(elo_surface, days_inactive_surf),
        "form": state["form"].get(canon_name, 0.5),
        "played": state["played"].get(canon_name, 0),
        "rank": ls.get("rank"), "rank_points": ls.get("rank_points"),
        "age": ls.get("age"), "ht": ls.get("ht"), "hand": ls.get("hand"),
        "last_match_date": ls.get("date"),
        "srv_raw": state["srv"].estimate(canon_name) if "srv" in state else None,
        "ret_raw": state["ret"].estimate(canon_name) if "ret" in state else None,
        "srv_surf_raw": srv_surf_t.estimate(canon_name) if srv_surf_t else None,
        "ret_surf_raw": ret_surf_t.estimate(canon_name) if ret_surf_t else None,
    }


def build_live_features(snap1: dict, snap2: dict, state: dict, canon1: str, canon2: str,
                         feature_names: list, surface: str, implied_prob_p1: float = None,
                         best_of: float = None, indoor: str = None) -> tuple[dict, list]:
    """Construit les valeurs de features pour p1 vs p2. Retourne (valeurs,
    features manquantes — le modèle ne pourra pas être utilisé si non vide).

    best_of/indoor: dernières valeurs connues du tournoi (deviné par
    app.upcoming._guess_tourney_context à partir de l'historique du même
    tournoi) — ne dépendent pas du joueur, contrairement au reste des
    features."""
    h2h1 = state["h2h"].get(canon1, {}).get(canon2, 0)
    h2h2 = state["h2h"].get(canon2, {}).get(canon1, 0)

    bo_known = best_of is not None and pd.notna(best_of)
    indoor_str = str(indoor).upper() if (indoor is not None and pd.notna(indoor)) else None

    # Domination aux points: même ajustement à la force de l'adversaire qu'à
    # l'entraînement (cf. app.features._adjust) — None tant que l'un des deux
    # joueurs n'a pas assez de points observés (MIN_SERVE_POINTS).
    def _dominance_diff(srv1, ret1, srv2, ret2):
        dom1 = _adjust(srv1, ret2, TOUR_AVG_RETURN)
        dom2 = _adjust(srv2, ret1, TOUR_AVG_RETURN)
        if dom1 is None or dom2 is None:
            return None
        return dom1 - dom2

    points_dominance_diff = _dominance_diff(
        snap1.get("srv_raw"), snap1.get("ret_raw"), snap2.get("srv_raw"), snap2.get("ret_raw"))
    points_dominance_surface_diff = _dominance_diff(
        snap1.get("srv_surf_raw"), snap1.get("ret_surf_raw"),
        snap2.get("srv_surf_raw"), snap2.get("ret_surf_raw"))

    computed = {
        "elo_diff": snap1["elo"] - snap2["elo"],
        "elo_surface_diff": snap1["elo_surface"] - snap2["elo_surface"],
        "elo_diff_recent": snap1["elo_recent"] - snap2["elo_recent"],
        "elo_surface_diff_recent": snap1["elo_surface_recent"] - snap2["elo_surface_recent"],
        "rank_diff": _num_diff(snap2["rank"], snap1["rank"]),  # positif => p1 mieux classé
        "rank_points_diff": _num_diff(snap1["rank_points"], snap2["rank_points"]),
        "age_diff": _num_diff(snap1["age"], snap2["age"]),
        "ht_diff": _num_diff(snap1["ht"], snap2["ht"]),
        "form_diff": snap1["form"] - snap2["form"],
        "h2h_diff": float(h2h1 - h2h2),
        "hand_advantage": (1 if snap1["hand"] == "L" else 0) - (1 if snap2["hand"] == "L" else 0),
        "experience_diff": float(snap1["played"] - snap2["played"]),
        "implied_prob_p1": implied_prob_p1,
        "surface_clay": 1.0 if surface == "Clay" else 0.0,
        "surface_grass": 1.0 if surface == "Grass" else 0.0,
        "surface_hard": 1.0 if surface == "Hard" else 0.0,
        "best_of_5": (1.0 if float(best_of) == 5 else 0.0) if bo_known else None,
        "is_indoor": (1.0 if indoor_str == "I" else 0.0) if indoor_str in ("I", "O") else None,
        "points_dominance_diff": points_dominance_diff,
        "points_dominance_surface_diff": points_dominance_surface_diff,
    }

    values = {f: computed.get(f) for f in feature_names}
    missing = [f for f, v in values.items() if v is None]
    return values, missing
