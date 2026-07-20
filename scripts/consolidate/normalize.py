"""Normalisation helpers shared by all loaders/matchers: noms de joueurs,
noms de tournois, rounds, tourney_id. Toute la logique de "comment comparer
deux enregistrements qui ne se ressemblent pas exactement" vit ici.
"""
import re
import unicodedata

PARTICLES = {"de", "del", "della", "da", "van", "von", "der", "den", "dos",
             "di", "le", "la", "bin", "al", "st", "mac", "mc", "o"}

GEN_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

FILLER_TOURNEY_WORDS = {
    "international", "tournament", "atp", "presented", "by", "the", "of",
}
# NB: volontairement PAS "open"/"championships"/"cup": ces mots distinguent
# souvent des tournois bien différents joués la même semaine/ville (ex.
# "Australian Open" vs "Australian Hardcourt Championships").


def strip_accents(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_name_full(name: str) -> str:
    """Normalise un nom complet ('Roger Federer') pour comparaison exacte."""
    if name is None:
        return ""
    s = strip_accents(str(name)).lower()
    s = s.replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    tokens = [t for t in s.split() if t not in GEN_SUFFIXES]
    return " ".join(tokens)


def surname_guess(full_name_norm: str) -> str:
    """Devine le nom de famille à partir d'un nom complet déjà normalisé,
    en tenant compte des particules ('del potro', 'van de zandschulp')."""
    tokens = full_name_norm.split()
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    # remonte depuis la fin tant que le token précédent est une particule
    i = len(tokens) - 1
    while i > 0 and tokens[i - 1] in PARTICLES:
        i -= 1
    return " ".join(tokens[i:])


def given_initials(full_name_norm: str, surname: str = None) -> str:
    """Initiale du prénom. On ne garde que la 1re lettre du 1er token: c'est
    la seule partie non ambiguë (le nombre de tokens de nom de famille est
    inconnu sans base de données de joueurs, cf. surname_candidates)."""
    tokens = full_name_norm.split()
    return tokens[0][0] if tokens else ""


def surname_candidates(full_name_norm: str):
    """Plusieurs hypothèses de nom de famille pour un nom complet normalisé,
    des plus courtes (1 mot) aux plus longues (3 mots), en plus de la
    supposition à base de particules. Nécessaire car un nom de famille
    composé sans particule (ex: espagnol 'Ramos Vinolas', 'Bautista Agut')
    ne peut pas être détecté de façon fiable autrement, et une source peut
    le tronquer ('Ramos') quand une autre le garde entier."""
    tokens = full_name_norm.split()
    if not tokens:
        return [""]
    cands = {surname_guess(full_name_norm)}
    for k in (1, 2, 3):
        if len(tokens) >= k:
            cands.add(" ".join(tokens[-k:]))
    return list(cands)


def best_surname_score(name_a_norm: str, name_b_norm: str) -> float:
    """Meilleur score de compatibilité entre 2 noms complets normalisés, en
    essayant toutes les hypothèses de longueur de nom de famille des deux
    côtés (cf. surname_candidates)."""
    best = 0.0
    for ca in surname_candidates(name_a_norm):
        for cb in surname_candidates(name_b_norm):
            s = names_compatible(ca, cb)
            if s > best:
                best = s
    return best


def parse_abbrev_name(name: str):
    """Parse un nom façon tennis-data.co: 'Del Potro J.M.' -> (surname, initials).
    Le dernier token (ou les 2 derniers si suffixe court) porte les initiales
    au format 'X.' ou 'X.Y.'."""
    if name is None:
        return "", ""
    raw = str(name).strip()
    s = strip_accents(raw).lower().replace("-", " ")
    s = re.sub(r"[^a-z. ]", " ", s)
    tokens = [t for t in s.split() if t]
    if not tokens:
        return "", ""
    last = tokens[-1]
    if "." in last:
        initials = "".join(ch for ch in last if ch.isalpha())
        surname_tokens = tokens[:-1]
    else:
        # pas de point détecté (rare / erreur de saisie) -> tout est surname
        initials = ""
        surname_tokens = tokens
    surname = " ".join(surname_tokens).replace(".", "").strip()
    return surname, initials


def names_compatible(surname_a: str, surname_b: str, fuzzy: bool = True) -> float:
    """Score [0,1] de compatibilité entre deux noms de famille normalisés,
    tolérant les troncatures de noms composés (ex: 'ramos' vs 'ramos vinolas')."""
    if not surname_a or not surname_b:
        return 0.0
    if surname_a == surname_b:
        return 1.0
    a_tokens, b_tokens = surname_a.split(), surname_b.split()
    if a_tokens[0] == b_tokens[0]:
        # même premier mot du nom composé -> quasi certain (ex: Ramos / Ramos-Vinolas)
        return 0.9
    if surname_a.startswith(surname_b) or surname_b.startswith(surname_a):
        return 0.85
    if not fuzzy:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, surname_a, surname_b).ratio()


def initials_compatible(initials: str, given_init: str) -> bool:
    """'initials' (ex: 'jm') vient d'un format abrégé, 'given_init' (ex: 'j')
    vient d'un nom complet. Compatible si le 1er caractère concorde (le nom
    complet peut omettre un 2e prénom que l'abrégé référence)."""
    if not initials or not given_init:
        return True  # information manquante: ne pas bloquer, laisser le nom trancher
    return initials[0] == given_init[0]


NORM_TID_RE = re.compile(r"^(\d{4})-0*(\d+)$")


def norm_tourney_id(tid: str) -> str:
    """Aligne les IDs Sackmann ('2023-0891') et TML ('2023-891') qui ne
    diffèrent que par le padding de zéros."""
    s = str(tid)
    m = NORM_TID_RE.match(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s


# Tournois ATP dont le nom sponsor change selon les années/sources, alors
# qu'il s'agit du même événement (même ville/même place dans le calendrier).
# Clé = nom canonique, valeurs = fragments (après normalisation) qui doivent
# tous être reconnus comme cet événement.
TOURNEY_ALIASES = {
    "indian wells": ["indian wells", "bnp paribas"],
    "miami": ["miami", "sony ericsson", "nasdaq 100", "lipton"],
    "cincinnati": ["cincinnati", "western southern", "western south"],
    "basel": ["basel", "swiss indoors"],
    "paris masters": ["paris masters", "rolex paris masters", "bnp paribas masters", "bercy"],
    "vienna": ["vienna", "erste bank"],
    "shanghai masters": ["shanghai masters", "shanghai rolex masters"],
    "madrid masters": ["madrid masters", "mutua madrid", "madrid open"],
    "rome masters": ["rome masters", "internazionali", "italian open", "foro italico", "italia"],
    "monte carlo masters": ["monte carlo"],
    "canada masters": ["canada masters", "rogers cup", "canadian open", "national bank open"],
    "queens club": ["queens club", "fever tree", "cinch championships", "stella artois", "aegon championships"],
    "halle": ["halle", "gerry weber", "noventi open", "terra wortmann"],
    "acapulco": ["acapulco", "mexican open", "abierto mexicano"],
    "dubai": ["dubai"],
    "doha": ["doha", "qatar", "exxonmobil"],
    "washington": ["washington", "citi open", "legg mason"],
    "beijing": ["beijing", "china open"],
    "tokyo": ["tokyo", "japan open", "rakuten"],
    "auckland": ["auckland", "asb classic"],
    # NB: "Heineken Open"/"Heineken Trophy" volontairement PAS mis en alias
    # d'Auckland: ce sponsoring a aussi été utilisé pour d'autres étapes du
    # circuit (ex. Shanghai), un marqueur seulement basé sur le sponsor
    # créerait un faux rapprochement entre 2 villes différentes.
    "sydney": ["sydney"],
    "brisbane": ["brisbane"],
    "adelaide": ["adelaide"],
    "buenos aires": ["buenos aires", "argentina open"],
    "rio de janeiro": ["rio open", "rio de janeiro"],
    "hamburg": ["hamburg"],
    "stuttgart": ["stuttgart", "mercedes cup"],
    "geneva": ["geneva"],
    "estoril": ["estoril"],
    "barcelona": ["barcelona", "godo", "conde de godo"],
    "munich": ["munich", "bmw open"],
    "eastbourne": ["eastbourne", "devonshire"],
    "newport": ["newport", "hall of fame"],
    "atlanta": ["atlanta", "bb t"],
    "los cabos": ["los cabos", "abierto los cabos"],
    "winston salem": ["winston salem"],
    "metz": ["metz", "moselle"],
    "antwerp": ["antwerp", "european open"],
    "stockholm": ["stockholm", "if stockholm"],
    "st petersburg": ["st petersburg", "petersburg"],
    "moscow": ["moscow", "kremlin"],
    "marseille": ["marseille", "open 13"],
    "rotterdam": ["rotterdam", "abn amro"],
    "montpellier": ["montpellier", "sud de france"],
    "delray beach": ["delray beach"],
    "memphis": ["memphis", "us national indoor"],
    "san jose": ["san jose"],
    "los angeles": ["los angeles"],
    "bastad": ["bastad", "swedish open"],
    "gstaad": ["gstaad", "swiss open"],
    "umag": ["umag", "croatia open"],
    "kitzbuhel": ["kitzbuhel", "austrian open", "generali open"],
    "cordoba": ["cordoba open"],
    "santiago": ["santiago", "chile open"],
    "houston": ["houston", "us men s clay"],
}


def canonicalize_tourney(name_norm: str) -> str:
    """Reconnaît un alias sponsor connu par inclusion de MOTS entiers (pas de
    sous-chaîne brute, pour éviter qu'un marqueur générique ('open') matche
    à l'intérieur d'un autre mot)."""
    tokens = set(name_norm.split())
    for canonical, markers in TOURNEY_ALIASES.items():
        for marker in markers:
            marker_tokens = marker.split()
            if all(t in tokens for t in marker_tokens):
                return canonical
    return name_norm


def norm_tourney_name(name: str) -> str:
    if name is None:
        return ""
    s = strip_accents(str(name)).lower()
    s = s.replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in FILLER_TOURNEY_WORDS]
    return " ".join(tokens)


def tourney_name_similarity(a: str, b: str) -> float:
    """Similarité conservatrice: exige un bon accord à la fois au niveau des
    mots (Jaccard) et des caractères (ratio), pour éviter qu'un nom compris
    dans l'autre (ex: 'australian' substring de 'australian hardcourt')
    gonfle artificiellement le score façon SequenceMatcher seul."""
    na, nb = norm_tourney_name(a), norm_tourney_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ca, cb = canonicalize_tourney(na), canonicalize_tourney(nb)
    if ca == cb:
        return 0.95
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    import difflib
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return min(jaccard, ratio)


# --- rounds -----------------------------------------------------------
# Rang depuis la finale (0 = finale) : permet d'aligner les codes Sackmann/TML
# ('F','SF','QF','R16',...) avec les libellés tennis-data.co ('The Final',
# 'Semifinals', '1st Round', ...) sans dépendre de la taille du tableau.
ATP_ROUND_RANK = {"F": 0, "BR": 0, "SF": 1, "QF": 2, "R16": 3, "R32": 4,
                  "R64": 5, "R128": 6}
ATP_ROUND_SPECIAL = {"RR"}  # round robin: pas d'ordre, géré par appariement de noms

TD_ROUND_LABELS_IN_ORDER = [
    "The Final", "Semifinals", "Quarterfinals",
    "4th Round", "3rd Round", "2nd Round", "1st Round",
]
TD_ROUND_RANK = {label: i for i, label in enumerate(TD_ROUND_LABELS_IN_ORDER)}
TD_ROUND_SPECIAL = {"Round Robin"}


def atp_round_rank(round_code: str):
    return ATP_ROUND_RANK.get(str(round_code))


def td_round_rank(round_label: str):
    return TD_ROUND_RANK.get(str(round_label))


def is_special_round(round_code: str) -> bool:
    return round_code in ATP_ROUND_SPECIAL or round_code in TD_ROUND_SPECIAL
