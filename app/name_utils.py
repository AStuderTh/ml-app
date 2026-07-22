"""Normalisation de noms de joueurs partagée entre les modules de cotes
(app/odds.py, app/polymarket.py) pour matcher des noms écrits différemment
d'une source à l'autre (accents, casse, initiales vs prénom complet)."""
import re
import unicodedata


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z\s]", "", name.lower())
    return " ".join(name.split())
