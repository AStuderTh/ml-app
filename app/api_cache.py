"""Cache disque partagé pour les appels aux APIs de cotes externes.

Contrairement à st.cache_data (perdu à chaque redémarrage du serveur
Streamlit, fréquent en dev et à chaque déploiement), ce cache persiste sur
disque sous data/cache/ — utile en particulier pour odds-api.io, dont le
plan gratuit limite le nombre de requêtes, pour ne pas retaper l'API à
chaque rechargement de page tant que le TTL n'est pas expiré."""
import hashlib
import json
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def cached_call(key: str, ttl_seconds: int, fetch_fn):
    """Renvoie fetch_fn() en le mettant en cache disque sous `key` pendant
    `ttl_seconds`. Si `fetch_fn` échoue (réseau, rate limit) et qu'une entrée
    expirée existe encore sur disque, elle est renvoyée en secours plutôt que
    de propager l'erreur."""
    path = _cache_path(key)
    now = time.time()

    cached = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            cached = None

    if cached is not None and now - cached["ts"] < ttl_seconds:
        return cached["value"]

    try:
        value = fetch_fn()
    except Exception:
        if cached is not None:
            return cached["value"]
        raise

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": now, "value": value}, f)
    except OSError:
        pass
    return value
