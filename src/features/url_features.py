"""
Extraction de features à partir d'une URL, pour la détection de phishing.

Usage:
    python -m src.features.url_features data/raw/urls.csv data/processed/url_features.csv

Le CSV d'entrée doit avoir une colonne "url" (et optionnellement "label").
"""

import re
import sys
import math
import socket
from urllib.parse import urlparse
from collections import Counter

import pandas as pd
import tldextract
import Levenshtein

# --- Listes de référence -----------------------------------------------

# Marques fréquemment usurpées en typosquatting (à étendre selon le domaine
# ciblé par ton projet — banque, ecommerce, etc.)
KNOWN_BRANDS = [
    "paypal", "google", "amazon", "microsoft", "apple", "facebook",
    "netflix", "instagram", "linkedin", "bankofamerica", "chase",
    "wellsfargo", "orange", "attijariwafabank", "cih", "bmce",
]

# TLDs statistiquement plus utilisés dans des campagnes de phishing
SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work"}

SUSPICIOUS_WORDS = [
    "verify", "secure", "account", "update", "login", "confirm",
    "banking", "signin", "ebayisapi", "webscr", "suspend", "urgent",
]


# --- Fonctions unitaires -------------------------------------------------

def url_length(url: str) -> int:
    return len(url)


def has_ip_address(url: str) -> int:
    """Détecte si le host est une IP brute plutôt qu'un nom de domaine
    (technique classique pour contourner la réputation d'un domaine)."""
    host = urlparse(url).netloc.split(":")[0]
    try:
        socket.inet_aton(host)
        return 1
    except OSError:
        return 0


def has_at_symbol(url: str) -> int:
    """Un '@' dans l'URL fait ignorer tout ce qui précède par le
    navigateur — souvent utilisé pour masquer le vrai domaine."""
    return int("@" in url)


def num_subdomains(url: str) -> int:
    ext = tldextract.extract(url)
    if not ext.subdomain:
        return 0
    return len(ext.subdomain.split("."))


def num_dots(url: str) -> int:
    return url.count(".")


def num_hyphens(url: str) -> int:
    return url.count("-")


def uses_https(url: str) -> int:
    return int(urlparse(url).scheme == "https")


def tld_is_suspicious(url: str) -> int:
    ext = tldextract.extract(url)
    return int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS)


def contains_suspicious_word(url: str) -> int:
    low = url.lower()
    return int(any(word in low for word in SUSPICIOUS_WORDS))


def domain_brand_similarity(url: str) -> float:
    """Distance de Levenshtein normalisée entre le domaine et la marque
    connue la plus proche — détecte le typosquatting
    (ex: 'paypa1-secure.com' proche de 'paypal').
    Retourne un score 0-1, 1 = identique à une marque connue."""
    ext = tldextract.extract(url)
    domain = ext.domain.lower()
    if not domain:
        return 0.0
    best = 0.0
    for brand in KNOWN_BRANDS:
        dist = Levenshtein.distance(domain, brand)
        max_len = max(len(domain), len(brand))
        similarity = 1 - dist / max_len if max_len else 0
        best = max(best, similarity)
    return round(best, 3)


def shannon_entropy(text: str) -> float:
    """Entropie de Shannon du domaine — les domaines générés
    automatiquement (DGA) ont une entropie plus élevée qu'un mot lisible."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return round(-sum((c / length) * math.log2(c / length) for c in counts.values()), 3)


def domain_entropy(url: str) -> float:
    ext = tldextract.extract(url)
    return shannon_entropy(ext.domain)


# --- Pipeline complet ------------------------------------------------

FEATURE_FUNCS = {
    "url_length": url_length,
    "has_ip_address": has_ip_address,
    "has_at_symbol": has_at_symbol,
    "num_subdomains": num_subdomains,
    "num_dots": num_dots,
    "num_hyphens": num_hyphens,
    "uses_https": uses_https,
    "tld_is_suspicious": tld_is_suspicious,
    "contains_suspicious_word": contains_suspicious_word,
    "domain_brand_similarity": domain_brand_similarity,
    "domain_entropy": domain_entropy,
}


def extract_features(url: str) -> dict:
    """Extrait toutes les features pour une seule URL.
    C'est CETTE fonction que l'API appellera pour le mode 'analyse d'une
    seule URL' du dashboard."""
    return {name: func(url) for name, func in FEATURE_FUNCS.items()}


def extract_features_batch(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """Applique extract_features sur tout un DataFrame (mode CSV batch)."""
    features = df[url_col].apply(extract_features).apply(pd.Series)
    return pd.concat([df, features], axis=1)


# --- Features alignées sur le dataset d'entraînement -----------------
# Ces fonctions reproduisent EXACTEMENT les colonnes du CSV fourni par
# l'utilisatrice (url_length, valid_url, at_symbol, ...), pour qu'une URL
# saisie en direct dans l'API produise les mêmes features que celles
# utilisées à l'entraînement (pas de "feature skew").

def _path_length(url: str) -> int:
    return len(urlparse(url).path)


def _nb_and(url: str) -> int:
    return url.count("&")


def _nb_or(url: str) -> int:
    return url.count("|")


def _nb_www(url: str) -> int:
    return url.lower().count("www")


def _nb_com(url: str) -> int:
    return url.lower().count("com")


def _nb_underscore(url: str) -> int:
    return url.count("_")


def _sensitive_words_count(url: str) -> int:
    """Nombre de mots sensibles présents dans l'URL (réutilise la même
    liste que contains_suspicious_word, mais en comptage plutôt qu'en
    booléen)."""
    low = url.lower()
    return sum(low.count(word) for word in SUSPICIOUS_WORDS)


def _valid_url(url: str) -> int:
    """HYPOTHÈSE (à confirmer avec la source du dataset) : une URL est
    considérée 'valide' si elle a un schéma ET un netloc correctement
    formés (ex: 'https://exemple.com' -> valide, 'exemple.com' ou
    'http:/exemple' -> non valide). À vérifier/ajuster si la définition
    réelle du dataset diffère."""
    parsed = urlparse(url)
    return int(bool(parsed.scheme) and bool(parsed.netloc) and "." in parsed.netloc)


DATASET_FEATURE_FUNCS = {
    "url_length": url_length,
    "valid_url": _valid_url,
    "at_symbol": has_at_symbol,
    "sensitive_words_count": _sensitive_words_count,
    "path_length": _path_length,
    "isHttps": uses_https,
    "nb_dots": num_dots,
    "nb_hyphens": num_hyphens,
    "nb_and": _nb_and,
    "nb_or": _nb_or,
    "nb_www": _nb_www,
    "nb_com": _nb_com,
    "nb_underscore": _nb_underscore,
}


def extract_dataset_features(url: str) -> dict:
    """Extrait les features dans le même format que le CSV d'entraînement
    (colonnes identiques, même ordre) — c'est CETTE fonction que l'API
    doit appeler pour scorer une URL en direct avec le modèle entraîné."""
    return {name: func(url) for name, func in DATASET_FEATURE_FUNCS.items()}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.features.url_features <input.csv> <output.csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(input_path)
    result = extract_features_batch(df)
    result.to_csv(output_path, index=False)
    print(f"{len(result)} URLs traitées -> {output_path}")
