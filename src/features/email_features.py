"""
Extraction de features à partir du sujet + corps d'un email, pour la
détection de phishing.

On travaille volontairement sur (subject, body) uniquement — pas sur
l'email brut avec headers — car les sources combinées (Enron, Ling, CEAS,
Nazario, Nigerian Fraud, SpamAssassin) n'ont pas toutes des headers
fiables. Utiliser uniquement subject+body évite un biais de fuite de
données où le modèle apprendrait "a des headers = phishing" au lieu du
vrai signal de contenu.

Usage:
    python -m src.features.email_features data/raw/emails.csv data/processed/email_features.csv

Le CSV d'entrée doit avoir des colonnes "subject" et "body"
(et optionnellement "label").
"""

import re
import sys

import pandas as pd

from .url_features import extract_features as extract_url_features

URGENT_WORDS = [
    "urgent", "immediately", "action required", "suspended", "verify now",
    "expire", "limited time", "click here", "confirm your account",
    "password", "locked", "unusual activity",
]

URL_REGEX = re.compile(r"https?://[^\s\"'<>]+")
HTML_TAG_REGEX = re.compile(r"<[^>]+>")


def _clean(text) -> str:
    """Gère les NaN/None proprement (fréquent dans les corpus combinés)."""
    if pd.isna(text):
        return ""
    return str(text)


# --- Features sur le corps ------------------------------------------

def body_length(body: str) -> int:
    return len(body)


def num_words(body: str) -> int:
    return len(body.split())


def num_urls_in_body(body: str) -> int:
    return len(URL_REGEX.findall(body))


def max_url_suspicion_score(body: str) -> float:
    """Réutilise url_features : pour chaque URL trouvée dans le corps,
    calcule sa similarité à une marque connue (typosquat) et garde le max."""
    urls = URL_REGEX.findall(body)
    if not urls:
        return 0.0
    return max(extract_url_features(u)["domain_brand_similarity"] for u in urls)


def has_html_tags(body: str) -> int:
    """Les emails de phishing sont plus souvent formatés en HTML
    (boutons, logos imités) que du texte brut."""
    return int(bool(HTML_TAG_REGEX.search(body)))


def num_exclamation_marks(text: str) -> int:
    return text.count("!")


def uppercase_ratio(text: str) -> float:
    """Proportion de lettres majuscules — le 'shouting' (COMPTE BLOQUÉ)
    est un signal classique de spam/phishing."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return round(upper / len(letters), 3)


def num_urgent_words(text: str) -> int:
    low = text.lower()
    return sum(low.count(word) for word in URGENT_WORDS)


# --- Pipeline complet ------------------------------------------------

def extract_features(subject, body) -> dict:
    """Extrait toutes les features à partir du sujet et du corps.
    C'est CETTE fonction que l'API appellera pour le mode 'analyse d'un
    seul email' du dashboard."""
    subject = _clean(subject)
    body = _clean(body)
    full_text = f"{subject} {body}"

    return {
        "subject_length": len(subject),
        "body_length": body_length(body),
        "num_words": num_words(body),
        "num_urls_in_body": num_urls_in_body(body),
        "max_url_suspicion_score": max_url_suspicion_score(body),
        "has_html_tags": has_html_tags(body),
        "num_exclamation_marks": num_exclamation_marks(full_text),
        "subject_uppercase_ratio": uppercase_ratio(subject),
        "num_urgent_words": num_urgent_words(full_text),
    }


def extract_features_batch(
    df: pd.DataFrame, subject_col: str = "subject", body_col: str = "body"
) -> pd.DataFrame:
    features = df.apply(
        lambda row: extract_features(row.get(subject_col), row.get(body_col)),
        axis=1,
    ).apply(pd.Series)
    return pd.concat([df, features], axis=1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.features.email_features <input.csv> <output.csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(input_path)
    result = extract_features_batch(df)
    result.to_csv(output_path, index=False)
    print(f"{len(result)} emails traités -> {output_path}")
