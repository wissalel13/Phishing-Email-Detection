"""Modèle ML figé (notebook 03/04) : normalisation, TF-IDF, régression logistique.

La normalisation et le retrait du préfixe `Subject:` sont ceux du notebook 05 ;
ne pas les modifier sans réentraîner le modèle.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
NUM_NAMES = ["body_length", "num_words", "num_urgent_words"]
_URL_OR_ADDRESS = re.compile(r"\S*(?:://|www\.|@)\S*")
_SUBJECT_PREFIX = re.compile(r"^Subject:\s*")


def normalize_like_corpus(text) -> str:
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t).lower().replace("'", "").replace("’", "")
    t = _URL_OR_ADDRESS.sub(lambda m: re.sub(r"[^a-z0-9_]", "", m.group(0)), t)
    t = re.sub(r"[^a-z0-9_\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class MLDecision:
    p_ml: float          # probabilité de phishing
    is_phishing: bool    # p_ml >= seuil
    in_band: bool        # dans la zone d'incertitude


class EmailML:
    def __init__(self, models_dir: str | Path | None = None):
        d = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.bundle = joblib.load(d / "email_model.joblib")
        self.vectorizer = joblib.load(d / "emails_tfidf_vectorizer.joblib")
        self.threshold = float(self.bundle["threshold"])
        self.t_low, self.t_high = (float(v) for v in self.bundle["uncertainty_band"])
        self.keep_cols = np.asarray(self.bundle["keep_cols"], dtype=int)
        self.uses_numeric = bool(self.bundle["numeric_features"])

    def _matrix(self, clean_texts: list[str]) -> sp.csr_matrix:
        x_tfidf = self.vectorizer.transform(clean_texts)
        if self.uses_numeric:
            import pandas as pd
            from src.features.email_features import extract_features_batch
            feats = extract_features_batch(pd.DataFrame({"subject": "", "body": clean_texts}))[NUM_NAMES]
            num = sp.csr_matrix(np.log1p(feats.values.astype(float)))
        else:
            num = sp.csr_matrix((x_tfidf.shape[0], len(NUM_NAMES)))
        return sp.hstack([x_tfidf, num]).tocsr()

    def predict_proba(self, texts) -> np.ndarray:
        """texts : emails bruts (préfixe `Subject:` accepté)."""
        clean = [normalize_like_corpus(_SUBJECT_PREFIX.sub("", str(t))) for t in texts]
        x = self._matrix(clean)[:, self.keep_cols]
        return sum(w * m.predict_proba(x)[:, 1] for m, w in zip(self.bundle["models"], self.bundle["weights"]))

    def decide(self, text: str) -> MLDecision:
        p = float(self.predict_proba([text])[0])
        return MLDecision(p_ml=p, is_phishing=p >= self.threshold, in_band=self.t_low <= p <= self.t_high)
