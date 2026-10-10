"""Point d'entrée unique pour analyser un email (ML figé + LLM Ollama).

Stratégie gelée par le notebook 05 : `llm_seul` (le LLM décide pour tous les emails).
Le ML est toujours calculé (quelques ms) : il sert de repli et reste visible dans le résultat.
Si le LLM est indisponible ou illisible, le verdict vient du ML et `mode` vaut `ml_degrade` :
l'appelant DOIT l'afficher, car le ML seul avait 60 % de faux positifs sur le jeu difficile.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.llm.client import DEFAULT_URL, OllamaClient, OllamaError
from src.llm.parse import parse_answer
from src.llm.prompts import DEFAULT_MAX_CHARS, build_prompt
from src.ml.email_ml import DEFAULT_MODELS_DIR, EmailML

SUPPORTED_STRATEGIES = ("llm_seul", "ml_seul")


@dataclass(frozen=True)
class EmailVerdict:
    verdict: str                  # "phishing" | "legitime"
    is_phishing: bool
    score: float                  # score de phishing 0..1 (celui du LLM, ou p_ml en mode dégradé)
    mode: str                     # "llm" | "ml_degrade" | "ml_seul"
    reason: str
    p_ml: float
    ml_is_phishing: bool
    in_band: bool
    llm_label: int | None
    degraded_reason: str | None   # "ollama_indisponible" | "reponse_illisible" | None
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class EmailDetector:
    def __init__(self, ml, llm, model: str, prompt_version: str,
                 strategy: str = "llm_seul", max_chars: int = DEFAULT_MAX_CHARS):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Stratégie non prise en charge : {strategy!r} (attendu : {SUPPORTED_STRATEGIES})")
        self.ml, self.llm = ml, llm
        self.model, self.prompt_version = model, prompt_version
        self.strategy, self.max_chars = strategy, max_chars

    @classmethod
    def from_models_dir(cls, models_dir: str | Path | None = None, ollama_url: str = DEFAULT_URL,
                        timeout: float = 180.0) -> "EmailDetector":
        d = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        cfg = json.loads((d / "llm_config.json").read_text(encoding="utf-8"))
        ml = EmailML(d)
        if abs(cfg["ml_threshold"] - ml.threshold) > 5e-4:
            raise ValueError(f"llm_config.json (seuil {cfg['ml_threshold']}) ne correspond pas au modèle ML "
                             f"chargé (seuil {ml.threshold:.4f}) : fichiers de versions différentes ?")
        llm = OllamaClient(url=ollama_url, options=cfg["ollama_options"],
                           cache_path=d / "llm_cache.jsonl", timeout=timeout)
        return cls(ml, llm, cfg["model"], cfg["prompt"], cfg["strategy"], cfg.get("max_chars", DEFAULT_MAX_CHARS))

    def analyser(self, text: str) -> EmailVerdict:
        """text : l'email tel que dans le jeu de données (`Subject: ...` suivi du corps)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Email vide.")
        t0 = time.time()
        ml = self.ml.decide(text)

        def verdict(is_phishing, score, mode, reason, llm_label=None, degraded=None):
            return EmailVerdict("phishing" if is_phishing else "legitime", bool(is_phishing), float(score), mode, reason,
                                ml.p_ml, ml.is_phishing, ml.in_band, llm_label, degraded, round(time.time() - t0, 2))

        if self.strategy == "ml_seul":
            return verdict(ml.is_phishing, ml.p_ml, "ml_seul", "Décision du modèle ML seul.")

        system, user = build_prompt(self.prompt_version, text, self.max_chars)
        try:
            rec = self.llm.call(self.model, system, user)
        except OllamaError as e:
            return verdict(ml.is_phishing, ml.p_ml, "ml_degrade", f"LLM indisponible : {e}", None, "ollama_indisponible")
        label, score, reason = parse_answer(rec["content"])
        if label is None or math.isnan(score):
            return verdict(ml.is_phishing, ml.p_ml, "ml_degrade", "Réponse du LLM illisible.", None, "reponse_illisible")
        return verdict(bool(label), score, "llm", reason, label)
