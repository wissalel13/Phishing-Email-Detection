"""Schémas de l'API (pydantic uniquement : testables sans FastAPI)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

MAX_CHARS_INPUT = 20000   # garde-fou ; le LLM ne lit de toute façon que les 1800 premiers caractères


class AnalyzeRequest(BaseModel):
    """Soit `text` (email complet), soit `subject` et/ou `body`."""
    text: Optional[str] = Field(None, max_length=MAX_CHARS_INPUT)
    subject: Optional[str] = Field(None, max_length=1000)
    body: Optional[str] = Field(None, max_length=MAX_CHARS_INPUT)

    @model_validator(mode="after")
    def _not_empty(self):
        if not self.email_text():
            raise ValueError("Fournir `text`, ou `subject` / `body`, non vides.")
        return self

    def email_text(self) -> str:
        """Texte au format du jeu de données : le sujet est la première phrase, sans préfixe `Subject:`."""
        if self.text and self.text.strip():
            return self.text.strip()
        subject = (self.subject or "").strip().rstrip(".")
        body = (self.body or "").strip()
        return f"{subject}. {body}".strip() if subject and body else (subject or body)


class AnalyzeResponse(BaseModel):
    verdict: str                    # "phishing" | "legitime"
    is_phishing: bool
    score: float                    # score de phishing 0..1
    mode: str                       # "llm" | "ml_degrade" | "ml_seul"
    reason: str
    p_ml: float
    ml_is_phishing: bool
    in_band: bool
    llm_label: Optional[int] = None
    degraded_reason: Optional[str] = None
    seconds: float
    avertissement: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                     # "ok" | "degraded"
    ml_loaded: bool
    ollama: bool
    model: str
    model_disponible: bool
    cache_size: int
