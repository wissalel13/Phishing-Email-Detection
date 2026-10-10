"""API FastAPI + petite interface web pour PhishGuard (emails).

Lancer, depuis la racine du dépôt :   uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Puis ouvrir http://127.0.0.1:8000  (interface)  ou  http://127.0.0.1:8000/docs  (Swagger).

Le contenu des emails n'est ni journalisé ni stocké par l'API. Aucune authentification :
ne pas exposer ce service sur un réseau sans proxy authentifié.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from src.api.ui import INDEX_HTML
from src.inference.email_pipeline import EmailDetector
from src.llm.client import OllamaError

WARNING_DEGRADED = (
    "Analyse NON vérifiée par le LLM ({why}) : ce verdict vient du modèle ML seul, "
    "qui se trompe souvent sur les emails difficiles (60 % de faux positifs sur le jeu de test). "
    "À traiter comme une simple indication."
)
_WHY = {"ollama_indisponible": "Ollama est indisponible", "reponse_illisible": "réponse du LLM illisible"}


def create_app(detector: EmailDetector | None = None) -> FastAPI:
    """`detector` injectable (tests) ; sinon chargé une fois au démarrage depuis models/."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector or EmailDetector.from_models_dir()
        yield

    app = FastAPI(title="PhishGuard AI - emails", version="1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return INDEX_HTML

    @app.get("/health", response_model=HealthResponse)
    def health():
        det = app.state.detector
        ollama, model_ok = True, False
        try:
            model_ok = det.model in det.llm.available_models()
        except OllamaError:
            ollama = False
        return HealthResponse(status="ok" if (ollama and model_ok) else "degraded", ml_loaded=True, ollama=ollama,
                              model=det.model, model_disponible=model_ok, cache_size=len(det.llm))

    # `def` (pas `async def`) : l'appel au LLM dure ~13 s et bloquerait la boucle d'événements.
    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest):
        try:
            v = app.state.detector.analyser(req.email_text())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        warning = None
        if v.mode == "ml_degrade":
            warning = WARNING_DEGRADED.format(why=_WHY.get(v.degraded_reason, "LLM indisponible"))
        return AnalyzeResponse(**v.to_dict(), avertissement=warning)

    return app


app = create_app()
