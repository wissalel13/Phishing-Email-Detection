"""Client Ollama avec cache disque (clé identique à celle du notebook 05)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:11434"
DEFAULT_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 220, "num_ctx": 4096}


class OllamaError(RuntimeError):
    """Ollama injoignable, en erreur ou trop lent."""


class OllamaClient:
    def __init__(self, url: str = DEFAULT_URL, options: dict | None = None,
                 cache_path: str | Path | None = None, timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.seed = self.options["seed"]
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec
                    except Exception:      # dernière ligne tronquée après un arrêt brutal
                        continue

    # -- cache ----------------------------------------------------------------------
    def cache_key(self, model: str, system: str, user: str) -> str:
        return hashlib.sha1(json.dumps([model, system, user, self.seed], ensure_ascii=False).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._cache)

    # -- Ollama ---------------------------------------------------------------------
    def available_models(self) -> set[str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
            return {m["name"] for m in r.json().get("models", [])}
        except Exception as e:
            raise OllamaError(f"Ollama injoignable sur {self.url} ({e}). Lancer `ollama serve`.") from e

    def call(self, model: str, system: str, user: str, use_cache: bool = True) -> dict:
        """-> {"key", "model", "content", "seconds", "cached"}. Lève OllamaError si l'appel échoue."""
        key = self.cache_key(model, system, user)
        if use_cache and key in self._cache:
            return {**self._cache[key], "cached": True}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False, "format": "json", "keep_alive": "30m", "options": self.options,
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except Exception as e:
            raise OllamaError(f"Appel Ollama en échec ({type(e).__name__}: {e})") from e
        rec = {"key": key, "model": model, "content": content, "seconds": round(time.time() - t0, 2)}
        if use_cache:
            with self._lock:
                self._cache[key] = rec
                if self.cache_path:
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {**rec, "cached": False}
