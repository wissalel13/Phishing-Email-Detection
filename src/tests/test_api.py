"""Tests de l'API avec un détecteur factice (sans Ollama). Nécessite fastapi et httpx."""
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except Exception:       # fastapi / httpx absents
    HAVE_FASTAPI = False

from src.inference.email_pipeline import EmailVerdict
from src.llm.client import OllamaError


class StubLLM:
    def __init__(self, models=None, error=None):
        self.models, self.error = models or {"m"}, error

    def available_models(self):
        if self.error:
            raise self.error
        return self.models

    def __len__(self):
        return 3


class StubDetector:
    model = "m"

    def __init__(self, mode="llm", llm=None):
        self.mode, self.llm, self.seen = mode, llm or StubLLM(), []

    def analyser(self, text):
        if not text.strip():
            raise ValueError("Email vide.")
        self.seen.append(text)
        degraded = self.mode == "ml_degrade"
        return EmailVerdict("phishing", True, 0.9, self.mode, "<script>alert(1)</script>", 0.8, True, False,
                            None if degraded else 1, "ollama_indisponible" if degraded else None, 0.1)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx non installés")
class ApiTests(unittest.TestCase):
    def client(self, det):
        from src.api.main import create_app
        return TestClient(create_app(detector=det))

    def test_analyze_ok_and_text_composition(self):
        det = StubDetector()
        with self.client(det) as c:
            r = c.post("/analyze", json={"subject": "Hi.", "body": "Click here"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual((body["verdict"], body["mode"], body["avertissement"]), ("phishing", "llm", None))
        self.assertEqual(det.seen, ["Hi. Click here"])

    def test_degraded_mode_carries_a_warning(self):
        with self.client(StubDetector(mode="ml_degrade")) as c:
            body = c.post("/analyze", json={"text": "hello"}).json()
        self.assertEqual(body["mode"], "ml_degrade")
        self.assertIn("NON vérifiée", body["avertissement"])

    def test_empty_request_is_422(self):
        with self.client(StubDetector()) as c:
            self.assertEqual(c.post("/analyze", json={}).status_code, 422)

    def test_health_ok_and_degraded(self):
        with self.client(StubDetector()) as c:
            self.assertEqual(c.get("/health").json()["status"], "ok")
        with self.client(StubDetector(llm=StubLLM(error=OllamaError("down")))) as c:
            h = c.get("/health").json()
        self.assertEqual((h["status"], h["ollama"]), ("degraded", False))

    def test_index_page(self):
        with self.client(StubDetector()) as c:
            r = c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("PhishGuard AI", r.text)


if __name__ == "__main__":
    unittest.main()
