"""Vérifie que la pipeline `src/` redonne les chiffres du notebook 05 sur `dev` (jamais sur `test`).

Prérequis : models/llm_cache.jsonl intact (sinon chaque email déclenche un appel Ollama, ~13 s).
Usage, depuis la racine du dépôt :  python scripts/check_parity_dev.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.email_pipeline import EmailDetector  # noqa: E402

CSV = ROOT / "data" / "hard" / "mini_jeu_etendu.csv"
EVAL = ROOT / "models" / "email_llm_eval.json"


def main() -> int:
    det = EmailDetector.from_models_dir()
    cached_before = len(det.llm)
    df = pd.read_csv(CSV)
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    y = dev["label"].astype(int).values
    res = [det.analyser(t) for t in dev["text"].fillna("").astype(str)]
    pred = [int(r.is_phishing) for r in res]
    rec = sum(p for p, t in zip(pred, y) if t == 1) / (y == 1).sum()
    fpr = sum(p for p, t in zip(pred, y) if t == 0) / (y == 0).sum()
    degraded = sum(r.mode != "llm" for r in res)
    new_calls = len(det.llm) - cached_before
    print(f"pipeline src/ sur dev : recall {rec:.3f} | fpr {fpr:.3f} | balanced accuracy {(rec + 1 - fpr) / 2:.3f}")
    print(f"emails en mode dégradé : {degraded} | appels Ollama hors cache : {new_calls}")

    expected = next(r for r in json.loads(EVAL.read_text(encoding="utf-8"))["règles_dev"] if r["règle"] == "llm_seul")
    ok = abs(rec - expected["recall"]) < 1e-6 and abs(fpr - expected["fpr"]) < 1e-6 and degraded == 0
    print(f"notebook 05 (llm_seul, dev) : recall {expected['recall']:.3f} | fpr {expected['fpr']:.3f}")
    print("PARITÉ OK" if ok else "ÉCART : la pipeline ne reproduit pas le notebook (voir les chiffres ci-dessus)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
