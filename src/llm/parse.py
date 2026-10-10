"""Lecture tolérante de la réponse JSON du LLM (identique au notebook 05)."""
from __future__ import annotations

import json
import math
import re


def parse_answer(content: str):
    """-> (label 1/0 ou None, score phishing 0..1 ou NaN, raison).

    label None = réponse illisible : l'appelant doit retomber sur le ML.
    """
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content or "", re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None, math.nan, ""
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict.startswith("phish"):
        label = 1
    elif verdict.startswith("legit"):
        label = 0
    else:
        return None, math.nan, str(obj.get("reason", ""))[:200]
    try:
        conf = float(obj.get("confidence", 50))
    except Exception:
        conf = 50.0
    conf = min(max(conf, 0.0), 100.0) / 100.0
    score = conf if label == 1 else 1.0 - conf
    return label, score, str(obj.get("reason", ""))[:300]
