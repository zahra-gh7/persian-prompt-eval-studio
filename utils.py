from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from typing import Any

import pandas as pd

from config import MODEL_TOKEN_PRICES


def format_timestamp(timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.utcnow()
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def safe_load_json(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = raw_text.strip()
        braces_start = cleaned.find("{")
        braces_end = cleaned.rfind("}")
        if braces_start != -1 and braces_end != -1:
            cleaned = cleaned[braces_start : braces_end + 1]
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        return {
            "accuracy": 1,
            "completeness": 1,
            "clarity": 1,
            "relevance": 1,
            "overall": 1,
            "reasoning": "Unable to parse judge response: invalid JSON.",
        }


def estimate_tokens(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    return max(1, len(normalized) // 4)


def estimate_cost(tokens: int, model: str = "gemini-pro") -> float:
    rate = MODEL_TOKEN_PRICES.get(model, 0.0)
    return tokens / 1000 * rate


def load_csv(uploaded_file: Any) -> pd.DataFrame:
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        content = uploaded_file.read()
        if isinstance(content, bytes):
            try:
                return pd.read_csv(io.BytesIO(content))
            except Exception:
                return pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))
    return pd.read_csv(uploaded_file)


def validate_csv_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            return {
                "valid": False,
                "error": "CSV file is empty."
            }

        return {"valid": True}

    except Exception as e:
        return {
            "valid": False,
            "error": f"Invalid CSV file: {str(e)}"
        }
    
def format_prompt_text(template: str, input_text: str) -> str:
    if "{input}" in template:
        return template.replace("{input}", input_text.strip())

    return f"{template.strip()}\n\n{input_text.strip()}"


def numeric_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
