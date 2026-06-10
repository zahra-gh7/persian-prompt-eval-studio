from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
PROMPTS_FILE = PROMPTS_DIR / "prompts.json"
RUNS_FILE = RESULTS_DIR / "runs.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
DEFAULT_RESPONSE_MAX_TOKENS = int(os.getenv("GEMINI_RESPONSE_MAX_TOKENS", "512"))
DEFAULT_JUDGE_MAX_TOKENS = int(os.getenv("GEMINI_JUDGE_MAX_TOKENS", "300"))
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

MODEL_TOKEN_PRICES = {
    "gemini-pro": 0.00025,
    "gemini-1.5-pro": 0.0005,
    "gemini-1.5-flash": 0.00015,
    "gemini-2.5-flash": 0.0002,
}

DEFAULT_PROMPTS = [
    {
        "name": "Persian Question Answering",
        "description": "A basic expert-answer prompt for Persian questions.",
        "template": "You are an expert Persian assistant. Answer the following question clearly and accurately:\n\n{input}",
    },
    {
        "name": "Structured Persian Response",
        "description": "A prompt that encourages a structured answer with reasoning.",
        "template": "You are a helpful assistant. Read the question and provide a short Persian answer with a concise explanation:\n\n{input}",
    },
]
