from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import BASE_DIR, PROMPTS_DIR, PROMPTS_FILE, RESULTS_DIR, RUNS_FILE, DEFAULT_PROMPTS


class PromptStorage:
    @staticmethod
    def initialize() -> None:
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        if not PROMPTS_FILE.exists():
            PromptStorage.save_prompts(DEFAULT_PROMPTS)

    @staticmethod
    def load_prompts() -> list[dict[str, Any]]:
        if not PROMPTS_FILE.exists():
            return []
        with PROMPTS_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def save_prompts(prompts: list[dict[str, Any]]) -> None:
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        with PROMPTS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(prompts, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def save_prompt(prompt: dict[str, Any]) -> None:
        prompts = PromptStorage.load_prompts()
        existing = next((item for item in prompts if item["name"] == prompt["name"]), None)
        if existing:
            prompts = [prompt if item["name"] == prompt["name"] else item for item in prompts]
        else:
            prompts.append(prompt)
        PromptStorage.save_prompts(prompts)

    @staticmethod
    def delete_prompt(name: str) -> None:
        prompts = PromptStorage.load_prompts()
        prompts = [item for item in prompts if item["name"] != name]
        PromptStorage.save_prompts(prompts)


class RunStorage:
    @staticmethod
    def initialize() -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not RUNS_FILE.exists():
            with RUNS_FILE.open("w", encoding="utf-8") as handle:
                json.dump([], handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_runs() -> list[dict[str, Any]]:
        if not RUNS_FILE.exists():
            return []
        with RUNS_FILE.open("r", encoding="utf-8") as handle:
            runs = json.load(handle)
        runs.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return runs

    @staticmethod
    def save_run(run_data: dict[str, Any]) -> None:
        RunStorage.initialize()
        runs = RunStorage.load_runs()
        runs.insert(0, run_data)
        with RUNS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(runs, handle, ensure_ascii=False, indent=2)

        run_file = RESULTS_DIR / f"run_{run_data['timestamp']}.json"
        with run_file.open("w", encoding="utf-8") as handle:
            json.dump(run_data, handle, ensure_ascii=False, indent=2)
