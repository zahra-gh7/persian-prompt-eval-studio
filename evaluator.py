from __future__ import annotations

import time
from typing import Any

from config import DEFAULT_RESPONSE_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_MODEL, DEFAULT_JUDGE_MODEL
from judge import evaluate_response
from llm_client import GeminiClient, LLMClientError
from metrics import build_metrics
from utils import estimate_cost, estimate_tokens, format_prompt_text, format_timestamp


class EvaluationRunner:
    def __init__(self, model: str | None = None, judge_model: str | None = None) -> None:
        self.client = GeminiClient()
        self.model = model or DEFAULT_MODEL
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL

    def run(
        self,
        dataset_name: str,
        dataset_inputs: list[str],
        prompts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []

        for row_index, input_text in enumerate(dataset_inputs, start=1):
            for prompt in prompts:
                prompt_text = format_prompt_text(prompt["template"], input_text)
                try:
                    start_time = time.perf_counter()
                    response = self.client.generate_response(
                        prompt_text,
                        model=self.model,
                        max_tokens=DEFAULT_RESPONSE_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE,
                    )
                    latency = time.perf_counter() - start_time
                    reply_text = response.get("output", "")
                    usage = response.get("usage", {})
                    token_count = usage.get("total_tokens") or estimate_tokens(prompt_text + reply_text)
                    assert token_count is not None
                    cost = estimate_cost(token_count, self.model)
                    judge_score = evaluate_response(
                        input_text=input_text,
                        prompt_template=prompt["template"],
                        response_text=reply_text,
                        llm_model=self.judge_model,
                    )
                except AssertionError:
                    latency = 0.0
                    reply_text = ""
                    token_count = estimate_tokens(prompt_text)
                    cost = estimate_cost(token_count, self.model)
                    judge_score = {
                        "accuracy": 1,
                        "completeness": 1,
                        "clarity": 1,
                        "relevance": 1,
                        "overall": 1,
                        "reasoning": "Response parsing failed."
                    }
                except LLMClientError as exc:
                    latency = 0.0
                    reply_text = f"[LLM error] {exc}"
                    token_count = estimate_tokens(prompt_text)
                    cost = estimate_cost(token_count, self.model)
                    judge_score = {
                        "accuracy": 1,
                        "completeness": 1,
                        "clarity": 1,
                        "relevance": 1,
                        "overall": 1,
                        "reasoning": str(exc),
                    }

                rows.append(
                    {
                        "row_index": row_index,
                        "input_text": input_text,
                        "prompt_name": prompt["name"],
                        "prompt_template": prompt["template"],
                        "response_text": reply_text,
                        "latency_seconds": round(latency, 3),
                        "token_count": int(token_count),
                        "estimated_cost": round(cost, 6),
                        "judge_score": judge_score,
                    }
                )

        run_data = {
            "timestamp": format_timestamp(),
            "dataset_name": dataset_name,
            "model": self.model,
            "judge_model": self.judge_model,
            "prompts": prompts,
            "rows": rows,
            "metrics": build_metrics(rows),
        }

        return run_data
