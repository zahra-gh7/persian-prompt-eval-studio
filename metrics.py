from __future__ import annotations

import pandas as pd
from typing import Any

from utils import numeric_average


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "average_score": 0.0,
            "average_latency": 0.0,
            "total_evaluations": 0,
            "best_prompt": None,
        }

    by_prompt: dict[str, list[float]] = {}
    latencies: list[float] = []
    scores: list[float] = []
    token_counts: list[int] = []
    costs: list[float] = []

    for row in rows:
        prompt_name = row["prompt_name"]
        score = float(row["judge_score"]["overall"])
        latencies.append(float(row.get("latency_seconds", 0.0)))
        scores.append(score)
        token_counts.append(int(row.get("token_count", 0)))
        costs.append(float(row.get("estimated_cost", 0.0)))
        by_prompt.setdefault(prompt_name, []).append(score)

    best_prompt = max(by_prompt.items(), key=lambda item: numeric_average(item[1]))[0]

    return {
        "average_score": round(numeric_average(scores), 2),
        "average_latency": round(numeric_average(latencies), 3),
        "total_evaluations": total,
        "best_prompt": best_prompt,
        "total_tokens": sum(token_counts),
        "total_estimated_cost": round(sum(costs), 6),
    }


def summarize_prompt_metrics(run_data: dict[str, Any]) -> pd.DataFrame:
    prompt_groups: dict[str, list[dict[str, Any]]] = {}
    for row in run_data.get("rows", []):
        prompt_groups.setdefault(row["prompt_name"], []).append(row)

    summary = []
    for prompt_name, records in prompt_groups.items():
        averages = [float(record["judge_score"]["overall"]) for record in records]
        latencies = [float(record.get("latency_seconds", 0.0)) for record in records]
        summary.append(
            {
                "prompt_name": prompt_name,
                "average_overall": round(numeric_average(averages), 2),
                "average_latency": round(numeric_average(latencies), 3),
                "evaluations": len(records),
            }
        )

    return pd.DataFrame(summary)


def compute_current_run_metrics(run_data: dict[str, Any]) -> dict[str, Any]:
    return build_metrics(run_data.get("rows", []))
