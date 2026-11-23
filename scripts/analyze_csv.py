"""Analyze one local CSV with the configured Low model and save its report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_analysis_agent.csv_analysis import analyze_csv
from data_analysis_agent.deepseek_provider import low_reasoning
from data_analysis_agent.model_budget import BudgetedModel


def create_model(requests: int, tokens: int) -> BudgetedModel:
    model = BudgetedModel(max_requests=requests, max_tokens=tokens)
    if low_reasoning(model.name).get("reasoning_effort") != "low":
        raise ValueError("Configure a DeepSeek Low primary model")
    model.max_requests = min(model.max_requests, requests)
    model.max_tokens = min(model.max_tokens, tokens)
    return model


def call_model(model: BudgetedModel, prompt: str) -> str:
    payload = json.loads(prompt.split("\n", 1)[1])
    first = len(model.planner_calls)
    try:
        return model(prompt, output_limit=3072, planning=True)
    finally:
        for call in model.planner_calls[first:]:
            call["analysis_phase"] = payload["phase"]
            if "correction" in payload:
                call["correction_of"] = payload["correction"]


def diagnostics(model: BudgetedModel, start: int = 0) -> dict:
    attempts = model.planner_calls[start:]
    tokens = model.usages[start:]
    return {"requested_model": model.name,
            "actual_models": sorted({call.get("reported_model") or call["model"]
                                     for call in attempts if "finish_reason" in call}),
            "requests": len(attempts), "token_usage": None if None in tokens else sum(tokens),
            "known_tokens": sum(value for value in tokens if value is not None),
            "model_attempts": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--strategy", choices=["fixed", "dynamic"], default="dynamic")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Use a new output path in an existing directory")
    model = create_model(8, 40_000)
    with args.output.open("x") as output:
        report = analyze_csv(args.csv, args.goal, strategy=args.strategy, planner=lambda prompt: call_model(model, prompt))
        json.dump({**report, **diagnostics(model)}, output, indent=2, ensure_ascii=False, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
