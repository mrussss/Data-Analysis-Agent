"""Dispatch according to whether a task needs evidence feedback."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .investigation import investigate, parse_comparison, report
from .model_budget import BudgetedModel
from .query_contract import answer_contract
from .schema import inspect_schema
from .sql_generator import generate_sql


def route_question(question: str, domain: str) -> str:
    prefix = question.strip().split(maxsplit=1)[0].lower() if question.strip() else ""
    if prefix not in {"why", "investigate", "decompose"}:
        return "query"
    comparison = parse_comparison(question, domain)
    return "workflow" if len(comparison.dimensions) == 1 else "adaptive"


def run_task(database: Path, question: str, domain: str, *, command: str = "auto",
             baseline: Path | None = None, budget: int = 7, model: BudgetedModel | None = None) -> dict:
    policy = (route_question(question, domain) if command == "auto"
              else "adaptive" if command == "investigate" else "query")
    if policy == "query":
        generation = generate_sql(question, inspect_schema(database), model) if model else None
        answer = answer_contract(database, question, domain, generation.sql if generation else None)
        payload = asdict(answer)
        if answer.contract:
            payload["metric_definition"] = asdict(answer.contract.spec)
        if generation:
            payload["model_proposal"] = asdict(generation)
    else:
        comparison = parse_comparison(question, domain)
        payload = report(investigate((baseline or database, database), comparison, policy=policy,
                                     budget=budget, planner=model.plan if model else None))
    payload["route"] = policy
    if model:
        payload["model_usage"] = {"name": model.name, "requests": model.requests,
                                  "tokens": model.tokens, "exhausted": model.exhausted,
                                  "unknown_usage_requests": model.usages.count(None)}
    return payload
