"""One development-only comparison against the retained Mode B model runs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from data_analysis_agent import business_verified
from data_analysis_agent.business_benchmark import _digest, _model_plan, _probe_low, _run, online_budget, score_deep
from data_analysis_agent.business_cases import DEEP_DEV, build_retail
from data_analysis_agent.model_budget import PROJECT_ENV_PATH, BudgetedModel

EXPERIMENT_ID = "mode-b-verified-dev-v1"
PROJECT = PROJECT_ENV_PATH.parent
HISTORY = PROJECT / "evaluation" / "business_deep_online_dev.json"
JOURNAL = PROJECT / "data" / "processed" / f".{EXPERIMENT_ID}-usage.json"
ACTION_BUDGET = 5
MAX_REQUESTS = 24
MAX_TOKENS = 80_000
MODEL = "deepseek-v4.1-flash"


def compare(model: BudgetedModel | None) -> dict:
    history = json.loads(HISTORY.read_text())
    if history["budget"] != ACTION_BUDGET or history["model"] != MODEL:
        raise ValueError("Historical model or action budget does not match")
    historical = {row["case"]: row for row in history["runs"] if row["arm"] == "llm_adaptive"}
    probe = _probe_low(model) if model else None
    rows, skipped = [], []
    with TemporaryDirectory(prefix=EXPERIMENT_ID) as folder:
        for case in DEEP_DEV:
            database = Path(folder) / f"{case.id}.duckdb"
            events = build_retail(case, database)
            rows.append({**historical[case.id], "historical": True})
            rows.append(_run(database, events, case, "rule_adaptive", ACTION_BUDGET, None))
            if not model:
                continue
            if probe["outcome"] != "accepted" or model.exhausted:
                reason = "low_probe_failed" if probe["outcome"] != "accepted" else "model_budget_exhausted"
                skipped.append({"case": case.id, "reason": reason})
                continue
            start = model.requests
            started = perf_counter()
            plan = _model_plan(model)

            def plan_with_phase(prompt):
                first = model.requests
                phase = json.loads(prompt.split("\n", 1)[1])["phase"]
                try:
                    return plan(prompt)
                finally:
                    for call in model.planner_calls[first:]:
                        call["decision_phase"] = phase

            state = business_verified.investigate_verified(database, case.goal, budget=ACTION_BUDGET, planner=plan_with_phase,
                                         events=events)
            calls = model.planner_calls[start:]
            usage = model.usages[start:]
            actual_models = sorted({call.get("reported_model") or call["model"] for call in calls
                                    if "finish_reason" in call})
            phases = {
                phase: sum(call["decision_phase"] == phase for call in calls)
                for phase in ("assess_and_plan", "final_assessment")
            }
            rows.append({
                "case": case.id, "arm": state.policy, **score_deep(state, case),
                "actions": len(state.evidence), "requests": len(calls),
                "tokens": None if None in usage else sum(usage), "actual_models": actual_models,
                "same_actual_model": len(actual_models) == 1 and actual_models == historical[case.id]["actual_models"],
                "latency_ms": round((perf_counter() - started) * 1000, 2), "response_diagnostics": calls,
                "requests_by_phase": phases, "standalone_verifier_requests": 0, "details": asdict(state),
            })

    for row in rows:
        collected = row["required_total"] - len(row["missing_evidence"])
        row["collection_coverage"] = {"found": collected, "total": row["required_total"]}
    return {
        "experiment_id": EXPERIMENT_ID, "historical_result": HISTORY.name,
        "historical_result_sha256": _digest(HISTORY),
        "strategy_sha256": _digest(Path(business_verified.__file__)),
        "budget": ACTION_BUDGET, "limits": {"requests": MAX_REQUESTS, "tokens": MAX_TOKENS},
        "model": MODEL, "execution": "online" if model else "offline",
        "low_probe": probe, "online_usage": model.usage() if model else None,
        "skipped": skipped, "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Use a new output file in an existing directory")
    if args.online:
        model = BudgetedModel(max_requests=MAX_REQUESTS, max_tokens=MAX_TOKENS)
        if model.name != MODEL:
            parser.error(f"This experiment requires {MODEL}")
        model.max_requests = min(model.max_requests, MAX_REQUESTS)
        model.max_tokens = min(model.max_tokens, MAX_TOKENS)
        with online_budget(model, (), journal_path=JOURNAL) as (_, _, _, journal):
            if journal["usage"]["requests"]:
                parser.error(f"{EXPERIMENT_ID} has already attempted online calls")
            with args.output.open("x") as output:
                payload = compare(model)
                payload.update(budget_journal_id=journal["id"], cumulative_online_usage=journal["usage"])
                json.dump(payload, output, default=str, ensure_ascii=False, indent=2)
    else:
        with args.output.open("x") as output:
            json.dump(compare(None), output, default=str, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
