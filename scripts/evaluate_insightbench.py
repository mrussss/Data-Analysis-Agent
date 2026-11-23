"""Run fixed and dynamic CSV investigations on two public InsightBench examples."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb

from data_analysis_agent import csv_analysis
from data_analysis_agent.business_benchmark import online_budget

from . import insightbench_smoke
from .analyze_csv import call_model, create_model, diagnostics
from .insightbench_assets import MANIFEST, MANIFEST_PATH, PROJECT, asset

EXPERIMENT = "insightbench-csv-v1"
JOURNAL = PROJECT / "data" / "processed" / f".{EXPERIMENT}-usage.json"
MAX_REQUESTS = 24
MAX_TOKENS = 80_000


def audit(run: dict) -> dict:
    observed = {row["id"]: row for row in run["evidence"] if row["status"] == "success"}
    checks = []
    for insight in run["insights"]:
        evidence_ids = {fact["evidence"] for fact in insight["facts"]}
        displays = []
        verified = bool(insight["facts"]) and evidence_ids == set(insight["evidence_ids"])
        for fact in insight["facts"]:
            source = observed.get(fact["evidence"])
            row, name = fact["row"], fact["column"]
            if not source or not 0 <= row < len(source["rows"]) or source["columns"].count(name) != 1:
                verified = False
                break
            value = source["rows"][row][source["columns"].index(name)]
            verified = verified and type(value) is type(fact["value"]) and value == fact["value"]
            displays.append(value)
        if verified:
            expected = csv_analysis.PLACEHOLDER.sub(lambda found: str(displays[int(found[1])]), insight["template"])
            verified = expected == insight["statement"]
        checks.append({
            "statement": insight["statement"], "evidence_ids_valid": verified,
            "result_values_match": verified,
            "numeric_cells": sum(type(value) in (int, float) for value in displays),
            "review_flags": (["tentative_hypothesis"] if insight["kind"] == "hypothesis" else [])
            + (["truncated_result"] if any(observed[eid]["truncated"] for eid in evidence_ids if eid in observed) else []),
        })
    return {"checked_insights": len(checks), "passed_value_checks": sum(item["result_values_match"] for item in checks),
            "semantic_correctness": "requires_human_review",
            "unanswered": run["assessment"]["gaps"], "checks": checks}


def _score_process(python: Path, predictions: list[dict]) -> dict:
    # The scoring process receives public predictions after execution and no model credentials.
    result = subprocess.run(
        [str(python.absolute()), "-m", "scripts.score_insightbench"], cwd=PROJECT,
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "OPENBLAS_NUM_THREADS": "1"},
        input=json.dumps(predictions, ensure_ascii=False), capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else "Scorer failed")
    return json.loads(result.stdout)


def _score_runs(python: Path, report: dict) -> None:
    inputs = [{"case": run["case"], "strategy": run["strategy"], "data_sha256": run["data_sha256"],
               "insights": [insight["statement"] for insight in run["insights"]]} for run in report["runs"]]
    scored = _score_process(python, inputs)
    for run, value in zip(report["runs"], scored.pop("runs"), strict=True):
        run["official_rouge1"] = value["rouge1"]
        run["official_score_details"] = value
    report["scoring"] = scored
    report.pop("scoring_error", None)
    for case in {run["case"] for run in report["runs"]}:
        arms = {run["strategy"]: run for run in report["runs"] if run["case"] == case}
        if set(arms) != {"fixed", "dynamic"}:
            continue
        fixed_text = {insight["statement"] for insight in arms["fixed"]["insights"]}
        report.setdefault("comparisons", {})[case] = {
            "same_actual_model": (len(arms["fixed"]["actual_models"]) == 1
                                  and arms["fixed"]["actual_models"] == arms["dynamic"]["actual_models"]),
            "dynamic_only_statements": [insight["statement"] for insight in arms["dynamic"]["insights"]
                                        if insight["statement"] not in fixed_text],
        }


def _save(path: Path, report: dict) -> None:
    with TemporaryDirectory(prefix=".insightbench-", dir=path.parent) as directory:
        pending = Path(directory) / "report.json"
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-agent", action="store_true")
    mode.add_argument("--offline-smoke", action="store_true")
    mode.add_argument("--rescore", type=Path)
    parser.add_argument("--cases", default="flag-1,flag-2")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned public assets")
    parser.add_argument("--scorer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Use a new output path in an existing directory")
    cases = tuple(args.cases.split(","))
    if len(set(cases)) != len(cases) or set(cases) - MANIFEST["cases"].keys():
        parser.error("Use distinct supported IDs flag-1 and flag-2")

    for name in MANIFEST["files"]:
        asset(name, download=args.download)
    # Check scoring dependencies before any model requests.
    _score_process(args.scorer_python, [])
    if args.rescore:
        report = json.loads(args.rescore.read_text())
        _score_runs(args.scorer_python, report)
        with args.output.open("x") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        print(args.output)
        return

    model = create_model(MAX_REQUESTS, MAX_TOKENS) if args.model_agent else None
    context = online_budget(model, (), journal_path=JOURNAL) if model else nullcontext((None, None, None, None))
    report = {
        "experiment_id": EXPERIMENT, "split": MANIFEST["split"], "upstream_revision": MANIFEST["revision"],
        "asset_manifest_sha256": sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "analysis_source_sha256": sha256(Path(csv_analysis.__file__).read_bytes()).hexdigest(),
        "duckdb_version": duckdb.__version__,
        "execution": "online_model" if model else "offline_scripted",
        "limits": {"requests": MAX_REQUESTS, "tokens": MAX_TOKENS, "actions_per_run": csv_analysis.ACTION_BUDGET},
        "runs": [],
    }
    with context as (_, _, _, journal):
        if journal and journal["usage"]["requests"]:
            parser.error(f"{EXPERIMENT} has already attempted online calls; use --rescore for saved predictions")
        with args.output.open("x") as file:
            if journal:
                report["budget_journal_id"] = journal["id"]
            json.dump(report, file)
        for case_id in cases:
            case = MANIFEST["cases"][case_id]
            started = perf_counter()
            with TemporaryDirectory(prefix="insightbench-csv-") as directory:
                snapshot = csv_analysis.prepare_csv(asset(case["csv"]), Path(directory))
                if snapshot.sha256 != MANIFEST["files"][case["csv"]]:
                    raise ValueError("Staged CSV hash differs from the manifest")
                import_ms = round((perf_counter() - started) * 1000, 2)
                order = ("fixed", "dynamic") if case_id == "flag-1" else ("dynamic", "fixed")
                for strategy in order:
                    if model and model.exhausted:
                        report.setdefault("skipped", []).append({"case": case_id, "strategy": strategy,
                                                                 "reason": "model_budget_exhausted"})
                        continue
                    start = model.requests if model else 0
                    planner = (lambda prompt: call_model(model, prompt)) if model else insightbench_smoke.plan
                    run = csv_analysis.investigate(snapshot, case["goal"], planner=planner, strategy=strategy)
                    usage = diagnostics(model, start) if model else {
                        "requested_model": None, "actual_models": [], "requests": 0,
                        "known_tokens": 0, "token_usage": 0, "model_attempts": [],
                    }
                    report["runs"].append({**run, **usage, "case": case_id, "import_ms": import_ms,
                                           "execution": report["execution"], "evidence_audit": audit(run)})
                    if journal:
                        report["online_usage"] = journal["usage"]
                    _save(args.output, report)
        try:
            _score_runs(args.scorer_python, report)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            report["scoring_error"] = str(error)
            _save(args.output, report)
            raise
        _save(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
