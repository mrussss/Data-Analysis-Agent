"""Run the same frozen tasks against all policies and independently score the outputs."""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from . import benchmark_cases as cases
from .investigation import Evidence, InvestigationState, assess, investigate, parse_comparison, report
from .observability import finalize_observability, start_observability_request
from .query_contract import answer_contract
from .question_service import answer_question
from .routing import route_question
from .schema import inspect_schema
from .sql_generator import generate_sql

QUERY_ARMS = ("a_only", "no_contract", "workflow", "adaptive", "auto")
INV_ARMS = ("a_only", "scan", "workflow", "frozen", "adaptive", "auto")
UPSTREAM_COMMIT = "0784efcfa1e3aa3b5ee8cb9efbf158db501427d9"


def equivalent(rows, expected) -> bool:
    return len(rows) == len(expected) and all(
        len(row) == len(gold) and all(
            actual == truth or (
                isinstance(truth, (float, int))
                and isinstance(actual, (float, int))
                and math.isclose(actual, truth, rel_tol=1e-9, abs_tol=1e-9)
            )
            for actual, truth in zip(row, gold)
        ) for row, gold in zip(rows, expected)
    )


def run_gold(database, case, arm, online, schema, expected):
    calls = []
    if online:
        model = online
    else:
        def model(prompt):
            calls.append(None)
            return {"status": "success", "sql": (case.retry or case.proposal)
                    if "Original SQL:" in prompt else case.proposal}
    if arm == "no_contract":
        raw = answer_question(database, case.question, model, schema=schema)
        result = raw.execution_result if raw.status == "success" else None
        details = asdict(raw)
    else:
        generation = generate_sql(case.question, schema, model)
        raw = answer_contract(database, case.question, "olist", generation.sql)
        result = raw.result
        details = {"answer": asdict(raw), "proposal": asdict(generation)}
    accepted = result is not None and not result.truncated
    # An unavailable answer requires refusal. All numeric gold tasks require a full result.
    correct = not accepted if expected is None else accepted and equivalent(result.rows, expected)
    return {
        "correct": correct, "accepted": accepted, "should_refuse": expected is None,
        "error_accepted": accepted and not correct, "gold_rows": expected,
        "details": details, "offline_proposals": len(calls),
    }


def evidence_reconciles(evidence: Evidence, delta: float) -> bool:
    if evidence.error:
        return False
    return math.isclose(sum((r.within or 0) + (r.mix or 0) for r in evidence.segments),
                        delta, abs_tol=1e-9, rel_tol=1e-9)


def supported(state: InvestigationState, hypothesis) -> bool:
    by_id = {e.id: e for e in state.evidence}
    if not hypothesis.evidence or not set(hypothesis.evidence) <= by_id.keys():
        return False
    source = by_id[hypothesis.evidence[0]]
    if source.action != hypothesis.action:
        return False
    return any(r.key == hypothesis.segment and getattr(r, hypothesis.effect) == hypothesis.contribution
               for r in source.segments)


def mechanism(hypothesis, edit) -> bool:
    coordinates = dict(zip(hypothesis.action, hypothesis.segment))
    if coordinates.get("state") != edit["state"]:
        return False
    if edit["weekday"] and coordinates.get("weekday") != str(edit["weekday"]):
        return False
    if edit["direction"] == 0:
        return (hypothesis.effect == "mix"
                and math.isclose(hypothesis.contribution, edit["mix_contribution"], abs_tol=1e-9))
    return (hypothesis.effect == "within"
            and hypothesis.contribution * edit["late_delta"] > 0
            and (edit["weekday"] or len(coordinates) == 1))


def effective_replans(state: InvestigationState) -> tuple[int, int]:
    opportunities = valid = 0
    for prior, next_step in zip(state.trace, state.trace[1:]):
        refuted = [u for u in prior["updates"] if u["before"] == "supported"
                   and u["after"] == "refuted" and u["evidence"][-1] == next_step["after"]]
        if refuted:
            opportunities += 1
            next_action = next_step["action"]
            valid += (len(next_action) == 2 and next_action != next_step["planned_before_evidence"]
                      and next_action != prior["action"]
                      and any(set(next_action) & set(u["action"]) for u in refuted))
    return opportunities, valid



def historical_matches(evidence: Evidence, expected, counts) -> bool:
    if not evidence.action:
        return not evidence.error
    if evidence.error:
        return False
    periods = [{tuple(row[:-2]): row[-2:] for row in rows} for rows in expected["+".join(evidence.action)]]
    center = (counts[0][1] / counts[0][0] + counts[1][1] / counts[1][0]) / 2
    seen = set()
    for row in evidence.segments:
        n, a = periods[0].get(row.key, (0, 0))
        m, b = periods[1].get(row.key, (0, 0))
        before, after = a / n if n else None, b / m if m else None
        # Compute the sum of both effects directly via centered contributions.
        difference = ((b - center * m) / counts[1][0] -
                      (a - center * n) / counts[0][0])
        if ((row.before_n, row.before, row.after_n, row.after) != (n, before, m, after)
                or not math.isclose((row.within or 0) + (row.mix or 0), difference, abs_tol=1e-9)):
            return False
        seen.add(row.key)
    return set(periods[0]).union(periods[1]) <= seen

def score_investigation(state, manifest, control):
    before_n, before_late = manifest["baseline"]
    after_n, after_late = manifest["current"]
    head = state.evidence[0]
    observed = head.segments[0] if head.segments else None
    expected = (before_late / before_n, after_late / after_n) if before_n and after_n else None
    numeric = bool(expected and observed and (observed.before_n, observed.after_n) == (before_n, after_n)
                   and equivalent(((observed.before, observed.after),), (expected,)))
    clean = numeric and all(evidence_reconciles(e, expected[1] - expected[0]) for e in state.evidence)
    edits = manifest["edits"]
    hypotheses = state.reported
    hits = [[index for index, edit in enumerate(edits) if mechanism(h, edit)] for h in hypotheses]
    found = set().union(*hits)
    opportunities, replans = effective_replans(state)
    evidence_supported = sum(supported(state, h) for h in hypotheses)
    if control:
        correct = state.stop_reason == control and not hypotheses
    elif edits:
        correct = clean and len(found) == len(edits) and evidence_supported == len(hypotheses)
    else:
        correct = (clean and len(state.evidence) > 1 and evidence_supported == len(hypotheses)
                   and all(historical_matches(e, manifest["historical_groups"],
                                              (manifest["baseline"], manifest["current"]))
                           for e in state.evidence))
    return {
        "correct": correct, "numeric_correct": numeric, "reconciles": bool(clean),
        "top1": bool(hits and hits[0]) if edits else None,
        "top3": bool(found) if edits else None,
        "all_mechanisms_top3": len(found) == len(edits) if edits else None,
        "supported_claims": evidence_supported, "claims": len(hypotheses),
        "ungrounded_claims": len(hypotheses) - evidence_supported,
        "control_false_attributions": len(hypotheses) if control else 0,
        "off_target_injection_claims": sum(not groups for groups in hits) if edits else 0,
        "replan_opportunities": opportunities, "effective_replans": replans,
        "stop_reason": state.stop_reason, "gold": manifest, "details": report(state),
    }


def _measure_run(arm, case, task_type, policy, run, online):
    first_request = online.requests if online else 0
    with start_observability_request() as collector:
        result = run()
        trace = finalize_observability(
            collector, route=policy, final_status="correct" if result["correct"] else "incorrect",
            validation_status="scored",
        )
    stages = trace.stages
    elapsed = next((stage.latency_ms for stage in stages if stage.stage == "investigation"),
                   trace.total_latency_ms)
    return {
        "arm": arm, "case": case, "type": task_type, "policy": policy, **result,
        "latency_ms": elapsed,
        "sql_calls": sum(s.stage == "sql_execution" for s in stages),
        "analysis_actions": sum(s.stage == "investigation_measure" for s in stages),
        "requests": online.requests - first_request if online else 0,
        "tokens": trace.total_tokens if trace.llm_calls else 0,
    }


def _percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def summarize(records):
    groups = {}
    for arm in sorted({record["arm"] for record in records}):
        whole = [record for record in records if record["arm"] == arm]
        for kind in ["all", *sorted({record["type"] for record in whole})]:
            runs = whole if kind == "all" else [r for r in whole if r["type"] == kind]
            key = f"{arm}/{kind}"
            groups[key] = {
                "n": len(runs), "correct": sum(r["correct"] for r in runs),
                "accuracy": sum(r["correct"] for r in runs) / len(runs),
                "errors_accepted": sum(r.get("error_accepted", False) for r in runs),
                "appropriate_refusals": sum(r.get("should_refuse", False) and not r.get("accepted")
                                            for r in runs),
                "sql_calls": sum(r["sql_calls"] for r in runs),
                "analysis_actions": sum(r["analysis_actions"] for r in runs),
                "requests": sum(r["requests"] for r in runs),
                "tokens": sum(r["tokens"] for r in runs) if all(r["tokens"] is not None for r in runs) else None,
                "p50_ms": round(_percentile([r["latency_ms"] for r in runs], .50), 2),
                "p95_ms": round(_percentile([r["latency_ms"] for r in runs], .95), 2),
                "failures": [r["case"] for r in runs if not r["correct"]],
            }
            buckets = groups[key]
            if any("error_accepted" in r for r in runs):
                query_n = sum("error_accepted" in r for r in runs)
                buckets["error_acceptance_rate"] = buckets["errors_accepted"] / query_n
            for cap in ("3", "5"):
                prefix = [r["equal_budget_prefixes"][cap] for r in runs if "equal_budget_prefixes" in r]
                if prefix:
                    groups[key][f"correct_at_{cap}_actions"] = [sum(row["correct"] for row in prefix), len(prefix)]
            for name in ("top1", "top3", "all_mechanisms_top3"):
                scores = [r[name] for r in runs if r.get(name) is not None]
                if scores:
                    groups[key][name] = [sum(scores), len(scores)]
            for name in ("claims", "supported_claims", "ungrounded_claims", "control_false_attributions",
                         "off_target_injection_claims", "replan_opportunities", "effective_replans"):
                if any(name in r for r in runs):
                    groups[key][name] = sum(r.get(name, 0) for r in runs)
            for numerator, denominator, label in (
                ("supported_claims", "claims", "evidence_support_rate"),
                ("ungrounded_claims", "claims", "ungrounded_claim_rate"),
                ("effective_replans", "replan_opportunities", "effective_replan_rate"),
            ):
                if denominator in buckets:
                    buckets[label] = buckets[numerator] / buckets[denominator] if buckets[denominator] else None
    return groups



def bike_smoke():
    database = Path("data/processed/bike.duckdb")
    source = Path("data/raw/hour.csv")
    periods = {"2012-06": [], "2012-07": []}
    workday = 0
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            month = row["dteday"][:7]
            if month in periods:
                periods[month].append(int(row["cnt"]))
            if month == "2012-06" and row["workingday"] == "1":
                workday += int(row["cnt"])
    query = answer_contract(database, "rentals from 2012-06-01 to 2012-07-01 where workingday=1", "bike")
    question = "why average_hourly_rentals from 2012-07-01 to 2012-08-01 versus 2012-06-01 to 2012-07-01"
    goal = parse_comparison(question, "bike")
    state = investigate((database, database), goal)
    expected = tuple(sum(values) / len(values) for values in periods.values())
    headline = state.evidence[0].segments
    correct = bool(query.result and query.result.rows == ((workday,),) and headline
                   and equivalent(((headline[0].before, headline[0].after),), (expected,))
                   and all(evidence_reconciles(e, expected[1] - expected[0]) for e in state.evidence))
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"dataset": "UCI Bike Sharing hour.csv", "source_sha256": digest,
            "correct": correct, "independent_workday_total": workday,
            "independent_hourly_means": expected, "query": asdict(query), "investigation": report(state)}

def evaluate(database: Path, *, split: str = "test", case_id: str | None = None, online=None, bike: bool = False):
    gold = cases.GOLD_TEST if split == "test" else cases.GOLD_DEV
    investigations = cases.INV_TEST if split == "test" else cases.INV_DEV
    runs = []
    incomplete = []
    schema = inspect_schema(database)

    def room_for(requests):
        return not online or (not online.exhausted and online.requests + requests <= online.max_requests)

    def keep_case(case, records):
        if online and online.exhausted:
            incomplete.append({"case": case.id, "reason": "incomplete online pairing", "partial_runs": records})
        else:
            runs.extend(records)
    for i, case in enumerate(gold):
        if case_id and case.id != case_id:
            continue
        if not room_for(6):
            incomplete.append({"case": case.id, "reason": "request budget for paired comparison"})
            continue
        arms = QUERY_ARMS[i % len(QUERY_ARMS):] + QUERY_ARMS[:i % len(QUERY_ARMS)]
        with duckdb.connect(str(database), read_only=True) as connection:
            expected = connection.execute(case.gold).fetchall() if case.gold else None
        records = [_measure_run(arm, case.id, "query", "query", lambda: run_gold(
            database, case, arm, online, schema, expected,
        ), online) for arm in arms]
        keep_case(case, records)
    with TemporaryDirectory(prefix="data-analysis-eval-", dir=database.parent) as temporary:
        for i, case in enumerate(investigations):
            if case_id and case.id != case_id:
                continue
            if not room_for(3 if case.dimension else 18):
                incomplete.append({"case": case.id, "reason": "request budget for paired comparison"})
                continue
            records = []
            database_after, manifest = cases.snapshot(database, case, Path(temporary) / "observed.duckdb")
            comparison = parse_comparison(case.question, "olist")
            arms = INV_ARMS[i % len(INV_ARMS):] + INV_ARMS[:i % len(INV_ARMS)]
            for arm in arms:
                policy = route_question(case.question, "olist") if arm == "auto" else arm
                kind = "fixed" if case.dimension else "open"
                limit = 1 if policy == "a_only" else 7
                selected = "scan" if policy == "a_only" else policy
                def run_investigation():
                    state = investigate((database, database_after), comparison, policy=selected,
                                        budget=limit, planner=online.plan if online else None)
                    scored = score_investigation(state, manifest, case.control)
                    if kind == "open" and arm != "a_only":
                        scored["equal_budget_prefixes"] = {}
                        for cap in (3, 5):
                            prefix = state
                            if len(state.evidence) > cap:
                                prefix = replace(state, evidence=state.evidence[:cap],
                                                 trace=state.trace[:cap], stop_reason="budget_exhausted",
                                                 remaining_budget=0)
                                prefix.hypotheses = assess(prefix)
                            score = score_investigation(prefix, manifest, case.control)
                            scored["equal_budget_prefixes"][str(cap)] = {
                                key: score[key] for key in ("correct", "top1", "top3", "effective_replans")
                            }
                    return scored
                records.append(_measure_run(arm, case.id, kind, policy, run_investigation, online))
            keep_case(case, records)
            if database_after != database:
                database_after.unlink()
    if not runs and not incomplete:
        raise ValueError(f"No tasks matching {case_id!r} in split {split}")
    with database.open("rb") as source:
        database_sha = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "split": split, "dataset": "Kaggle olistbr/brazilian-ecommerce, version 2",
        "database_sha256": database_sha, "duckdb": duckdb.__version__,
        "suite_sha256": hashlib.sha256(Path(cases.__file__).read_bytes()).hexdigest(),
        "model": online.name if online else None,
        "online_status": ("partial" if incomplete else "executed" if online else "not_requested"),
        "online_skipped_cases": incomplete,
        "online_usage": ({"requests": online.requests, "measured_tokens": online.tokens,
                          "unknown_usage_requests": online.usages.count(None)} if online else None),
        "date": date.today().isoformat(), "bike_smoke": bike_smoke() if bike else None,
        "summary": summarize(runs), "runs": runs,
    }
