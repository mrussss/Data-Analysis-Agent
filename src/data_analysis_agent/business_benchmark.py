"""Score business reports against fixed answers and persist reproducible paired traces."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

from . import business_cases
from .business_investigation import event_findings
from .investigation import investigate


ARMS = ("workflow", "rule_adaptive", "llm_frozen", "llm_adaptive")


def score(state, case):
    observed = event_findings(state.goal, state.evidence)
    valid = {(evidence, fact) for evidence, facts in observed.items() for fact in facts}
    claims = {(item["evidence"], item["fact"]) for item in state.claims}
    supported = claims & valid
    facts = {fact for _, fact in supported}
    # Complete global coverage with every snapshot in stock proves the same
    # for any SKU observed in both periods. A stable aggregate rate alone cannot.
    stock = next((item for item in state.evidence
                  if item.action == ("stock",) and (item.id, "stock:availability=stable") in supported), None)
    scope = next((item for item in state.evidence if item.action == ("scope",)), None)
    if (stock and scope and {row[0] for row in stock.rows} == {0, 1}
            and all(rate == 1 and recorded == total for _, rate, recorded, total, *_ in stock.rows)):
        for sku in ("core", "plus"):
            if (scope.id, f"scope:sku={sku}") in supported and {
                row[0] for row in scope.rows if row[1:3] == ("sku", sku)
            } == {0, 1}:
                facts.add(f"stock_{sku}:availability=stable")
    required = set(case.findings)
    missing = sorted(required - facts)
    obtained = {fact.split("=")[0] for choices in observed.values() for fact in choices}
    events = {fact for fact in required if fact.startswith("events:")}
    competing = {fact for fact in required if fact.endswith(("=stable", "=wrong_subject", "=after", "=missing"))
                 and not fact.startswith("headline:")}
    completed = state.verdict is not None and not state.model_error and state.remaining_budget >= 0
    error = any(item.error for item in state.evidence)
    attribution = state.verdict in {"causal", "observed_constraint", "plausible", "ambiguous"}
    wrong_attribution = bool(attribution and state.verdict != case.verdict)
    unsupported_attribution = bool(attribution and not required <= facts)
    scores = {
        "correct": bool(completed and not error and not missing and claims == supported
                        and state.verdict == case.verdict),
        "statistical_localization": required <= facts if case.kind == "A" else None,
        "mechanism_identified": (state.verdict == "observed_constraint"
                                 and "stock:stockout_sales_loss=present" in facts)
        if case.verdict == "observed_constraint" else None,
        "event_retrieval": all(fact.split("=")[0] in obtained for fact in events) if events else None,
        "event_matching": events <= facts if events else None,
        "competing_evidence": competing <= facts if competing else None,
        "correct_non_attribution": state.verdict == case.verdict and required <= facts
        if case.verdict in {"insufficient", "ambiguous", "stable"} else None,
        "wrong_attribution": wrong_attribution,
        "unsupported_attribution": unsupported_attribution,
        "completed": completed,
        "supported_claims": len(supported),
        "claims": len(claims),
        "required_found": len(required & facts),
        "required_total": len(required),
        "missing_findings": missing,
        "unsupported_claims": sorted(claims - supported),
        "failure": state.model_error or ("tool_error" if error else
                   "insufficient_findings" if missing else "unsupported_claim" if claims != supported
                   else "verdict_mismatch" if state.verdict != case.verdict else None),
        "rule_path_changes": sum(step.get("planned") is not None and step["planned"] != step["action"]
                                 for step in state.trace if step.get("decision_source") == "rule_adaptive"),
        "refutations": sum(item.state == "refuted" for item in state.hypotheses),
        "changes_after_refutation": sum(step.get("changed_after_refutation", False) for step in state.trace),
    }
    return scores


def _model_plan(model):
    def plan(prompt):
        first = len(model.planner_calls)
        try:
            return model(prompt, output_limit=1024, planning=True)
        finally:
            if len(model.planner_calls) > first:
                model.planner_calls[-1]["response_status"] = "content" if (
                    model.planner_calls[-1].get("finish_reason") == "stop"
                    and not model.planner_calls[-1].get("blank")
                ) else "provider_failure"
    return plan


def _run(database, events, case, arm, budget, model):
    index = model.requests if model else 0
    started = perf_counter()
    state = investigate((database, database), case.goal, policy=arm, budget=budget,
                        planner=_model_plan(model) if model else None, events=events)
    latency_ms = (perf_counter() - started) * 1000
    calls = model.planner_calls[index:] if model else []
    usage = model.usages[index:] if model else []
    costs = [call.get("provider_cost_usd") for call in calls]
    return {
        "case": case.id, "type": case.kind, "arm": arm, **score(state, case),
        "execution": "online_model" if usage else "model_unavailable" if model else "deterministic",
        "latency_ms": round(latency_ms, 2), "requests": len(usage),
        "tokens": None if None in usage else sum(usage),
        "cost_usd": None if None in costs else sum(costs),
        "actions": len(state.evidence), "sql_calls": sum(bool(sql) for item in state.evidence for sql in item.sql),
        "event_calls": sum(item.action == ("events",) for item in state.evidence),
        "response_diagnostics": calls, "details": asdict(state),
        "gold": {"verdict": case.verdict, "findings": case.findings},
    }


def summarize(rows):
    fields = ("correct", "statistical_localization", "mechanism_identified", "event_retrieval",
              "event_matching", "competing_evidence", "wrong_attribution", "unsupported_attribution",
              "correct_non_attribution", "completed")
    output = {}
    for kind in "ABCD":
        for arm in ARMS:
            group = [row for row in rows if row["type"] == kind and row["arm"] == arm]
            if not group:
                continue
            rates = {}
            for key in fields:
                applicable = [row[key] for row in group if row[key] is not None]
                rates[key] = {"hits": sum(applicable), "n": len(applicable)}
            claims = sum(row["claims"] for row in group)
            output.setdefault(kind, {})[arm] = {
                "n": len(group), **rates, "mean_ms": round(mean(row["latency_ms"] for row in group), 2),
                "actions": sum(row["actions"] for row in group),
                "sql_calls": sum(row["sql_calls"] for row in group),
                "event_calls": sum(row["event_calls"] for row in group),
                "requests": sum(row["requests"] for row in group),
                "tokens": None if any(row["tokens"] is None for row in group) else sum(row["tokens"] for row in group),
                "cost_usd": None if any(row["cost_usd"] is None for row in group) else sum(row["cost_usd"] for row in group),
                "evidence_support": {"hits": sum(row["supported_claims"] for row in group), "n": claims},
                "required_coverage": {"hits": sum(row["required_found"] for row in group),
                                      "n": sum(row["required_total"] for row in group)},
                "refutations": sum(row["refutations"] for row in group),
                "changes_after_refutation": sum(row["changes_after_refutation"] for row in group),
                "rule_path_changes": sum(row["rule_path_changes"] for row in group),
            }
    return output


def _digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate_business(database: Path, *, split: str, case_ids: tuple[str, ...] = (), budget: int = 5,
                      model=None):
    suite = business_cases.TEST if split == "test" else business_cases.DEV
    designated = business_cases.ONLINE_PAIRS if split == "test" else business_cases.ONLINE_DEV
    selected = case_ids or (designated if model else tuple(case.id for case in suite))
    choices = {case.id: case for case in suite}
    if len(set(selected)) != len(selected) or set(selected) - set(designated if model else choices):
        raise ValueError("Choose unique case IDs for this split and execution mode")
    if model and (model.name != "deepseek-v4.1-flash" or budget != 5):
        raise ValueError("The online protocol requires deepseek-v4.1-flash and five actions")
    rows, unpaired = [], []
    with TemporaryDirectory(prefix="mode-b-business-") as folder:
        for case_id in selected:
            case = choices[case_id]
            if model and (model.exhausted or model.requests + budget + 2 > model.max_requests):
                unpaired.append({"case": case.id, "reason": "No remaining budget to start both model arms"})
                continue
            snapshot, events = database, ()
            if case.goal.domain == "retail":
                snapshot = Path(folder) / f"{len(rows)}.duckdb"
                events = business_cases.build_retail(case, snapshot)
            for arm in ARMS[:2]:
                rows.append(_run(snapshot, events, case, arm, budget, None))
            if not model:
                continue
            # Alternate model order by fixed suite index, independent of case selection.
            order = ARMS[2:] if suite.index(case) % 2 == 0 else ARMS[2:][::-1]
            for arm in order:
                if model.exhausted:
                    unpaired.append({"case": case.id, "arm": arm, "reason": "Token budget or unknown provider usage"})
                else:
                    rows.append(_run(snapshot, events, case, arm, budget, model))
    sources = ("business_cases.py", "business_tools.py", "business_investigation.py", "business_benchmark.py",
               "investigation.py", "model_budget.py")
    return {
        "split": split, "budget": budget, "execution": "online" if model else "offline",
        "model": model.name if model else None, "designated_online_cases": designated,
        "database_sha256": _digest(database),
        "protocol_sha256": {name: _digest(Path(__file__).with_name(name)) for name in sources},
        "online_usage": {"requests": model.requests, "tokens": model.tokens,
                         "unknown_usage_requests": model.usages.count(None)} if model else None,
        "skipped_or_partial": unpaired, "summary": summarize(rows), "runs": rows,
    }
