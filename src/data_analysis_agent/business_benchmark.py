"""Score business reports against fixed answers and persist reproducible paired traces."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from . import business_cases
from .business_investigation import event_findings
from .investigation import investigate
from .model_budget import PROJECT_ENV_PATH, PlannerResponseError, parse_model_json


ARMS = ("workflow", "rule_adaptive", "llm_frozen", "llm_adaptive")
BUSINESS_MAX_REQUESTS = 80
BUSINESS_MAX_TOKENS = 260_000
PLANNER_OUTPUT_LIMIT = 3072
BUSINESS_USAGE_PATH = PROJECT_ENV_PATH.parent / "data" / "processed" / ".business_online_usage.json"
USAGE_COUNTERS = ("requests", "tokens", "unknown_usage_requests", "reserved_unknown_tokens")


@contextmanager
def online_budget(model, paths, *, journal_path=BUSINESS_USAGE_PATH):
    """Import earlier results once and checkpoint every outbound attempt."""
    import fcntl

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another business online experiment is using this budget") from None
        journal = (json.loads(journal_path.read_text()) if journal_path.exists() else
                   {"id": uuid4().hex, "imported": [],
                    "usage": {**dict.fromkeys(USAGE_COUNTERS, 0), "unbounded_usage": False}, "seen_test": []})
        seen = set(journal["seen_test"])
        imported = set(journal["imported"])
        base = dict(journal["usage"])
        previous = []

        def save(usage):
            journal.update(usage=usage, imported=sorted(imported), seen_test=sorted(seen))
            pending = journal_path.with_suffix(".tmp")
            with pending.open("w") as handle:
                json.dump(journal, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            pending.replace(journal_path)

        for path in paths:
            raw = path.read_bytes()
            fingerprint = hashlib.sha256(raw).hexdigest()
            record = json.loads(raw)
            previous.append(record)
            usage = record.get("online_usage")
            if not isinstance(usage, dict):
                raise ValueError("--prior-results requires online experiment records")
            journal_id = record.get("budget_journal_id")
            if journal_id and journal_id != journal["id"]:
                raise ValueError("Prior result belongs to another budget journal")
            if not journal_id and fingerprint not in imported:
                if (usage.get("unbounded_usage") or (usage["unknown_usage_requests"] and
                        (usage.get("unbounded_usage", True) or not usage.get("reserved_unknown_tokens")))):
                    raise ValueError("Unbounded prior model usage requires reconciliation")
                for key in USAGE_COUNTERS:
                    value = usage.get(key, 0)
                    if type(value) is not int or value < 0:
                        raise ValueError("Invalid prior model usage counter")
                    base[key] += value
                imported.add(fingerprint)
            if record.get("split") == "test":
                seen.update(row["case"] for row in record["runs"] if row["requests"])

        if base["unbounded_usage"]:
            raise ValueError("Unbounded model usage requires reconciliation before restarting")
        model.max_requests = min(BUSINESS_MAX_REQUESTS, model.max_requests) - base["requests"]
        model.max_tokens = min(BUSINESS_MAX_TOKENS, model.max_tokens) - base["tokens"] - base["reserved_unknown_tokens"]

        def checkpoint(pending_tokens=0):
            current = model.usage()
            total = {key: base[key] + current[key] for key in USAGE_COUNTERS}
            if pending_tokens:
                total["unknown_usage_requests"] += 1
                total["reserved_unknown_tokens"] += pending_tokens
            total["unbounded_usage"] = current["unbounded_usage"]
            save(total)

        def begin_test(case_id):
            seen.add(case_id)
            save(journal["usage"])

        save(base)
        model.checkpoint_usage = checkpoint
        try:
            yield previous, seen, begin_test, journal
        finally:
            model.checkpoint_usage = None



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
        first = model.requests
        recovery = json.loads(prompt.split("\n", 1)[1]).get("correction")
        try:
            content = model(prompt, output_limit=PLANNER_OUTPUT_LIMIT, planning=True)
            try:
                decision = parse_model_json(content)
            except PlannerResponseError as exc:
                model.planner_calls[-1]["response_status"] = exc.kind
                raise
            model.planner_calls[-1]["response_status"] = "json"
            return decision
        finally:
            if recovery:
                for attempt in model.planner_calls[first:]:
                    attempt["recovery_for"] = recovery["error"]
                    attempt["recovery_phase"] = recovery["phase"]
    return plan


def _probe_low(model):
    first = model.requests
    outcome = "accepted"
    try:
        content = model('Return exactly this JSON object: {"ok":true}', output_limit=1024, planning=True)
        decision = parse_model_json(content)
        if set(decision) != {"ok"} or decision["ok"] is not True:
            outcome = "invalid_probe_response"
    except (RuntimeError, PlannerResponseError) as exc:
        outcome = getattr(exc, "kind", "budget_limit")
    attempts = model.planner_calls[first:]
    response = next((entry for entry in reversed(attempts) if "finish_reason" in entry), {})
    reported = response.get("reported_effort")
    return {"outcome": outcome, "attempts": attempts, "reported_effort": reported,
            "thinking_observed": bool(response.get("reasoning_chars") or response.get("reasoning_tokens")),
            "verification": "gateway_reported_low" if reported == "low"
            else "accepted_effective_effort_unverified" if outcome == "accepted" else "unverified"}


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
        "execution": ("deterministic" if not model else "online_model"
                      if any("finish_reason" in item for item in calls) else
                      "transport_or_provider_failure" if usage else "model_unavailable"),
        "latency_ms": round(latency_ms, 2), "requests": len(usage),
        "tokens": None if None in usage else sum(usage),
        "cost_usd": None if None in costs else sum(costs),
        "actions": len(state.evidence), "sql_calls": sum(bool(sql) for item in state.evidence for sql in item.sql),
        "event_calls": sum(item.action == ("events",) for item in state.evidence),
        "actual_models": sorted({call.get("reported_model") or call["model"] for call in calls
                                 if "finish_reason" in call}),
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
                      model=None, begin_test=None):
    suite = business_cases.TEST if split == "test" else business_cases.DEV
    designated = business_cases.ONLINE_PAIRS if split == "test" else tuple(case.id for case in suite)
    selected = case_ids or (designated if model else tuple(case.id for case in suite))
    choices = {case.id: case for case in suite}
    if len(set(selected)) != len(selected) or set(selected) - set(designated if model else choices):
        raise ValueError("Choose unique case IDs for this split and execution mode")
    if model and (model.name != "deepseek-v4.1-flash" or budget != 5):
        raise ValueError("The online protocol requires deepseek-v4.1-flash and five actions")
    rows, unpaired, paired = [], [], set()
    probe = _probe_low(model) if model and split == "dev" else None
    gate = not probe or probe["outcome"] == "accepted"
    with TemporaryDirectory(prefix="mode-b-business-") as folder:
        for case_id in selected:
            case = choices[case_id]
            if model and not gate:
                unpaired.append({"case": case.id, "reason": "Low gateway probe failed"})
                continue
            # Reserve the ordinary longest pair. Any network retries or
            # format correction still use the same hard request cap.
            reserve = budget + 2
            if model and (model.exhausted or model.requests + reserve > model.max_requests):
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
            if split == "test" and begin_test:
                begin_test(case_id)
            # Alternate model order by fixed suite index, independent of case selection.
            order = ARMS[2:] if suite.index(case) % 2 == 0 else ARMS[2:][::-1]
            first_model_row = len(rows)
            for arm in order:
                if model.exhausted:
                    unpaired.append({"case": case.id, "arm": arm, "reason": "Token budget or unknown provider usage"})
                else:
                    rows.append(_run(snapshot, events, case, arm, budget, model))
            reached = {row["arm"]: row["actual_models"] for row in rows[first_model_row:]
                       if row["execution"] == "online_model" and row["failure"] != "budget_limit"}
            if reached.keys() != set(ARMS[2:]):
                unpaired.append({"case": case.id, "reason": "Incomplete model pair"})
            elif (reached["llm_frozen"] != reached["llm_adaptive"]
                  or len(reached["llm_frozen"]) != 1):
                unpaired.append({"case": case.id, "reason": "Model versions differ within or between arms",
                                 "models": reached})
            else:
                paired.add(case.id)
    sources = ("business_cases.py", "business_tools.py", "business_investigation.py", "business_benchmark.py",
               "investigation.py", "model_budget.py", "deepseek_provider.py")
    return {
        "split": split, "budget": budget, "execution": "online" if model else "offline",
        "model": model.name if model else None, "designated_online_cases": designated,
        "database_sha256": _digest(database),
        "protocol_sha256": {name: _digest(Path(__file__).with_name(name)) for name in sources},
        "online_usage": model.usage() if model else None, "low_probe": probe,
        "paired_cases": sorted(paired),
        "paired_summary": summarize([row for row in rows if row["case"] in paired]),
        "skipped_or_partial": unpaired, "summary": summarize(rows), "runs": rows,
    }
