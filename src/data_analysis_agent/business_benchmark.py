"""Score business reports against fixed answers and persist reproducible paired traces."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from itertools import combinations
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from . import business_cases
from .business_investigation import event_findings
from .business_tools import catalog, observe
from .sql_executor import run_readonly_sql
from .investigation import investigate
from .model_budget import PROJECT_ENV_PATH, PlannerResponseError, parse_model_json


ARMS = ("workflow", "rule_adaptive", "llm_frozen", "llm_adaptive")
BUSINESS_MAX_REQUESTS = 80
BUSINESS_MAX_TOKENS = 260_000
PLANNER_OUTPUT_LIMIT = 3072
DEEP_STUDY = "mode-b-deep-v1"
DEEP_FREEZE_PATH = PROJECT_ENV_PATH.parent / "evaluation" / "business_deep_frozen_manifest.json"
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


def _deep_found(case, facts):
    facts = set(facts)
    facts.update(canonical for canonical, substitute in case.alternatives if substitute in facts)
    return facts


def _deep_obtained(case, evidence):
    return _deep_found(case, (fact for facts in event_findings(case.goal, evidence).values() for fact in facts))


def score_deep(state, case):
    """Score collection, calibration, and cited outcome independently."""
    result = score(state, case)
    required = set(case.findings)
    observed = event_findings(state.goal, state.evidence)
    claims = _deep_found(case, (item["fact"] for item in state.claims
                                if item["fact"] in observed.get(item["evidence"], ())))
    collected = _deep_obtained(case, state.evidence)
    missing = sorted(required - collected)
    first = next((count for count in range(1, len(state.evidence) + 1)
                  if required <= _deep_obtained(case, state.evidence[:count])), None)
    expected_verdict = case.verdict if not missing else "insufficient"
    judgment = state.verdict == expected_verdict if result["completed"] else None
    correct = bool(result["completed"] and not missing and required <= claims and judgment
                   and result["claims"] == result["supported_claims"]
                   and not any(item.error for item in state.evidence))
    result.update(
        correct=correct, evidence_complete=not missing, judgment_correct=judgment,
        gold_verdict_match=state.verdict == case.verdict if result["completed"] else None,
        expected_verdict_from_observed=expected_verdict, missing_evidence=missing,
        missing_findings=sorted(required - claims), decisive_action=first,
        extra_actions_after_decisive=len(state.evidence) - first if first else None,
        counterevidence_found=set(case.counterevidence) <= collected if case.counterevidence else None,
        wrong_attribution=(state.verdict in {"causal", "observed_constraint", "plausible", "ambiguous"}
                           and state.verdict != expected_verdict),
        failure=None if correct else state.model_error or (
            "missing_evidence" if missing else "verdict_mismatch" if not judgment else "incomplete_citations"
        ),
    )
    return result


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
    deep = isinstance(case, business_cases.DeepCase)
    scores = score_deep(state, case) if deep else score(state, case)
    return {
        "case": case.id, "type": case.kind, "arm": arm, **scores,
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
        "gold": {"verdict": case.verdict, "findings": case.findings,
                 **({"alternatives": case.alternatives, "counterevidence": case.counterevidence} if deep else {})},
    }


def summarize(rows):
    fields = ("correct", "statistical_localization", "mechanism_identified", "event_retrieval",
              "event_matching", "competing_evidence", "wrong_attribution", "unsupported_attribution",
              "correct_non_attribution", "completed", "evidence_complete", "judgment_correct", "counterevidence_found")
    output = {}
    for kind in "ABCD":
        for arm in ARMS:
            group = [row for row in rows if row["type"] == kind and row["arm"] == arm]
            if not group:
                continue
            rates = {}
            for key in fields:
                if key in group[0]:
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


def _protocol_digests(*, deep=False):
    sources = ("business_cases.py", "business_tools.py", "business_investigation.py", "business_benchmark.py",
               "investigation.py", "model_budget.py", "deepseek_provider.py")
    if deep:
        sources += ("__main__.py", "sql_executor.py")
    return {name: _digest(Path(__file__).with_name(name)) for name in sources}


def _digest_value(value):
    return hashlib.sha256(json.dumps(value, default=str, separators=(",", ":")).encode()).hexdigest()


def _read_panel(path):
    result = run_readonly_sql(path, "SELECT * FROM retail ORDER BY day, entity, sku, region", max_rows=200)
    if result.error or result.truncated:
        raise ValueError("Unable to audit full deep retail snapshot")
    return result.rows


def audit_deep():
    """Check raw panel invariants and enumerate an optimistic fixed-set ceiling."""
    rows, coverage = {}, {3: {}, 5: {}}
    baseline_panels = {}
    with TemporaryDirectory(prefix="mode-b-deep-audit-") as folder:
        for case in (*business_cases.DEEP_DEV, *business_cases.DEEP_TEST):
            before_path = Path(folder) / f"{case.id}-base.duckdb"
            path = Path(folder) / f"{case.id}.duckdb"
            business_cases.build_retail(case, before_path, apply_changes=False)
            events = business_cases.build_retail(case, path)
            before, current = _read_panel(before_path), _read_panel(path)
            if len(before) != 168 or len(current) != len(before):
                raise ValueError(f"Unexpected panel size for {case.id}")
            cell_loss = {}
            for original, observed in zip(before, current):
                if (original[:4] != observed[:4]
                        or ((original[0] < case.goal.current[0] or original[1] != case.goal.entity)
                            and original != observed)
                        or not (0 <= observed[9] <= observed[4] and 0 <= observed[10] <= observed[4])):
                    raise ValueError(f"Invalid before/after rows for {case.id}")
                if observed[1] == case.goal.entity:
                    cell = "_".join(observed[2:4])
                    cell_loss[cell] = cell_loss.get(cell, 0) + original[4] - observed[4]
            affected = max(cell_loss, key=cell_loss.get)
            if (f"scope:cell={affected}" not in case.findings or sum(cell_loss.values()) <= 0
                    or list(cell_loss.values()).count(cell_loss[affected]) != 1):
                raise ValueError(f"Raw cell-loss audit disagrees with {case.id}")
            identifiability = None
            if case.verdict in {"insufficient", "ambiguous"}:
                raw_pairs = [(original, observed) for original, observed in zip(before, current)
                             if observed[1] == case.goal.entity and observed[0] >= case.goal.current[0]]
                affected_pairs = [(original, observed) for original, observed in raw_pairs
                                  if "_".join(observed[2:4]) == affected]
                if case.verdict == "insufficient":
                    if not all(row[7] is None and row[8] is None for _, row in affected_pairs):
                        raise ValueError("Missing inventory and channel coverage is not established")
                    identifiability = "Neither inventory nor channel observations exist in the affected cohort."
                else:
                    if not all(row[5] > original[5] and row[8] < original[8] for original, row in affected_pairs):
                        raise ValueError("Competing observations do not coincide for the full affected cohort")
                    if any(original != observed for original, observed in raw_pairs
                           if "_".join(observed[2:4]) != affected):
                        raise ValueError("An additional changed cohort can separate the observations")
                    identifiability = "Price and traffic change together across the affected cohort and period."
            # Persist the pre-injection values once. Entity names distinguish
            # otherwise identical case snapshots and are included in their hashes.
            baseline = tuple(sorted((row[0], "merchant" if row[1] == case.goal.entity else "peer", *row[2:])
                                    for row in before))
            baseline_id = _digest_value(baseline)
            baseline_panels[baseline_id] = baseline
            obs = {action: observe(path, case.goal, action, f"E{i}", events)
                   for i, action in enumerate(("headline", *catalog(case.goal)), 1)}
            if any(item.error for item in obs.values()):
                raise ValueError(f"Tool audit failed for {case.id}")
            required = set(case.findings)
            actions = tuple(action for action in obs if action != "headline")
            witness = None
            for size in range(1, 5):
                fixed_coverage = coverage.get(size + 1) if case in business_cases.DEEP_TEST else None
                if witness and fixed_coverage is None:
                    continue
                for batch in combinations(actions, size):
                    evidence = [obs[action] for action in ("headline", *batch)]
                    if required <= _deep_obtained(case, evidence):
                        if fixed_coverage is not None:
                            fixed_coverage.setdefault(tuple(sorted(batch)), set()).add(case.id)
                        if witness is None:
                            witness = batch
                        if fixed_coverage is None:
                            break
            if not witness:
                raise ValueError(f"No evidence path within five actions for {case.id}")
            rows[case.id] = {"before_sha256": _digest_value(before), "baseline_panel": baseline_id,
                             "observed_sha256": _digest_value(current), "events_sha256": _digest_value(events),
                             "cell_unit_loss": cell_loss, "changes": case.changes, "verdict": case.verdict,
                             "findings": case.findings, "alternatives": case.alternatives,
                             "identifiability": identifiability, "minimum_actions": 1 + len(witness), "example": ["headline", *witness]}
    return {"study_id": DEEP_STUDY, "protocol_sha256": _protocol_digests(deep=True),
            "online_order": business_cases.DEEP_ONLINE_ORDER, "cases": rows,
            "baseline_panels": baseline_panels,
            "fixed_set_bound": {budget: _fixed_set_bound(candidates) for budget, candidates in coverage.items()}}


def _fixed_set_bound(candidates):
    if not candidates:
        return {"success": 0, "one_set": [], "cases": []}
    actions, cases = min(candidates.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    return {"success": len(cases), "one_set": ["headline", *actions], "cases": sorted(cases)}


def evaluate_business(database: Path, *, split: str, case_ids: tuple[str, ...] = (), budget: int = 5,
                      model=None, begin_test=None, deep=False):
    freeze = None
    if deep:
        suite = business_cases.DEEP_TEST if split == "test" else business_cases.DEEP_DEV
        designated = business_cases.DEEP_ONLINE_ORDER if split == "test" else tuple(case.id for case in suite)
        if model:
            freeze = json.loads(DEEP_FREEZE_PATH.read_text())
            if (freeze["study_id"] != DEEP_STUDY or freeze["protocol_sha256"] != _protocol_digests(deep=True)
                    or tuple(freeze["online_order"]) != business_cases.DEEP_ONLINE_ORDER):
                raise ValueError("Deep suite differs from the frozen manifest")
    else:
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
                if freeze:
                    record = freeze["cases"][case.id]
                    if (record["observed_sha256"] != _digest_value(_read_panel(snapshot))
                            or record["events_sha256"] != _digest_value(events)):
                        raise ValueError(f"Deep data snapshot differs from the frozen manifest for {case.id}")
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
    return {
        **({"study_id": DEEP_STUDY, "freeze_sha256": _digest(DEEP_FREEZE_PATH) if freeze else None} if deep else {}),
        "split": split, "suite": "deep" if deep else "legacy", "budget": budget,
        "execution": "online" if model else "offline",
        "model": model.name if model else None, "designated_online_cases": designated,
        "database_sha256": _digest(database),
        "protocol_sha256": _protocol_digests(deep=deep),
        "online_usage": model.usage() if model else None, "low_probe": probe,
        "paired_cases": sorted(paired),
        "paired_summary": summarize([row for row in rows if row["case"] in paired]),
        "skipped_or_partial": unpaired, "summary": summarize(rows), "runs": rows,
    }
