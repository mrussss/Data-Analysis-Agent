"""One bounded investigation loop. Tools return facts; policies choose actions."""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .business_tools import BusinessGoal

from .observability import observed_stage
from .query_contract import DIMENSIONS, QueryContract, aggregate_sql, parse_question
from .sql_executor import run_readonly_sql


@dataclass(frozen=True)
class Comparison:
    baseline: QueryContract
    current: QueryContract
    dimensions: tuple[str, ...]

    def __post_init__(self):
        if (self.baseline.metric, self.baseline.filters, self.baseline.grain) != (
            self.current.metric, self.current.filters, self.current.grain
        ) or self.current.grain != "total":
            raise ValueError("Comparisons require the same metric and filters at total grain")
        if not self.dimensions or len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Choose distinct investigation dimensions")
        if set(self.dimensions) - set(DIMENSIONS[self.current.spec.domain]):
            raise ValueError("Unknown investigation dimension")


def parse_comparison(question: str, domain: str) -> Comparison:
    match = re.fullmatch(
        r"(?:why|investigate|decompose) (.+?) versus (\d{4}-\d{2}-\d{2} to "
        r"\d{4}-\d{2}-\d{2})(?: by ([a-z_]+))?[?.]?", question.strip(), re.I,
    )
    if not match:
        raise ValueError("Use 'why <metric> from YYYY-MM-DD to YYYY-MM-DD versus YYYY-MM-DD to YYYY-MM-DD'")
    current_text, earlier, dimension = match.groups()
    baseline_text = re.sub(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", earlier, current_text)
    current = parse_question(current_text, domain)
    baseline = parse_question(baseline_text, domain)
    return Comparison(baseline, current, (dimension.lower(),) if dimension else DIMENSIONS[domain])


@dataclass(frozen=True)
class Segment:
    key: tuple[str, ...]
    before_n: int
    after_n: int
    before: float | None
    after: float | None
    within: float | None
    mix: float | None


@dataclass(frozen=True)
class Evidence:
    id: str
    action: tuple[str, ...]
    sql: tuple[str, str]
    segments: tuple[Segment, ...]
    error: str | None = None
    cohort: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Action:
    dimensions: tuple[str, ...] = ()
    cohort: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    action: tuple[str, ...]
    segment: tuple[str, ...]
    effect: str
    state: str
    priority: float
    contribution: float | None
    evidence: tuple[str, ...]
    cohort: tuple[tuple[str, str], ...] = ()
    id: str | None = None
    direction: str | None = None


@dataclass
class InvestigationState:
    goal: Comparison
    available_tools: tuple[str, ...]
    remaining_budget: int
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    model_error: str | None = None
    policy: str = "rule_adaptive"
    model_hypotheses: list[Hypothesis] = field(default_factory=list)

    @property
    def reported(self) -> list[Hypothesis]:
        if self.stop_reason in {"insufficient_evidence", "no_material_change", "planner_error"}:
            return []
        return [item for item in self.hypotheses if item.state == "supported"][:3]


@observed_stage("investigation_measure")
def measure(databases: tuple[Path, Path], goal: Comparison, action: tuple[str, ...],
            evidence_id: str, *, cohort: tuple[tuple[str, str], ...] = (),
            reference: Segment | None = None) -> Evidence:
    contracts = tuple(replace(contract, filters=contract.filters + cohort)
                      for contract in (goal.baseline, goal.current))
    sql = tuple(aggregate_sql(contract, action, sufficient_statistics=True) for contract in contracts)
    results = [run_readonly_sql(db, statement, max_rows=300) for db, statement in zip(databases, sql)]
    if any(result.error or result.truncated for result in results):
        error = next(("Truncated evidence" if result.truncated else result.error.message
                      for result in results if result.error or result.truncated))
        return Evidence(evidence_id, action, sql, (), error, cohort)
    by_period = [{tuple(row[:-2]): (float(row[-2] or 0), row[-1]) for row in result.rows}
                 for result in results]
    sizes = [sum(n for _, n in period.values()) for period in by_period]
    if min(sizes) < 24:
        return Evidence(evidence_id, action, sql, (), f"Only {sizes} usable observations", cohort)
    center = sum(sum(a for a, _ in period.values()) / size
                 for period, size in zip(by_period, sizes)) / 2
    if reference is not None:
        center = (reference.before + reference.after) / 2
        sizes = [reference.before_n, reference.after_n]
    segments = []
    for key in sorted(set(by_period[0]) | set(by_period[1])):
        (a, n), (b, m) = [period.get(key, (0, 0)) for period in by_period]
        if goal.current.spec.mean:
            before, after = a / n if n else None, b / m if m else None
            w, v = n / sizes[0], m / sizes[1]
            # An entering/exiting stratum has no observed within-stratum change.
            # Allocate its observed contribution to composition.
            midpoint = (before + after) / 2 if n and m else before if n else after if m else center
            mix = (v - w) * (midpoint - center)
            within = (after - before) * (v + w) / 2 if n and m else None
        else:
            before, after, within, mix = a, b, b - a, None
        segments.append(Segment(key, n, m, before, after, within, mix))
    return Evidence(evidence_id, action, sql, tuple(segments), cohort=cohort)


def assess(state: InvestigationState) -> list[Hypothesis]:
    """Support material effects and challenge marginal accounts with matched subdivisions."""
    if state.evidence[0].error:
        return []
    head = state.evidence[0].segments[0]
    scale = max(abs(head.before), abs(head.after), 1e-9)
    usable = [e for e in state.evidence if not e.error]
    by_action = {e.action: e for e in usable if not e.cohort}
    hypotheses = []
    for evidence in usable:
        fixed = dict(evidence.cohort)
        action = tuple(dim for dim in state.goal.dimensions if dim in fixed or dim in evidence.action)
        if not action:
            continue
        for row in evidence.segments:
            values = fixed | dict(zip(evidence.action, row.key))
            key = tuple(values[dim] for dim in action)
            for effect in ("within", "mix"):
                value = getattr(row, effect)
                if value is None or abs(value) < scale * 0.015:
                    continue
                status = ("supported" if min(row.before_n, row.after_n) >= 24
                          or effect == "mix" else "tentative")
                ids = [evidence.id]
                score = abs(value)
                if len(action) == 1 and effect == "within" and state.goal.current.spec.mean:
                    for split in usable:
                        coords = dict(split.cohort)
                        dimensions = set(split.action) | coords.keys()
                        if len(dimensions) != 2 or action[0] not in dimensions:
                            continue
                        children = [child for child in split.segments
                                    if (coords | dict(zip(split.action, child.key)))[action[0]] == key[0]
                                    and min(child.before_n, child.after_n) >= 24]
                        if len(children) < 2:
                            continue
                        shifts = [child.after - child.before for child in children]
                        if max(shifts) - min(shifts) > max(scale * 0.25,
                                                          abs(row.after - row.before) * 1.5):
                            status = "refuted"
                            ids.append(split.id)
                if len(action) >= 2:
                    # Prefer the simpler account when the conditional shift is similar.
                    score *= 0.8
                    if effect == "within" and state.goal.current.spec.mean and row.before_n and row.after_n:
                        shift = row.after - row.before
                        for index, dim in enumerate(action):
                            parent = by_action.get((dim,))
                            parent_row = next((r for r in parent.segments
                                               if r.key == (key[index],)), None) if parent else None
                            if parent_row and parent_row.before_n and parent_row.after_n:
                                parent_shift = parent_row.after - parent_row.before
                                if math.isclose(shift, parent_shift, abs_tol=scale * 0.20):
                                    status = "redundant"
                hypotheses.append(Hypothesis(action, key, effect, status, score, value, tuple(ids), evidence.cohort))
    distinct = {}
    for item in hypotheses:
        key = item.action, item.segment, item.effect
        if key not in distinct or distinct[key].cohort:
            distinct[key] = item
    return sorted(distinct.values(), key=lambda item: (-item.priority, item.action, item.segment, item.effect))


def _next_action(state: InvestigationState, pending: list[tuple[str, ...]],
                 latest_action: tuple[str, ...], latest_id: str | None = None) -> tuple[str, ...]:
    if not latest_action:
        return pending[0]
    contradicted = {h.action[0] for h in state.hypotheses
                    if h.state == "refuted" and h.evidence[-1] == latest_id}
    if contradicted:
        alternative = next((action for action in pending
                            if len(action) == 2 and contradicted.intersection(action)), None)
        if alternative:
            return alternative
    if len(latest_action) == 2:
        # After a corroborating split, check independent axes before splitting again.
        return pending[0]
    strongest = next((item for item in state.hypotheses
                      if len(item.action) == 1 and item.state == "supported"), None)
    if strongest:
        related = [action for action in pending if len(action) == 2 and strongest.action[0] in action]
        if related:
            return related[0]
    return pending[0]


def _observe(state: InvestigationState, databases: tuple[Path, Path], action: Action,
             step: dict) -> Evidence:
    previous = {(h.action, h.segment, h.effect): h.state for h in state.hypotheses}
    evidence = measure(databases, state.goal, action.dimensions, f"E{len(state.evidence) + 1}",
                       cohort=action.cohort, reference=state.evidence[0].segments[0] if action.cohort else None)
    state.evidence.append(evidence)
    state.remaining_budget -= 1
    state.hypotheses = assess(state)
    step.update({
        "action": action.dimensions, "cohort": dict(action.cohort), "evidence": evidence.id,
        "updates": [
            {"action": h.action, "segment": h.segment, "effect": h.effect,
             "before": previous.get((h.action, h.segment, h.effect), "untested"),
             "after": h.state, "evidence": h.evidence}
            for h in state.hypotheses if previous.get((h.action, h.segment, h.effect)) != h.state
        ],
    })
    state.trace.append(step)
    return evidence


def _rule_loop(state: InvestigationState, databases: tuple[Path, Path]) -> None:
    singles = [(dimension,) for dimension in state.goal.dimensions]
    pairs = list(itertools.combinations(state.goal.dimensions, 2))
    pending = ([*singles[:2], *pairs[:1], *singles[2:], *pairs[1:]]
               if state.policy == "workflow" else [*singles, *pairs])
    planned = pending[0] if pending else None
    while state.remaining_budget and pending:
        latest = state.evidence[-1]
        chosen = (_next_action(state, pending, latest.action, latest.id)
                  if state.policy == "rule_adaptive" else pending[0])
        pending.remove(chosen)
        next_plan = ((_next_action(state, pending, chosen) if state.policy == "rule_adaptive" else pending[0])
                     if pending else None)
        _observe(state, databases, Action(chosen), {
            "decision_source": "rule" if state.policy == "rule_adaptive" else "fixed",
            "after": latest.id, "planned_before_evidence": planned,
        })
        planned = next_plan
    state.stop_reason = ("budget_exhausted" if pending else
                         "no_material_change" if _no_material_change(state) else "coverage_complete")


def _no_material_change(state: InvestigationState) -> bool:
    head = state.evidence[0].segments[0]
    covered = {e.action for e in state.evidence if not e.error and not e.cohort}
    required = {*itertools.combinations(state.goal.dimensions, 2),
                *((dimension,) for dimension in state.goal.dimensions)}
    return (required <= covered and not state.reported
            and math.isclose(head.before, head.after, rel_tol=0.015, abs_tol=1e-9))


def _parse_action(payload: dict, goal: Comparison) -> Action:
    if not isinstance(payload, dict) or set(payload) - {"dimensions", "cohort"}:
        raise ValueError("action must contain only dimensions and optional cohort")
    dims, cohort = payload.get("dimensions"), payload.get("cohort", {})
    if (not isinstance(dims, list) or any(dim not in goal.dimensions for dim in dims)
            or len(set(dims)) != len(dims) or len(dims) > 2
            or not isinstance(cohort, dict) or len(cohort) > 1
            or any(dim not in goal.dimensions or dim in dims or not isinstance(value, str)
                   for dim, value in cohort.items()) or not (dims or cohort)):
        raise ValueError("Unsupported dimensions or cohort")
    filters = tuple(cohort.items())
    replace(goal.current, filters=goal.current.filters + filters)
    return Action(tuple(dim for dim in goal.dimensions if dim in dims), filters)


def _action_json(action: Action) -> dict:
    return {"dimensions": action.dimensions, "cohort": dict(action.cohort)}


def _observation(evidence: Evidence) -> dict:
    def important(row: Segment) -> float:
        return max(abs(row.within or 0), abs(row.mix or 0))

    rows = sorted(evidence.segments, key=important, reverse=True)[:10]
    return {"id": evidence.id, "action": evidence.action, "cohort": dict(evidence.cohort),
            "error": evidence.error, "segments": [asdict(row) for row in rows],
            "omitted_segments": len(evidence.segments) - len(rows)}


def _model_loop(state: InvestigationState, databases: tuple[Path, Path], planner) -> None:
    frozen = state.policy == "llm_frozen"
    history: list[dict] = []
    attempted: list[Action] = []
    correction = None
    # One opportunity to correct a malformed or repeated decision, using an
    # existing decision slot. It never expands the SQL or model request budget.
    for _ in range(state.remaining_budget):
        visible = state.evidence[:1] if frozen else state.evidence
        payload = {
            "goal": asdict(state.goal), "metric_definition": state.goal.current.spec.definition,
            "dimensions": state.goal.dimensions, "remaining_actions": state.remaining_budget,
            "evidence": [_observation(item) for item in visible],
            "already_executed": [_action_json(action) for action in attempted],
            "past_decisions": history,
            "tested_hypotheses": [] if frozen else [asdict(h) for h in state.model_hypotheses],
            "statistical_candidates": [] if frozen else [asdict(h) for h in state.hypotheses[:10]],
        }
        if correction:
            payload["previous_decision_error"] = correction
        failure = "invalid_schema"
        proposed_action = False
        try:
            decision = planner(json.dumps(payload, default=str))
            if not isinstance(decision, dict):
                raise ValueError("Planner must return an object")
            proposed_action = "action" in decision
            ids = decision.get("evidence_ids")
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(item, str) or item not in {e.id for e in visible} for item in ids)):
                raise ValueError("Decision must cite visible evidence")
            question = decision.get("question")
            if not isinstance(question, str) or not question.strip() or len(question) > 300:
                raise ValueError("Decision must state a short investigative question")
            step = {"decision_source": "model", "after": visible[-1].id, "evidence_ids": ids,
                    "question": question, "prior_plan": history[-1]["next_action"] if history else None,
                    "visible_evidence": [item.id for item in visible]}
            if "stop" in decision:
                reason = decision["stop"]
                if "action" in decision:
                    raise ValueError("Choose either stop or action")
                if reason not in ("insufficient_evidence", "no_material_change", "sufficient_evidence"):
                    raise ValueError("Unsupported stop reason")
                if reason == "sufficient_evidence" and any(
                    set(h.evidence) <= set(ids) for h in state.reported
                ):
                    state.stop_reason = reason
                elif reason == "no_material_change" and _no_material_change(state):
                    state.stop_reason = reason
                else:
                    state.stop_reason = "insufficient_evidence"
                state.model_error = None
                state.trace.append({**step, "stop_requested": reason, "stop_reason": state.stop_reason})
                return
            failure = "invalid_action"
            action = _parse_action(decision.get("action"), state.goal)
            if action in attempted:
                failure = "repeated_action"
                raise ValueError("Action already executed")
            failure = "invalid_schema"
            next_action = (_parse_action(decision["next_action"], state.goal)
                           if decision.get("next_action") is not None else None)
            if next_action == action or next_action in attempted:
                next_action = None  # A provisional action is not executed.
            direction = decision.get("expect")
            if direction not in (None, "increase", "decrease", "stable"):
                raise ValueError("Unsupported expectation")
            revises = decision.get("revises")
            if revises is not None and revises not in {f"H{i + 1}" for i in range(len(history))}:
                raise ValueError("Only an existing hypothesis ID can be revised")
        except (ValueError, TypeError, RuntimeError) as exc:
            kind = getattr(exc, "kind", failure)
            if isinstance(exc, RuntimeError) and "budget" in str(exc):
                kind = "budget_limit"
            state.model_error = str(exc)
            state.trace.append({"decision_source": "model", "error": state.model_error, "failure_kind": kind,
                                "proposed_action": proposed_action, "visible_evidence": [item.id for item in visible]})
            if correction is None and kind in {"repeated_action", "invalid_json", "empty_output"}:
                correction = kind
                continue
            state.stop_reason = "planner_error"
            return
        state.model_error = None
        hid = f"H{len(history) + 1}"
        attempted.append(action)
        hypothesis = Hypothesis(action.dimensions, (), "within", "pending", 0, None, tuple(ids),
                                action.cohort, hid, direction)
        state.model_hypotheses.append(hypothesis)
        original = next((h for h in state.model_hypotheses if h.id == revises), None)
        step.update({
            "hypothesis_id": hid, "hypothesis_before": "pending", "revises": revises,
            "next_action": _action_json(next_action) if next_action else None,
            "replan": False,
        })
        changed = bool(not frozen and step["prior_plan"] and _action_json(action) != step["prior_plan"]
                       and visible[-1].id in ids and len(visible) > 1)
        step["evidence_driven_change"] = changed
        step["replan"] = bool(
                changed and original
                and original.state == "refuted" and original.evidence[-1] == visible[-1].id in ids
                and (set(original.action) | dict(original.cohort).keys())
                    & (set(action.dimensions) | dict(action.cohort).keys())
        )
        evidence = _observe(state, databases, action, step)
        # Use actual group keys from the result. Cohort filters never become group keys.
        measured = [(row, effect, value) for row in evidence.segments
                    for effect, value in (("within", row.within), ("mix", row.mix))
                    if value is not None and (effect == "mix" or min(row.before_n, row.after_n) >= 24)
                    and (direction is None or effect == "within")]
        observed = max(measured, key=lambda item: abs(item[2]), default=None)
        if observed:
            row, effect, value = observed
            head = state.evidence[0].segments[0]
            cutoff = max(abs(head.before), abs(head.after), 1e-9) * 0.015
            actual = "increase" if value > cutoff else "decrease" if value < -cutoff else "stable"
            status = ("observed" if direction is None else "supported" if actual == direction else "refuted")
            hypothesis = replace(hypothesis, segment=row.key, effect=effect, contribution=value)
        else:
            status = "insufficient"
        state.model_hypotheses[-1] = replace(hypothesis, state=status, evidence=(*ids, evidence.id))
        step["hypothesis_after"] = status
        history.append({"hypothesis_id": hid, "action": _action_json(action), "question": question,
                        "expect": direction, "next_action": step["next_action"]})
    state.stop_reason = ("planner_error" if state.model_error else
                         "decision_budget_exhausted" if state.remaining_budget else "budget_exhausted")


@observed_stage("investigation")
def investigate(databases: tuple[Path, Path], goal: Comparison | BusinessGoal, *, policy: str = "rule_adaptive",
                budget: int = 7, planner=None, events=()) -> InvestigationState:
    policy = {"adaptive": "rule_adaptive", "frozen": "scan"}.get(policy, policy)
    if policy not in {"rule_adaptive", "workflow", "scan", "llm_adaptive", "llm_frozen"}:
        raise ValueError("Unknown investigation policy")
    if budget < 1 or policy.startswith("llm_") != (planner is not None):
        raise ValueError("A model planner is required only for LLM policies, with positive action budget")
    if not isinstance(goal, Comparison):
        from .business_investigation import run_business

        if databases[0] != databases[1]:
            raise ValueError("Business investigations compare windows in one snapshot")
        return run_business(databases[0], goal, policy=policy, budget=budget, planner=planner, events=events)
    state = InvestigationState(goal, ("compare", "decompose", "condition", "verify"), budget, policy=policy)
    headline = _observe(state, databases, Action(), {"decision_source": "shared_headline"})
    if headline.error:
        state.stop_reason = "insufficient_evidence"
    elif policy.startswith("llm_"):
        _model_loop(state, databases, planner)
    else:
        _rule_loop(state, databases)
    return state


def report(state: InvestigationState) -> dict:
    return {
        **asdict(state),
        "facts": asdict(state.evidence[0].segments[0]) if state.evidence and state.evidence[0].segments else None,
        "statistical_attributions": [asdict(item) for item in state.reported],
        "unverified_explanations": ["Mechanisms outside the measured data require independent evidence."],
        "interpretation": "Observed contrasts and arithmetic contributions are descriptive.",
    }
