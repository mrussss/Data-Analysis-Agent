"""One bounded investigation loop. Tools return facts; policies choose actions."""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class Hypothesis:
    action: tuple[str, ...]
    segment: tuple[str, ...]
    effect: str
    state: str
    priority: float
    contribution: float
    evidence: tuple[str, ...]


@dataclass
class InvestigationState:
    goal: Comparison
    available_tools: tuple[tuple[str, ...], ...]
    remaining_budget: int
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    model_error: str | None = None

    @property
    def reported(self) -> list[Hypothesis]:
        if self.stop_reason in {"insufficient_evidence", "no_material_change"}:
            return []
        return [item for item in self.hypotheses if item.state == "supported"][:3]


@observed_stage("investigation_measure")
def measure(databases: tuple[Path, Path], goal: Comparison, action: tuple[str, ...],
            evidence_id: str) -> Evidence:
    contracts = (goal.baseline, goal.current)
    sql = tuple(aggregate_sql(contract, action, sufficient_statistics=True) for contract in contracts)
    results = [run_readonly_sql(db, statement, max_rows=300) for db, statement in zip(databases, sql)]
    if any(result.error or result.truncated for result in results):
        error = next(("Truncated evidence" if result.truncated else result.error.message
                      for result in results if result.error or result.truncated))
        return Evidence(evidence_id, action, sql, (), error)
    by_period = [{tuple(row[:-2]): (float(row[-2] or 0), row[-1]) for row in result.rows}
                 for result in results]
    sizes = [sum(n for _, n in period.values()) for period in by_period]
    if min(sizes) < 24:
        return Evidence(evidence_id, action, sql, (), f"Only {sizes} usable observations")
    center = sum(sum(a for a, _ in period.values()) / size
                 for period, size in zip(by_period, sizes)) / 2
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
    return Evidence(evidence_id, action, sql, tuple(segments))


def assess(state: InvestigationState) -> list[Hypothesis]:
    """Support material effects and challenge marginal accounts with matched subdivisions."""
    if state.evidence[0].error:
        return []
    head = state.evidence[0].segments[0]
    scale = max(abs(head.before), abs(head.after), 1e-9)
    by_action = {e.action: e for e in state.evidence if not e.error}
    hypotheses = []
    for action, evidence in by_action.items():
        if not action:
            continue
        for row in evidence.segments:
            for effect in ("within", "mix"):
                value = getattr(row, effect)
                if value is None or abs(value) < scale * 0.015:
                    continue
                status = ("supported" if min(row.before_n, row.after_n) >= 24
                          or effect == "mix" else "tentative")
                ids = [evidence.id]
                score = abs(value)
                if len(action) == 1 and effect == "within" and state.goal.current.spec.mean:
                    for pair, split in by_action.items():
                        if len(pair) != 2 or action[0] not in pair:
                            continue
                        offset = pair.index(action[0])
                        children = [child for child in split.segments
                                    if child.key[offset] == row.key[0] and
                                    min(child.before_n, child.after_n) >= 24]
                        if len(children) < 2:
                            continue
                        shifts = [child.after - child.before for child in children]
                        if max(shifts) - min(shifts) > max(scale * 0.25,
                                                          abs(row.after - row.before) * 1.5):
                            status = "refuted"
                            ids.append(split.id)
                if len(action) == 2:
                    # Prefer the simpler account when the conditional shift is similar.
                    score *= 0.8
                    if effect == "within" and state.goal.current.spec.mean and row.before_n and row.after_n:
                        shift = row.after - row.before
                        for index, dim in enumerate(action):
                            parent = by_action.get((dim,))
                            parent_row = next((r for r in parent.segments
                                               if r.key == (row.key[index],)), None) if parent else None
                            if parent_row and parent_row.before_n and parent_row.after_n:
                                parent_shift = parent_row.after - parent_row.before
                                if math.isclose(shift, parent_shift, abs_tol=scale * 0.20):
                                    status = "redundant"
                hypotheses.append(Hypothesis(action, row.key, effect, status, score, value, tuple(ids)))
    return sorted(hypotheses, key=lambda item: (-item.priority, item.action, item.segment, item.effect))


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


@observed_stage("investigation")
def investigate(databases: tuple[Path, Path], goal: Comparison, *, policy: str = "adaptive",
                budget: int = 7, planner=None) -> InvestigationState:
    singles = [(dimension,) for dimension in goal.dimensions]
    pairs = list(itertools.combinations(goal.dimensions, 2))
    actions = [(), *singles, *pairs]
    if policy == "workflow":
        actions = [(), *singles[:2], *pairs[:1], *singles[2:], *pairs[1:]]
    state = InvestigationState(goal, tuple(actions), budget)
    planned_before_evidence = None
    while state.remaining_budget and actions:
        default = actions[0]
        chosen = default
        if policy == "adaptive" and state.evidence:
            latest = state.evidence[-1]
            chosen = _next_action(state, actions, latest.action, latest.id)
        prediction = None
        if planner is not None and state.evidence and policy in {"adaptive", "frozen"}:
            latest = state.evidence[-1]
            observations = None if policy == "frozen" else {
                "id": latest.id, "action": latest.action, "error": latest.error,
                "segments": [asdict(segment) for segment in sorted(
                    latest.segments, key=lambda row: -abs((row.within or 0) + (row.mix or 0)),
                )[:8]],
                "hypotheses": [asdict(item) for item in state.hypotheses[:12]],
            }
            try:
                proposal = planner(json.dumps({
                    "goal": asdict(goal), "available": actions, "recent_evidence": observations,
                    "rule_choice": chosen,
                }, default=str))
                proposed = tuple(proposal.get("action", ()))
                if proposed not in actions:
                    raise ValueError("Planner returned an unavailable action")
                chosen = proposed
                if (isinstance(proposal.get("segment"), list)
                        and len(proposal["segment"]) == len(chosen)
                        and all(isinstance(value, str) for value in proposal["segment"])
                        and proposal.get("effect") in ("within", "mix")
                        and proposal.get("direction") in (-1, 1)):
                    prediction = {key: proposal[key] for key in ("segment", "effect", "direction")}
            except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
                state.model_error = str(exc)
        state.trace.append({"action": chosen, "default": default,
                            "planned_before_evidence": planned_before_evidence,
                            "after": state.evidence[-1].id if state.evidence else None})
        actions.remove(chosen)
        planned_before_evidence = ((_next_action(state, actions, chosen)
                                    if policy == "adaptive" else actions[0]) if actions else None)
        previous = {(h.action, h.segment, h.effect): h.state for h in state.hypotheses}
        evidence = measure(databases, goal, chosen, f"E{len(state.evidence) + 1}")
        state.evidence.append(evidence)
        if prediction is not None:
            target = next((row for row in evidence.segments
                           if row.key == tuple(prediction["segment"])), None)
            value = getattr(target, prediction["effect"]) if target else None
            state.trace[-1]["model_hypothesis"] = {
                **prediction, "evidence": evidence.id,
                "status": ("insufficient" if value is None else "supported"
                           if value * prediction["direction"] > 0 else "refuted"),
                "observed_contribution": value,
            }
        state.remaining_budget -= 1
        state.hypotheses = assess(state)
        state.trace[-1]["updates"] = [
            {"action": h.action, "segment": h.segment, "effect": h.effect,
             "before": previous.get((h.action, h.segment, h.effect), "untested"),
             "after": h.state, "evidence": h.evidence}
            for h in state.hypotheses
            if previous.get((h.action, h.segment, h.effect)) != h.state
        ]
        if not chosen and evidence.error:
            state.stop_reason = "insufficient_evidence"
            break
    if not state.stop_reason:
        head = state.evidence[0].segments[0] if state.evidence and not state.evidence[0].error else None
        if head is None or (not state.hypotheses and any(e.error for e in state.evidence)):
            state.stop_reason = "insufficient_evidence"
        elif not actions and not state.reported and math.isclose(
                head.before, head.after, rel_tol=0.015, abs_tol=1e-9):
            state.stop_reason = "no_material_change"
        else:
            state.stop_reason = "budget_exhausted" if actions else "coverage_complete"
    return state


def report(state: InvestigationState) -> dict:
    return {
        **asdict(state),
        "facts": asdict(state.evidence[0].segments[0]) if state.evidence and state.evidence[0].segments else None,
        "statistical_attributions": [asdict(item) for item in state.reported],
        "unverified_explanations": ["Mechanisms outside the measured data require independent evidence."],
        "interpretation": "Observed contrasts and arithmetic contributions are descriptive.",
    }
