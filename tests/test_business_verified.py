import json

from data_analysis_agent.business_cases import DEEP_DEV, build_retail
from data_analysis_agent.business_verified import investigate_verified


def reply(action=None, *, gaps=(), hypotheses=None, verdict=None, claims=()):
    result = {"assessment": {"sufficient": not gaps, "gaps": list(gaps), "hypotheses": hypotheses or {}}}
    if action:
        result.update(action=action, question="Check the outstanding question.")
    if verdict:
        result["report"] = {"verdict": verdict, "claims": list(claims)}
    return result


def test_verified_rejects_stop_with_gap_and_revises_next_check(tmp_path):
    case = DEEP_DEV[1]
    db = tmp_path / "dev.duckdb"
    events = build_retail(case, db)
    plans = iter([
        reply("stop", gaps=["Where did units fall?"]),
        reply("scope", gaps=["Where did units fall?"], hypotheses={"Notice applies": {
            "status": "open", "evidence_ids": [],
        }}),
        reply("events", gaps=["Does the notice target the affected cohort?"], hypotheses={"Notice applies": {
            "status": "open", "evidence_ids": [],
        }}),
        reply("access_plus_north", gaps=["What moved within Plus North?"], hypotheses={"Notice applies": {
            "status": "refuted", "evidence_ids": ["E2", "E3"],
        }}),
        reply("stop", verdict="association", claims=[
            {"evidence": "E2", "fact": "scope:cell=plus_north"},
            {"evidence": "E3", "fact": "events:notice:scope=mismatch"},
            {"evidence": "E4", "fact": "access_plus_north:visits=down"},
        ]),
    ])
    prompts = []

    def plan(prompt):
        prompts.append(json.loads(prompt.split("\n", 1)[1]))
        return next(plans)

    state = investigate_verified(db, case.goal, budget=5, planner=plan, events=events)
    assert state.stop_reason == "evidence_sufficient"
    assert state.remaining_budget == 1
    assert state.verdict == "association"
    assert [item.action[0] for item in state.evidence] == ["headline", "scope", "events", "access_plus_north"]
    assert state.trace[1]["error"] == "inconsistent_next_action"
    transition = state.trace[-2]
    assert transition["decision_before"]["assessment"]["hypotheses"]["Notice applies"]["status"] == "open"
    assert transition["decision_after"]["assessment"]["hypotheses"]["Notice applies"]["status"] == "refuted"
    assert prompts[-1]["previous"] == transition["decision_after"]
    assert "plus_north" in prompts[0]["cohort_tool_families"]["access"]["remaining_cohorts"]
    assert "plus_north" not in prompts[-1]["cohort_tool_families"]["access"]["remaining_cohorts"]
    assert all("gold" not in prompt and "counterevidence" not in prompt for prompt in prompts)


def test_verified_at_budget_keeps_partial_claims_and_reports_insufficient(tmp_path):
    case = DEEP_DEV[1]
    db = tmp_path / "dev.duckdb"
    events = build_retail(case, db)
    plans = iter([
        reply("scope", gaps=["Where did units fall?"]),
        reply(gaps=["Does the notice explain the decline?"], verdict="association", claims=[
            {"evidence": "E2", "fact": "scope:cell=plus_north"},
        ]),
    ])
    state = investigate_verified(db, case.goal, budget=2, planner=lambda _: next(plans), events=events)
    assert state.stop_reason == "budget_exhausted"
    assert state.verdict == "insufficient"
    assert state.claims == [{"evidence": "E2", "fact": "scope:cell=plus_north"}]
    assert state.trace[-1]["decision_after"]["assessment"]["gaps"]


def test_verified_rejects_fabricated_citations(tmp_path):
    case = DEEP_DEV[0]
    db = tmp_path / "dev.duckdb"
    events = build_retail(case, db)
    state = investigate_verified(
        db, case.goal, budget=1, events=events,
        planner=lambda _: reply(verdict="association", claims=[{"evidence": "E1", "fact": "made_up=present"}]),
    )
    assert state.verdict == "insufficient"
    assert state.stop_reason == "planner_error"
    assert state.model_error == "unverified_report_claim"
    assert len(state.evidence) == 1
    assert not state.claims


def test_verified_exposes_tool_error_without_accepting_it_as_evidence(tmp_path):
    goal = DEEP_DEV[0].goal
    prompts = []

    def plan(prompt):
        prompts.append(json.loads(prompt.split("\n", 1)[1]))
        if len(prompts) == 1:
            return reply(gaps=["Could the observed sales establish a decline?"], verdict="insufficient",
                         hypotheses={"Sales declined": {"status": "supported", "evidence_ids": ["E1"]}})
        return reply(gaps=["Could the observed sales establish a decline?"], verdict="insufficient")

    state = investigate_verified(tmp_path / "missing.duckdb", goal, budget=1, planner=plan)
    assert prompts[0]["evidence"][0]["error"] == "database_missing"
    assert prompts[1]["correction"]["error"] == "invalid_hypothesis"
    assert state.stop_reason == "budget_exhausted"
    assert state.verdict == "insufficient"
