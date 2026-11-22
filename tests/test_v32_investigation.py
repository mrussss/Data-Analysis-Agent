import json

import duckdb
import pytest

from data_analysis_agent.benchmark import effective_replans, score_investigation
from data_analysis_agent.benchmark_cases import Case, counts, snapshot
from data_analysis_agent.investigation import investigate, parse_comparison, report
from data_analysis_agent.routing import route_question


WINDOW = ("2018-01-01", "2018-01-08")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "orders.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE customers AS
                SELECT 'SP' AS customer_id, 'SP' AS customer_state UNION ALL SELECT 'RJ', 'RJ';
            CREATE TABLE orders AS
                SELECT CAST(i AS VARCHAR) AS order_id, IF((i // 7) % 2 = 0, 'SP', 'RJ') AS customer_id,
                       DATE '2018-01-01' + (i % 7)::INTEGER AS order_purchase_timestamp,
                       'delivered' AS order_status,
                       DATE '2018-01-13' + (i % 5 = 0)::INT AS order_delivered_customer_date,
                       DATE '2018-01-13' AS order_estimated_delivery_date,
                       DATE '2018-01-08' AS order_approved_at,
                       DATE '2018-01-08' AS order_delivered_carrier_date
                FROM range(1400) AS t(i);
        """)
    return path


def choice(dimensions, *, evidence="E1", cohort=None, expect=None,
           next_action=None, revises=None):
    return {
        "action": {"dimensions": dimensions, "cohort": cohort or {}}, "expect": expect,
        "evidence_ids": [evidence], "question": "Which population accounts for the change?",
        "next_action": next_action, "revises": revises,
    }


def test_model_changes_executed_action_after_refutation_without_seeing_manifest(source, tmp_path):
    case = Case("hidden-mechanism", WINDOW, WINDOW, (("SP", 3, 1, 65),))
    observed, manifest = snapshot(source, case, tmp_path / "observed.duckdb")
    assert counts(source, WINDOW) == (1400, 280)
    assert manifest["current"] == (1400, 345)
    goal = parse_comparison(case.question, "olist")
    planned = {"dimensions": ["status"]}
    inputs = []

    def planner(payload):
        data = json.loads(payload)
        inputs.append(data)
        assert "rule_choice" not in data and "hidden-mechanism" not in payload and "matched" not in payload
        if len(inputs) == 1:
            return choice(["state"], expect="stable", next_action=planned)
        if len(inputs) == 2:
            assert data["tested_hypotheses"][0]["state"] == "refuted"
            return choice(["weekday"], cohort={"state": "SP"}, evidence="E2", revises="H1")
        return {"stop": "sufficient_evidence", "evidence_ids": ["E3"],
                "question": "Can the localized shift account for the observed change?"}

    adaptive = investigate((source, observed), goal, policy="llm_adaptive", budget=6, planner=planner)
    assert [e.action for e in adaptive.evidence] == [(), ("state",), ("weekday",)]
    assert adaptive.evidence[-1].cohort == (("state", "SP"),)
    assert "state = 'SP'" in adaptive.evidence[-1].sql[0]
    assert adaptive.model_hypotheses[-1].segment == ("3",)
    assert adaptive.model_hypotheses[-1].state == "observed"
    assert effective_replans(adaptive) == (1, 1)
    scored = score_investigation(adaptive, manifest, None)
    assert scored["correct"] and scored["top1"] and scored["supported_claims"] == scored["claims"]
    assert "gold" not in report(adaptive)
    assert adaptive.stop_reason == "sufficient_evidence"
    assert adaptive.remaining_budget == 3


def test_frozen_only_sees_headline_and_own_actions(source):
    data = []

    def planner(payload):
        data.append(json.loads(payload))
        assert not data[-1]["tested_hypotheses"] and not data[-1]["statistical_candidates"]
        assert [e["id"] for e in data[-1]["evidence"]] == ["E1"]
        if len(data) == 1:
            return choice(["state"])
        assert len(data[-1]["past_decisions"]) == 1
        return choice([], cohort={"state": "RJ"})

    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    state = investigate((source, source), goal, policy="llm_frozen", budget=3, planner=planner)
    assert [step["evidence"] for step in state.trace] == ["E1", "E2", "E3"]
    assert state.evidence[-1].cohort == (("state", "RJ"),)
    assert effective_replans(state) == (0, 0)


@pytest.mark.parametrize("invalid", [
    choice(["state"], evidence="E999"),
    choice([], cohort={"state": "SP' OR TRUE --"}),
])
def test_invalid_model_action_stops_without_a_rule_fallback(source, invalid):
    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    state = investigate((source, source), goal, policy="llm_adaptive",
                        budget=3, planner=lambda _: invalid)
    assert len(state.evidence) == 1
    assert state.stop_reason == "planner_error"
    assert not state.reported


@pytest.mark.parametrize("case", [
    Case("normal", WINDOW, WINDOW, control="no_material_change"),
    Case("empty", ("2017-01-01", "2017-02-01"), WINDOW, control="insufficient_evidence"),
])
def test_unsupported_attribution_stops_without_claims(source, case):
    state = investigate((source, source), parse_comparison(case.question, "olist"))
    assert state.stop_reason == case.control
    assert state.reported == []


def test_route_uses_task_requirements():
    assert route_question("orders from 2018-01-01 to 2018-01-08", "olist") == "query"
    open_question = Case("test", WINDOW, WINDOW).question
    assert route_question(open_question, "olist") == "adaptive"
    assert route_question(open_question + " by weekday", "olist") == "workflow"


def test_repeated_action_can_be_corrected_in_the_existing_decision_budget(source):
    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    decisions = iter([choice(["state"]), choice(["state"]), choice(["weekday"])])
    attempts = []

    def planner(payload):
        attempts.append(json.loads(payload))
        return next(decisions)

    result = investigate((source, source), goal, policy="llm_adaptive", budget=4, planner=planner)
    assert [item.action for item in result.evidence] == [(), ("state",), ("weekday",)]
    assert attempts[-1]["previous_decision_error"] == "repeated_action"
    assert result.stop_reason == "decision_budget_exhausted"
    assert result.model_error is None
    assert len(attempts) == 3 and result.remaining_budget == 1


@pytest.mark.parametrize(("content", "finish", "status"), [
    ("   ", "stop", "empty_output"),
    ('{"action":', "length", "truncated_output"),
    ("not JSON", "stop", "invalid_json"),
    ('```json\n{"evidence_ids":["E1"]}\n```', "stop", "json"),
])
def test_planner_diagnoses_provider_output_without_storing_raw_content(monkeypatch, content, finish, status):
    from types import SimpleNamespace as NS

    from data_analysis_agent import model_budget

    provider_arguments = {}

    def create(**kwargs):
        provider_arguments.update(kwargs)
        return NS(choices=[NS(finish_reason=finish, message=NS(content=content, reasoning_content=""))],
                  usage=NS(prompt_tokens=40, completion_tokens=10, total_tokens=50))

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-test")
    monkeypatch.delenv("MODEL_SPEND_CAP_USD", raising=False)
    monkeypatch.setattr(model_budget, "OpenAI",
                        lambda **_: NS(chat=NS(completions=NS(create=create))))
    model = model_budget.BudgetedModel()
    if status == "json":
        assert model.plan("{}") == {"evidence_ids": ["E1"]}
    else:
        with pytest.raises(model_budget.PlannerResponseError, match=status):
            model.plan("{}")
    record, = model.planner_calls
    assert record["response_status"] == status
    assert (record["finish_reason"], record["content_chars"], record["total_tokens"]) == (finish, len(content), 50)
    assert not {"content", "prompt", "headers"} & record.keys()
    assert provider_arguments["extra_body"] == {"thinking": {"type": "disabled"}}
    assert provider_arguments["max_tokens"] == 3072 and provider_arguments["reasoning_effort"] == "none"


def test_business_frozen_plan_is_isolated_and_uses_shared_event_search(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "after_onset")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    payloads = []

    def plan(prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        payloads.append(payload)
        assert not {"verdict", "changes", "findings", "case"} & set(payload["goal"])
        assert case.id not in json.dumps(payload)
        if len(payloads) == 1:
            assert [item["id"] for item in payload["evidence"]] == ["E1"]
            assert {"stock", "events", "timeline"} <= payload["tools"].keys()
            return {"actions": ["events", "timeline"], "reason": "Check event and onset dates."}
        assert [item["id"] for item in payload["evidence"]] == ["E1", "E2", "E3"]
        return {"verdict": "insufficient", "claims": [
            {"evidence": id, "fact": fact} for id, fact in zip(("E1", "E3", "E2"), case.findings)
        ]}

    state = investigate((database, database), case.goal, budget=5, policy="llm_frozen", planner=plan, events=events)
    assert len(payloads) == 2
    assert [step["visible_evidence"] for step in state.trace[1:]] == [["E1"], ["E1"]]
    assert score(state, case)["correct"]


def test_business_refutation_changes_executed_path(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "recall_scope")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    actions = iter(("events", "stock_core", "scope", "complaints"))

    def plan(prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if "verdicts" in payload:
            return {"verdict": "plausible", "claims": [
                {"evidence": item["id"], "fact": fact} for item in payload["evidence"] for fact in item["facts"]
            ]}
        action = next(actions)
        decision = {"action": action, "question": "Can this evidence distinguish the explanations?"}
        if action == "stock_core":
            decision.update(expect="stock_core:availability=down", next="pricing")
        if action == "scope":
            assert payload["previous"]["refutation"] == {"evidence": "E3", "actual": "stock_core:availability=stable"}
        return decision

    state = investigate((database, database), case.goal, budget=5, policy="llm_adaptive", planner=plan, events=events)
    result = score(state, case)
    assert result["correct"] and result["changes_after_refutation"] == 1
    assert result["refutations"] == 1 and state.remaining_budget == 0
    fixed = investigate((database, database), case.goal, budget=5, policy="workflow", events=events)
    assert all(item.action != ("stock_core",) for item in fixed.evidence)
    assert score(fixed, case)["correct"]


def test_business_scoring_detects_false_cause_despite_real_event(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "stock_vs_bad_press")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    state = investigate((database, database), case.goal, budget=5, events=events)
    evidence = next(item for item in state.evidence if item.action == ("stock",))
    assert evidence.rows[0][1:4] == (1.0, 56, 56) and evidence.rows[1][1:4] == (0.5, 56, 56)
    assert evidence.rows[0][-1] == 1400 and evidence.rows[1][-1] == 0
    assert score(state, case)["correct"]
    state.verdict = "causal"
    state.claims.append({"evidence": "E999", "fact": "events:press=matched"})
    result = score(state, case)
    assert result["event_retrieval"] and result["event_matching"]
    assert result["wrong_attribution"] and not result["correct"]
    assert result["unsupported_claims"] == [("E999", "events:press=matched")]


def test_business_frozen_answers_are_reachable_and_splits_disjoint(tmp_path):
    from pathlib import Path

    from data_analysis_agent.business_cases import DEV, TEST, build_retail
    from data_analysis_agent.business_investigation import event_findings
    from data_analysis_agent.business_tools import observe

    assert {item.goal for item in TEST}.isdisjoint(item.goal for item in DEV)
    assert [sum(item.kind == group for item in TEST) for group in "ABCD"] == [8, 8, 8, 6]
    olist = Path("data/processed/olist.duckdb")
    if not olist.is_file():
        pytest.skip("Build the audited Olist database to verify all observational gold")
    for case in (*TEST, *DEV):
        database, events = olist, ()
        if case.goal.domain == "retail":
            database = tmp_path / f"{case.id}.duckdb"
            events = build_retail(case, database)
        prefixes = {fact.split(":")[0] for fact in case.findings}
        if case.goal.dimension in prefixes:
            prefixes.remove(case.goal.dimension)
            prefixes.add("breakdown")
        evidence = [observe(database, case.goal, action, f"E{i}", events)
                    for i, action in enumerate(sorted(prefixes | {"headline"}), 1)]
        assert all(not item.error for item in evidence), case.id
        findings = {fact for facts in event_findings(case.goal, evidence).values() for fact in facts}
        assert set(case.findings) <= findings, case.id
        assert len(evidence) <= 5, case.id
