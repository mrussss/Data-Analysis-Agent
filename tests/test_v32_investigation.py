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


def choice(dimensions, segment, *, evidence="E1", cohort=None, direction="increase",
           next_action=None, revises=None):
    return {
        "decision": "investigate", "action": {"dimensions": dimensions, "cohort": cohort or {}},
        "hypothesis": {"segment": segment, "effect": "within", "direction": direction},
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
            return choice(["state"], ["SP"], direction="stable", next_action=planned)
        if len(inputs) == 2:
            assert data["tested_hypotheses"][0]["state"] == "refuted"
            return choice(["weekday"], ["3"], cohort={"state": "SP"}, evidence="E2", revises="H1")
        return {"decision": "stop", "reason": "sufficient_evidence", "evidence_ids": ["E3"],
                "question": "Can the localized shift account for the observed change?"}

    adaptive = investigate((source, observed), goal, policy="llm_adaptive", budget=6, planner=planner)
    assert [e.action for e in adaptive.evidence] == [(), ("state",), ("weekday",)]
    assert adaptive.evidence[-1].cohort == (("state", "SP"),)
    assert "state = 'SP'" in adaptive.evidence[-1].sql[0]
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
            return choice(["state"], [None])
        assert len(data[-1]["past_decisions"]) == 1
        return choice([], [], cohort={"state": "RJ"})

    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    state = investigate((source, source), goal, policy="llm_frozen", budget=3, planner=planner)
    assert [step["evidence"] for step in state.trace] == ["E1", "E2", "E3"]
    assert state.evidence[-1].cohort == (("state", "RJ"),)
    assert effective_replans(state) == (0, 0)


@pytest.mark.parametrize("invalid", [
    choice(["state"], ["SP"], evidence="E999"),
    choice([], [], cohort={"state": "SP' OR TRUE --"}),
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
