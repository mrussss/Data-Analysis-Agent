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


def test_agent_changes_path_after_refutation_and_never_receives_manifest(source, tmp_path):
    case = Case("hidden-mechanism", WINDOW, WINDOW, (("SP", 3, 1, 65),))
    observed, manifest = snapshot(source, case, tmp_path / "observed.duckdb")
    assert counts(source, WINDOW) == (1400, 280)
    assert manifest["current"] == (1400, 345)
    goal = parse_comparison(case.question, "olist")
    runtime_inputs = []

    def planner(payload):
        candidate = json.loads(payload)
        runtime_inputs.append(candidate)
        return {"action": candidate["rule_choice"]}

    adaptive = investigate((source, observed), goal, budget=4, planner=planner)
    frozen = investigate((source, observed), goal, policy="frozen", budget=4, planner=planner)
    assert [e.action for e in adaptive.evidence] == [(), ("state",), ("state", "weekday"), ("state", "status")]
    assert [e.action for e in adaptive.evidence] != [e.action for e in frozen.evidence]
    assert effective_replans(adaptive) == (1, 1)
    assert score_investigation(adaptive, manifest, None)["top1"]
    assert all("hidden-mechanism" not in json.dumps(data) and "matched" not in json.dumps(data)
               for data in runtime_inputs)
    assert runtime_inputs[-1]["recent_evidence"] is None
    assert "gold" not in report(adaptive)
    assert adaptive.stop_reason == "budget_exhausted"


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
