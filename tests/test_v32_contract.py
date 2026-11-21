from dataclasses import FrozenInstanceError

import duckdb
import pytest

from data_analysis_agent.query_contract import answer_contract, aggregate_sql, parse_question
from data_analysis_agent.sql_executor import run_readonly_sql


QUESTION = "gmv from 2018-01-01 to 2018-02-01 where state=sp and status=DELIVERED"


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "olist.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE orders (
                order_id VARCHAR, customer_id VARCHAR, order_purchase_timestamp DATE,
                order_status VARCHAR, order_delivered_customer_date DATE, order_estimated_delivery_date DATE
            );
            INSERT INTO orders VALUES
                ('a', 'c', '2018-01-01', 'delivered', NULL, NULL),
                ('b', 'c', '2018-02-01', 'delivered', NULL, NULL),
                ('d', 'c', '2018-01-01', 'canceled', NULL, NULL);
            CREATE TABLE customers AS SELECT 'c' AS customer_id, 'SP' AS customer_state;
            CREATE TABLE order_items (order_id VARCHAR, price INTEGER, freight INTEGER);
            INSERT INTO order_items VALUES ('a', 10, 2), ('a', 20, 3), ('b', 50, 9), ('d', 100, 9);
            CREATE TABLE order_payments (order_id VARCHAR, payment INTEGER);
            INSERT INTO order_payments VALUES ('a', 12), ('a', 23);
            CREATE VIEW order_item_summary AS
                SELECT order_id, COUNT(*) AS item_count,
                       SUM(price) AS item_transaction_value, SUM(freight) AS freight_value
                FROM order_items GROUP BY 1;
        """)
    return path


def test_immutable_contract_restores_grain_filter_window_and_money(database):
    contract = parse_question(QUESTION, "olist")
    with pytest.raises(FrozenInstanceError):
        contract.grain = "day"
    fanout = "SELECT SUM(price + freight) FROM order_items JOIN order_payments USING(order_id)"
    assert run_readonly_sql(database, fanout).rows == ((70,),)
    answer = answer_contract(database, QUESTION, "olist", fanout,
                             repair=lambda locked, error: aggregate_sql(locked, cte=True))
    assert answer.contract == contract
    assert answer.result.rows == ((30,),)
    assert [step.status for step in answer.trace] == ["error", "success"]


@pytest.mark.parametrize("replacement", ["gmv_with_freight", "orders", "2018-03-01", "canceled", "by month"])
def test_drift_in_second_attempt_is_refused(database, replacement):
    question = {
        "2018-03-01": QUESTION.replace("2018-02-01", replacement),
        "canceled": QUESTION.replace("DELIVERED", replacement),
        "by month": QUESTION + " " + replacement,
    }.get(replacement, QUESTION.replace("gmv", replacement))
    drift = aggregate_sql(parse_question(question, "olist"))
    answer = answer_contract(database, QUESTION, "olist", "SELECT 1",
                             repair=lambda locked, issue: drift)
    assert answer.status == "refused"
    assert answer.result is None
    assert len(answer.trace) == 2


@pytest.mark.parametrize("question", ["net profit", "gmv including freight", "average_review_score"])
def test_question_requires_supported_semantics(database, question):
    answer = answer_contract(database, question + " from 2018-01-01 to 2018-02-01", "olist")
    assert answer.status == "refused" and answer.contract is None


def test_executor_blocks_writes_and_caps_runtime(database):
    assert run_readonly_sql(database, "UPDATE orders SET order_status = 'canceled'").error.code == "unsafe_sql"
    timeout = run_readonly_sql(database,
                              "SELECT SUM(sin(i)) FROM range(1000000000) AS t(i)",
                              timeout_seconds=0.05)
    assert timeout.error.code == "timeout"
    assert answer_contract(database, QUESTION, "olist").result.rows == ((30,),)


def test_online_budget_counts_attempts_stops_on_limit_and_uses_default(monkeypatch):
    from types import SimpleNamespace

    from data_analysis_agent import model_budget

    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=13),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    def client(**kwargs):
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completions)))

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "model-for-testing")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("MAX_LLM_REQUESTS", "1")
    monkeypatch.setenv("MAX_LLM_TOTAL_TOKENS", "2000")
    monkeypatch.delenv("MODEL_SPEND_CAP_USD", raising=False)
    monkeypatch.setattr(model_budget, "OpenAI", client)
    model = model_budget.BudgetedModel()
    assert model("test") == "{}"
    with pytest.raises(RuntimeError, match="budget"):
        model("again")
    assert model.requests == 1 and model.tokens == 13 and len(calls) == 1
