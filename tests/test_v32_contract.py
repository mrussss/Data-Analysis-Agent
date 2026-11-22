"""V32 contract regression tests."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import duckdb
import pytest

import data_analysis_agent.question_service as service_module
import data_analysis_agent.sql_executor as sql_executor_module
from data_analysis_agent import run_readonly_sql as sql_executor_run_readonly_sql
from data_analysis_agent.analysis_planner import (
    build_python_analysis_plan_prompt,
    generate_python_analysis_plan,
)
from data_analysis_agent.metric_catalog import (
    BUSINESS_SEMANTICS_V1,
    format_business_semantics_context,
    format_semantic_layer_context,
    METRIC_DEFINITIONS_V2,
    MetricDefinition,
    QUERY_CONSTRAINTS_V2,
)
from data_analysis_agent.query_contract import aggregate_sql, answer_contract, parse_question
from data_analysis_agent.question_service import answer_question, MAX_SQL_REPAIR_ATTEMPTS
from data_analysis_agent.schema import ColumnSchema, DatabaseSchema, SchemaObject
from data_analysis_agent.sql_executor import run_readonly_sql, SQLExecutionError, SQLResult
from data_analysis_agent.sql_generator import (
    build_text_to_sql_prompt,
    format_schema_context,
    generate_sql,
    SQLGenerationError,
    SQLGenerationResult,
)
from data_analysis_agent.sql_repair import (
    build_sql_repair_prompt,
    repair_sql,
    SQLRepairError,
    SQLRepairResult,
)
from data_analysis_agent.tool_router import route_question, ToolRouteDecision, ToolRoutingError
from scripts.build_duckdb import build_database


# V32 contract

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


# Sql executor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sql_executor") / "olist.duckdb"
    build_database(DATA_DIR, path)
    return path


def test_executes_with_select(database_path: Path) -> None:
    sql = "WITH statuses AS (SELECT order_status FROM orders) SELECT COUNT(*) FROM statuses"
    result = sql_executor_run_readonly_sql(database_path, sql)

    assert result.status == "success"
    assert result.rows == ((99_441,),)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET order_status = 'canceled'",
        "INSERT INTO orders SELECT * FROM orders",
        "DROP TABLE orders",
        "CREATE TABLE copy_of_orders AS SELECT * FROM orders",
        "ALTER TABLE orders ADD COLUMN unsafe INTEGER",
        "TRUNCATE TABLE orders",
        "COPY orders TO '/tmp/orders.csv'",
        "ATTACH 'other.duckdb' AS other",
        "DETACH other",
        "INSTALL httpfs",
        "LOAD httpfs",
        "CALL checkpoint()",
        "EXPORT DATABASE '/tmp/olist_export'",
        "IMPORT DATABASE '/tmp/olist_export'",
        "VACUUM",
        "SET threads = 1",
        "PRAGMA version",
    ],
)
def test_rejects_unsafe_statement(database_path: Path, sql: str) -> None:
    result = sql_executor_run_readonly_sql(database_path, sql)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "unsafe_sql"
    assert result.rows == ()


def test_rejects_multiple_statements(database_path: Path) -> None:
    result = sql_executor_run_readonly_sql(database_path, "SELECT 1; SELECT 2")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "multiple_statements"


def test_comments_and_string_literals_do_not_trigger_false_rejection(
    database_path: Path,
) -> None:
    sql = "-- DELETE is text only\nSELECT 'DROP TABLE orders' AS statement_text"
    result = sql_executor_run_readonly_sql(database_path, sql)

    assert result.status == "success"
    assert result.rows == (("DROP TABLE orders",),)


def test_limits_materialized_rows_and_marks_truncation(database_path: Path) -> None:
    result = sql_executor_run_readonly_sql(
        database_path,
        "SELECT order_id, order_item_id FROM order_items ORDER BY order_id, order_item_id",
        max_rows=5,
    )

    assert result.status == "success"
    assert result.returned_row_count == 5
    assert len(result.rows) == 5
    assert result.truncated is True


def test_exact_row_limit_is_not_marked_truncated(database_path: Path) -> None:
    result = sql_executor_run_readonly_sql(database_path, "SELECT * FROM orders LIMIT 5", max_rows=5)

    assert result.returned_row_count == 5
    assert result.truncated is False


def test_fetches_only_max_rows_plus_one_without_fetchall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "placeholder.duckdb"
    database_path.touch()

    class FakeCursor:
        description = (("value",),)

        def fetchmany(self, size: int):
            assert size == 4
            return [(1,), (2,), (3,), (4,)]

        def fetchall(self):
            raise AssertionError("fetchall must not be used by the executor")

    class FakeConnection:
        def interrupt(self):
            raise AssertionError("A completed query must cancel the execution timer")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, sql: str):
            assert sql == "SELECT value FROM large_table"
            return FakeCursor()

    def fake_connect(path: str, *, read_only: bool, config: dict[str, str]):
        assert path == str(database_path)
        assert read_only is True
        assert config["enable_external_access"] == "false"
        return FakeConnection()

    monkeypatch.setattr(sql_executor_module.duckdb, "connect", fake_connect)

    result = sql_executor_run_readonly_sql(
        database_path,
        "SELECT value FROM large_table",
        max_rows=3,
    )

    assert result.rows == ((1,), (2,), (3,))
    assert result.returned_row_count == 3
    assert result.truncated is True


def test_reports_invalid_sql(database_path: Path) -> None:
    result = sql_executor_run_readonly_sql(database_path, "SELECT COUNT(* FROM orders")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_sql"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM table_that_does_not_exist",
        "SELECT column_that_does_not_exist FROM orders",
    ],
)
def test_reports_unknown_table_or_column(database_path: Path, sql: str) -> None:
    result = sql_executor_run_readonly_sql(database_path, sql)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "unknown_table_or_column"


def test_external_file_access_is_disabled(database_path: Path) -> None:
    result = sql_executor_run_readonly_sql(database_path, "SELECT * FROM read_csv('/etc/passwd')")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "execution_error"
    assert "file system operations are disabled" in result.error.message.lower()


# Sql generator

SCHEMA = DatabaseSchema(
    objects=(
        SchemaObject(
            name="orders",
            object_type="table",
            grain="one row per order",
            columns=(
                ColumnSchema("order_id", "VARCHAR", False, True),
                ColumnSchema("order_status", "VARCHAR", False, False),
                ColumnSchema(
                    "order_purchase_timestamp", "TIMESTAMP", True, False
                ),
            ),
        ),
        SchemaObject(
            name="order_payment_summary",
            object_type="view",
            grain="one row per order",
            columns=(
                ColumnSchema("order_id", "VARCHAR", True, False),
                ColumnSchema("payment_value", "DECIMAL(38,2)", True, False),
            ),
        ),
    )
)


def test_normal_question_produces_structured_success_result() -> None:
    sql = "SELECT COUNT(*) AS order_count FROM orders"

    result = generate_sql(
        "How many orders are in the dataset?",
        SCHEMA,
        lambda prompt: {"status": "success", "sql": sql},
    )

    assert result == SQLGenerationResult(
        question="How many orders are in the dataset?",
        sql=sql,
        status="success",
        error=None,
    )


def test_schema_context_is_deterministic_and_contains_types_and_grain() -> None:
    reversed_schema = DatabaseSchema(objects=tuple(reversed(SCHEMA.objects)))

    context = format_schema_context(SCHEMA)

    assert context == format_schema_context(reversed_schema)
    assert "TABLE orders" in context
    assert "VIEW order_payment_summary" in context
    assert "Grain: one row per order" in context
    assert "- order_purchase_timestamp: TIMESTAMP" in context
    assert context.index("VIEW order_payment_summary") < context.index("TABLE orders")


def test_prompt_preserves_chinese_question_verbatim() -> None:
    question = "商品成交金额每个月的环比变化怎么样？"

    assert question in build_text_to_sql_prompt(question, SCHEMA)


def test_prompt_does_not_include_gold_answers_or_baseline_sql() -> None:
    prompt = build_text_to_sql_prompt(
        "How many orders are in the dataset?",
        SCHEMA,
    )

    assert "99,441" not in prompt
    assert "99441" not in prompt
    assert "BASELINE_QUERIES" not in prompt
    assert "baseline_key" not in prompt
    assert "SELECT COUNT(*) AS total_orders FROM orders" not in prompt


def test_malformed_model_output_becomes_structured_error() -> None:
    result = generate_sql("How many orders are there?", SCHEMA, lambda prompt: "not json")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_model_output"
    assert result.sql is None


def test_model_exception_becomes_structured_error() -> None:
    def failing_model(prompt: str):
        raise RuntimeError("provider unavailable")

    result = generate_sql("How many orders are there?", SCHEMA, failing_model)

    assert result.status == "error"
    assert result.error == SQLGenerationError(
        code="model_error",
        message="provider unavailable",
    )


def test_model_can_decline_to_generate_unreliable_sql() -> None:
    reason = "The schema has no product cost field."

    result = generate_sql(
        "What is Olist's gross profit margin?",
        SCHEMA,
        lambda prompt: {"status": "error", "error": reason},
    )

    assert result.status == "error"
    assert result.sql is None
    assert result.error == SQLGenerationError(
        code="cannot_generate",
        message=reason,
    )


# Sql repair

sql_repair_QUESTION = "How many orders are in the dataset?"


ORIGINAL_SQL = "SELECT COUNT(*) FROM orderz"


REPAIRED_SQL = "SELECT COUNT(*) FROM orders"


sql_repair_SCHEMA = DatabaseSchema(
    objects=(
        SchemaObject(
            name="orders",
            object_type="table",
            grain="one row per order",
            columns=(ColumnSchema("order_id", "VARCHAR", False, True),),
        ),
    )
)


EXECUTION_ERROR = SQLExecutionError(
    code="unknown_table_or_column",
    message='Table with name "orderz" does not exist.',
)


def test_repair_prompt_contains_question_sql_error_schema_and_safety_rules() -> None:
    prompt = build_sql_repair_prompt(
        sql_repair_QUESTION,
        sql_repair_SCHEMA,
        ORIGINAL_SQL,
        EXECUTION_ERROR,
    )

    assert f"Original question:\n{sql_repair_QUESTION}" in prompt
    assert f"Original SQL:\n{ORIGINAL_SQL}" in prompt
    assert "Execution error:\nunknown_table_or_column" in prompt
    assert 'Table with name "orderz" does not exist.' in prompt
    assert "TABLE orders" in prompt
    assert "- order_id: VARCHAR" in prompt
    assert "SELECT or WITH ... SELECT" in prompt
    assert "Do not read external files" in prompt
    assert "do not answer or explain" in prompt
    assert "JSON object" in prompt


def test_repair_success_returns_sql_unchanged() -> None:
    result = repair_sql(
        sql_repair_QUESTION,
        sql_repair_SCHEMA,
        ORIGINAL_SQL,
        EXECUTION_ERROR,
        lambda prompt: {"status": "success", "sql": REPAIRED_SQL},
    )

    assert result == SQLRepairResult(
        question=sql_repair_QUESTION,
        original_sql=ORIGINAL_SQL,
        repaired_sql=REPAIRED_SQL,
        status="success",
        error=None,
    )


def test_model_decline_returns_cannot_repair() -> None:
    result = repair_sql(
        sql_repair_QUESTION,
        sql_repair_SCHEMA,
        ORIGINAL_SQL,
        EXECUTION_ERROR,
        lambda prompt: {"status": "error", "error": "Insufficient schema."},
    )

    assert result.status == "error"
    assert result.repaired_sql is None
    assert result.error == SQLRepairError(
        code="cannot_repair",
        message="Insufficient schema.",
    )


# Question service

question_service_SCHEMA = DatabaseSchema(objects=())


question_service_QUESTION = "How many orders are in the dataset?"


SQL = "SELECT COUNT(*) AS order_count FROM orders"


question_service_REPAIRED_SQL = "SELECT COUNT(*) AS order_count FROM orders;"


def _failed_execution(sql: str, code="unknown_table_or_column") -> SQLResult:
    error = SQLExecutionError(
        code=code,
        message='Table with name "missing" does not exist.',
    )
    return SQLResult(
        executed_sql=sql,
        columns=(),
        rows=(),
        returned_row_count=0,
        truncated=False,
        status="error",
        error=error,
    )


def _successful_execution(sql: str) -> SQLResult:
    return SQLResult(
        executed_sql=sql,
        columns=("order_count",),
        rows=((99_441,),),
        returned_row_count=1,
        truncated=False,
        status="success",
        error=None,
    )


def test_execution_failure_is_repaired_and_reexecuted_successfully(monkeypatch) -> None:
    model_prompts: list[str] = []
    model_outputs = iter(
        [
            {"status": "success", "sql": SQL},
            {"status": "success", "sql": question_service_REPAIRED_SQL},
        ]
    )
    execution_calls: list[tuple[str, int]] = []

    def model(prompt: str):
        model_prompts.append(prompt)
        return next(model_outputs)

    def fake_executor(database_path, sql: str, max_rows: int) -> SQLResult:
        execution_calls.append((sql, max_rows))
        if len(execution_calls) == 1:
            return _failed_execution(sql)
        return _successful_execution(sql)

    monkeypatch.setattr(service_module, "run_readonly_sql", fake_executor)

    result = answer_question(
        "unused.duckdb",
        question_service_QUESTION,
        model,
        max_rows=25,
        schema=question_service_SCHEMA,
    )

    assert result.status == "success"
    assert result.generated_sql == SQL
    assert result.repaired_sql == question_service_REPAIRED_SQL
    assert result.repair_attempted is True
    assert result.execution_result == _successful_execution(question_service_REPAIRED_SQL)
    assert result.generation_error is None
    assert result.repair_error is None
    assert result.execution_error is None
    assert execution_calls == [(SQL, 25), (question_service_REPAIRED_SQL, 25)]
    assert len(model_prompts) == 2
    assert "Original SQL:" in model_prompts[1]


def test_second_execution_failure_stops_without_third_attempt(monkeypatch) -> None:
    model_prompts: list[str] = []
    model_outputs = iter(
        [
            {"status": "success", "sql": SQL},
            {"status": "success", "sql": question_service_REPAIRED_SQL},
        ]
    )
    execution_calls: list[str] = []

    def model(prompt: str):
        model_prompts.append(prompt)
        return next(model_outputs)

    def fake_executor(database_path, sql: str, max_rows: int) -> SQLResult:
        execution_calls.append(sql)
        return _failed_execution(sql)

    monkeypatch.setattr(service_module, "run_readonly_sql", fake_executor)

    result = answer_question(
        "unused.duckdb",
        question_service_QUESTION,
        model,
        schema=question_service_SCHEMA,
    )

    assert result.status == "execution_error"
    assert result.generated_sql == SQL
    assert result.repaired_sql == question_service_REPAIRED_SQL
    assert result.repair_attempted is True
    assert result.execution_result == _failed_execution(question_service_REPAIRED_SQL)
    assert result.execution_error == _failed_execution(question_service_REPAIRED_SQL).error
    assert result.repair_error is None
    assert execution_calls == [SQL, question_service_REPAIRED_SQL]
    assert len(model_prompts) == 2
    assert MAX_SQL_REPAIR_ATTEMPTS == 1


def test_generation_failure_does_not_call_executor(monkeypatch) -> None:
    def unexpected_executor(database_path, sql, max_rows):
        raise AssertionError("executor must not run after generation failure")

    monkeypatch.setattr(service_module, "run_readonly_sql", unexpected_executor)
    monkeypatch.setattr(
        service_module,
        "repair_sql",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repair must not run after generation failure")
        ),
    )

    result = answer_question(
        "unused.duckdb",
        question_service_QUESTION,
        lambda prompt: {
            "status": "error",
            "error": "No reliable SQL can be generated.",
        },
        schema=question_service_SCHEMA,
    )

    assert result.status == "generation_error"
    assert result.generation_error == SQLGenerationError(
        code="cannot_generate",
        message="No reliable SQL can be generated.",
    )


def test_safety_rejection_is_not_repaired(monkeypatch) -> None:
    execution_calls: list[str] = []

    def fake_executor(database_path, sql: str, max_rows: int) -> SQLResult:
        execution_calls.append(sql)
        return _failed_execution(sql, code="unsafe_sql")

    monkeypatch.setattr(service_module, "run_readonly_sql", fake_executor)
    monkeypatch.setattr(
        service_module,
        "repair_sql",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("safety rejection must not enter repair")
        ),
    )

    result = answer_question(
        "unused.duckdb",
        question_service_QUESTION,
        lambda prompt: {"status": "success", "sql": SQL},
        schema=question_service_SCHEMA,
    )

    assert result.status == "execution_error"
    assert result.repair_attempted is False
    assert result.execution_error is not None
    assert result.execution_error.code == "unsafe_sql"
    assert execution_calls == [SQL]


# Metric catalog

def _metric(identifier: str) -> MetricDefinition:
    return next(
        metric for metric in METRIC_DEFINITIONS_V2 if metric.identifier == identifier
    )


def test_v2_metrics_have_stable_unique_identifiers() -> None:
    identifiers = tuple(metric.identifier for metric in METRIC_DEFINITIONS_V2)

    assert identifiers == (
        "average_review_score",
        "average_items_per_order",
        "average_delivery_duration_days",
    )
    assert len(identifiers) == len(set(identifiers))


def test_average_review_score_rejects_implicit_order_level_aggregation() -> None:
    instructions = " ".join(_metric("average_review_score").instructions)

    assert "Do not aggregate reviews to order grain" in instructions
    assert "average of order-level average review scores" in instructions


def test_average_items_per_order_denominator_is_all_orders() -> None:
    metric = _metric("average_items_per_order")

    assert metric.population.startswith("All orders")
    assert "divided by total number of orders" in metric.aggregation
    assert "Use all rows in orders as the denominator" in " ".join(
        metric.instructions
    )


def test_delivery_duration_uses_exact_seconds_divided_by_86400() -> None:
    metric = _metric("average_delivery_duration_days")

    assert "DATE_DIFF('second'" in metric.aggregation
    assert "/ 86400.0" in metric.aggregation
    assert "AVG the per-order duration_days values" in metric.aggregation
    assert "continuous elapsed seconds / 86400.0" in metric.time_semantics


def test_delivery_duration_does_not_implicitly_filter_delivered_status() -> None:
    instructions = " ".join(
        _metric("average_delivery_duration_days").instructions
    )

    assert "Do not automatically add order_status = 'delivered'" in instructions
    assert "only when the user explicitly requests it" in instructions


def test_explicit_top_bottom_n_semantics_are_preserved() -> None:
    constraint = QUERY_CONSTRAINTS_V2[0]
    instructions = " ".join(constraint.instructions)

    assert constraint.identifier == "explicit_top_bottom_n"
    assert "only the requested N rows" in constraint.definition
    assert "descending order for Top-N and ascending order for Bottom-N" in instructions
    assert "LIMIT or a mathematically equivalent bounded ranking" in instructions


def test_backward_compatible_formatter_accepts_v1_collection() -> None:
    context = format_business_semantics_context(BUSINESS_SEMANTICS_V1)

    assert "average_review_score" in context
    assert "explicit_top_bottom_n" in context


def test_formatter_excludes_secrets_and_runtime_information() -> None:
    context = format_semantic_layer_context().lower()

    assert "api key" not in context
    assert "deepseek" not in context
    assert "prompt secret" not in context
    assert "runtime" not in context


def test_metric_chinese_labels_are_canonical_and_identifiers_are_unchanged() -> None:
    assert tuple(
        (metric.identifier, metric.zh_label) for metric in METRIC_DEFINITIONS_V2
    ) == (
        ("average_review_score", "平均评论分"),
        ("average_items_per_order", "每单平均商品数"),
        ("average_delivery_duration_days", "平均配送时长"),
    )


# Analysis planner

analysis_planner_SCHEMA = DatabaseSchema(
    objects=(
        SchemaObject(
            name="order_items",
            object_type="table",
            grain="one row per order item",
            columns=(
                ColumnSchema("order_id", "VARCHAR", False, False),
                ColumnSchema("price", "DECIMAL(12,2)", False, False),
                ColumnSchema("freight_value", "DECIMAL(12,2)", False, False),
            ),
        ),
        SchemaObject(
            name="order_payments",
            object_type="table",
            grain="one row per payment record",
            columns=(
                ColumnSchema("order_id", "VARCHAR", False, False),
                ColumnSchema("payment_value", "DECIMAL(12,2)", False, False),
            ),
        ),
        SchemaObject(
            name="orders",
            object_type="table",
            grain="one row per order",
            columns=(
                ColumnSchema("order_id", "VARCHAR", False, True),
                ColumnSchema(
                    "order_purchase_timestamp",
                    "TIMESTAMP",
                    False,
                    False,
                ),
            ),
        ),
    )
)


def _success_output(sql: str, columns: list[object]) -> dict[str, object]:
    return {
        "status": "success",
        "sql": sql,
        "python_columns": columns,
    }


def test_correlation_plan_rejects_sql_corr_function() -> None:
    result = generate_python_analysis_plan(
        "What is the correlation between item price and freight value?",
        analysis_planner_SCHEMA,
        "correlation",
        lambda prompt: _success_output(
            "SELECT CORR(price, freight_value) AS correlation FROM order_items",
            ["price", "freight_value"],
        ),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_analysis_plan"
    assert "corr" in result.error.message


def test_describe_plan_rejects_sql_summary_functions() -> None:
    result = generate_python_analysis_plan(
        "Describe payment values.",
        analysis_planner_SCHEMA,
        "describe",
        lambda prompt: _success_output(
            "SELECT AVG(payment_value) AS payment_value FROM order_payments",
            ["payment_value"],
        ),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_analysis_plan"
    assert "avg" in result.error.message


def test_calculate_growth_plan_returns_ordered_period_value_series() -> None:
    sql = """
        SELECT
            DATE_TRUNC('month', o.order_purchase_timestamp) AS month,
            SUM(oi.price) AS total_item_transaction_value
        FROM orders AS o
        JOIN order_items AS oi ON o.order_id = oi.order_id
        GROUP BY month
        ORDER BY month
    """

    result = generate_python_analysis_plan(
        "How did total item transaction value change month over month?",
        analysis_planner_SCHEMA,
        "calculate_growth",
        lambda prompt: _success_output(
            sql,
            ["month", "total_item_transaction_value"],
        ),
    )

    assert result.status == "success"
    assert result.sql == sql
    assert result.python_columns == ("month", "total_item_transaction_value")


def test_calculate_growth_plan_rejects_lag() -> None:
    sql = """
        SELECT
            month,
            metric_value - LAG(metric_value) OVER (ORDER BY month) AS metric_value
        FROM monthly_values
        ORDER BY month
    """

    result = generate_python_analysis_plan(
        "How did the metric change month over month?",
        analysis_planner_SCHEMA,
        "calculate_growth",
        lambda prompt: _success_output(sql, ["month", "metric_value"]),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_analysis_plan"
    assert "lag" in result.error.message


def test_calculate_growth_plan_rejects_current_previous_arithmetic() -> None:
    sql = """
        SELECT
            current.month,
            current.metric_value - previous.metric_value AS metric_value
        FROM monthly_values AS current
        JOIN monthly_values AS previous
          ON current.month = previous.month + INTERVAL '1 month'
        ORDER BY current.month
    """

    result = generate_python_analysis_plan(
        "How did the metric change month over month?",
        analysis_planner_SCHEMA,
        "calculate_growth",
        lambda prompt: _success_output(sql, ["month", "metric_value"]),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_analysis_plan"
    assert "current-versus-previous arithmetic" in result.error.message


def test_calculate_growth_plan_requires_ordered_sql() -> None:
    result = generate_python_analysis_plan(
        "How did the metric change month over month?",
        analysis_planner_SCHEMA,
        "calculate_growth",
        lambda prompt: _success_output(
            "SELECT month, metric_value FROM monthly_values",
            ["month", "metric_value"],
        ),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_analysis_plan"
    assert "ORDER BY" in result.error.message


def test_malformed_json_returns_invalid_model_output() -> None:
    result = generate_python_analysis_plan(
        "Describe payment values.",
        analysis_planner_SCHEMA,
        "describe",
        lambda prompt: "not json",
    )

    assert result.error is not None
    assert result.error.code == "invalid_model_output"


def analysis_planner_test_prompt_preserves_chinese_question_verbatim() -> None:
    question = "商品成交金额每个月的环比变化怎么样？"

    prompt = build_python_analysis_plan_prompt(
        question,
        analysis_planner_SCHEMA,
        "calculate_growth",
    )

    assert question in prompt


# Tool router

@pytest.mark.parametrize(
    "question",
    [
        "How many orders are in the dataset?",
        "What is the average payment value per order?",
        "Give me descriptive statistics for payment values.",
        (
            "Give me summary statistics including mean, standard deviation, "
            "median, min, and max for item price."
        ),
        "What is the Pearson correlation between item price and freight value?",
        "What was total item transaction value by month?",
        "Show average payment value by month.",
        "List all customer states ranked by unique customer count.",
        "What percentage of orders contain multiple items?",
    ],
)
def test_sql_native_questions_route_to_sql_only(question: str) -> None:
    result = route_question(
        question,
        lambda prompt: json.dumps(
            {
                "status": "success",
                "route": "sql_only",
                "python_operation": None,
                "reason": "SQL can answer this aggregation directly.",
            }
        ),
    )

    assert result == ToolRouteDecision(
        question=question,
        route="sql_only",
        python_operation=None,
        reason="SQL can answer this aggregation directly.",
        status="success",
        error=None,
    )


@pytest.mark.parametrize(
    "question",
    [
        "How did total item transaction value change month over month?",
        (
            "Which month had the largest month-over-month decline in total "
            "item transaction value?"
        ),
    ],
)
def test_period_over_period_questions_route_to_calculate_growth(
    question: str,
) -> None:
    result = route_question(
        question,
        lambda prompt: {
            "status": "success",
            "route": "sql_then_python",
            "python_operation": "calculate_growth",
            "reason": "The final result requires consecutive-period analysis.",
        },
    )

    assert result.status == "success"
    assert result.route == "sql_then_python"
    assert result.python_operation == "calculate_growth"


def test_unsupported_python_operation_returns_structured_error() -> None:
    result = route_question(
        "Fit a regression model.",
        lambda prompt: {
            "status": "success",
            "route": "sql_then_python",
            "python_operation": "regression",
            "reason": "Run regression after SQL.",
        },
    )

    assert result.status == "error"
    assert result.error == ToolRoutingError(
        code="unsupported_route",
        message="Unsupported Python operation: 'regression'",
    )


def test_model_can_reject_request_with_no_supported_route() -> None:
    result = route_question(
        "Forecast next year's sales.",
        lambda prompt: {
            "status": "error",
            "error": "Forecasting is not supported by the available tools.",
        },
    )

    assert result.route is None
    assert result.python_operation is None
    assert result.error == ToolRoutingError(
        code="unsupported_route",
        message="Forecasting is not supported by the available tools.",
    )


def test_routing_output_containing_sql_is_rejected() -> None:
    result = route_question(
        "How many orders are there?",
        lambda prompt: {
            "status": "success",
            "route": "sql_only",
            "python_operation": None,
            "reason": "Use SQL.",
            "sql": "SELECT COUNT(*) FROM orders",
        },
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_model_output"
