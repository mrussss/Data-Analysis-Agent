"""Interfaces regression tests."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import fields
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

import data_analysis_agent.__main__ as cli_module
import data_analysis_agent.api as api_module
import data_analysis_agent.integration_tools as integration_tools
from data_analysis_agent import mcp_server
from data_analysis_agent.analysis_planner import PythonAnalysisPlan
from data_analysis_agent.answer_synthesis import AnswerSynthesis
from data_analysis_agent.final_answer_service import FinalAnswerResult
from data_analysis_agent.metric_catalog import METRIC_DEFINITIONS_V2
from data_analysis_agent.multi_tool_service import MultiToolQuestionError, MultiToolQuestionResult
from data_analysis_agent.natural_language_answer import NaturalLanguageAnswer
from data_analysis_agent.observability import RequestObservability
from data_analysis_agent.python_analysis import GrowthPoint, GrowthResult, PythonAnalysisResult
from data_analysis_agent.question_service import QuestionAnswerResult
from data_analysis_agent.result_validation import ResultValidation, ValidationIssue
from data_analysis_agent.sql_executor import SQLExecutionError, SQLResult
from data_analysis_agent.streamlit_view import (
    extract_analysis_details,
    extract_growth_chart_data,
    primary_answer_text,
    synthesis_is_blocked,
)
from data_analysis_agent.tool_router import ToolRouteDecision, ToolRoutingError
from data_analysis_agent.validated_question_service import ValidatedQuestionResult


# Api

QUESTION = "How many orders are in the dataset?"


CHINESE_QUESTION = "商品成交金额每个月的环比变化怎么样？"


def _fake_model(prompt: str) -> dict[str, object]:
    raise AssertionError(f"Unexpected model call: {prompt}")


def _fake_narrative_model(prompt: str) -> str:
    return "共有 99441 个订单。"


def _sql_result() -> SQLResult:
    return SQLResult(
        executed_sql="SELECT COUNT(*) AS order_count FROM orders",
        columns=("order_count",),
        rows=((99441,),),
        returned_row_count=1,
        truncated=False,
        status="success",
        error=None,
    )


def _final_result(
    *,
    route: str = "sql_only",
    warnings: tuple[str, ...] = (),
    repaired: bool = False,
    answer: str = "Result: 99441",
    natural_language_answer: NaturalLanguageAnswer | None = None,
    observability: RequestObservability | None = None,
) -> FinalAnswerResult:
    sql_result = _sql_result()
    sql_answer = (
        QuestionAnswerResult(
            question=QUESTION,
            generated_sql="SELECT COUNT(*) AS order_count FROM orders",
            status="success",
            execution_result=sql_result,
            generation_error=None,
            execution_error=None,
            repaired_sql=(
                "SELECT COUNT(*) AS order_count FROM orders"
                if repaired
                else None
            ),
            repair_attempted=repaired,
            repair_error=None,
        )
        if route == "sql_only"
        else None
    )
    plan = (
        PythonAnalysisPlan(
            question=QUESTION,
            python_operation="calculate_growth",
            sql="SELECT month, value FROM monthly_values ORDER BY month",
            python_columns=("month", "value"),
            status="success",
            error=None,
        )
        if route == "sql_then_python"
        else None
    )
    python_result = (
        PythonAnalysisResult(
            operation="calculate_growth",
            status="success",
            result=GrowthResult(
                points=(GrowthPoint("2017-01", 10.0, None, None, None),),
                period_count=1,
            ),
            error=None,
        )
        if route == "sql_then_python"
        else None
    )
    result = MultiToolQuestionResult(
        question=QUESTION,
        route_decision=ToolRouteDecision(
            question=QUESTION,
            route=route,
            python_operation=(
                "calculate_growth" if route == "sql_then_python" else None
            ),
            reason="deterministic test route",
            status="success",
            error=None,
        ),
        status="success",
        sql_answer_result=sql_answer,
        analysis_plan=plan,
        sql_result=sql_result,
        python_result=python_result,
        error=None,
    )
    validation = ResultValidation(
        "valid_with_warnings" if warnings else "valid",
        tuple(
            ValidationIssue("test_warning", "warning", warning)
            for warning in warnings
        ),
    )
    return FinalAnswerResult(
        validated_result=ValidatedQuestionResult(result, validation),
        synthesis=AnswerSynthesis("success", answer, warnings),
        natural_language_answer=natural_language_answer,
        observability=observability,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    database_path = tmp_path / "olist.duckdb"
    database_path.touch()
    monkeypatch.setattr(api_module, "DEFAULT_DATABASE_PATH", database_path)
    monkeypatch.setattr(
        api_module,
        "create_natural_language_model",
        lambda: _fake_narrative_model,
    )
    return TestClient(api_module.app, raise_server_exceptions=False)


def _configure_result(
    monkeypatch: pytest.MonkeyPatch,
    final_result: FinalAnswerResult,
) -> None:
    monkeypatch.setattr(api_module, "create_model", lambda: _fake_model)
    monkeypatch.setattr(
        api_module,
        "answer_question_for_user",
        lambda *args, **kwargs: final_result,
    )


def _unexpected(*args: object, **kwargs: object) -> object:
    raise AssertionError("Core tool must not be called directly by the API adapter.")


def test_analyze_successful_sql_only_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_result(monkeypatch, _final_result())

    response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["route"] == "sql_only"
    assert body["validation"] == "valid"
    assert body["analysis_tool"] is None


def test_analyze_successful_sql_then_python_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_result(monkeypatch, _final_result(route="sql_then_python"))

    response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["route"] == "sql_then_python"
    assert response.json()["analysis_tool"] == "calculate_growth"


def test_chinese_question_is_passed_unchanged_with_zh_locale(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(api_module, "create_model", lambda: _fake_model)

    def answer(
        database_path,
        question,
        model,
        *,
        locale="en",
        natural_language_model=None,
    ):
        calls.append(
            (database_path, question, model, locale, natural_language_model)
        )
        return _final_result(answer="结果：99441")

    monkeypatch.setattr(api_module, "answer_question_for_user", answer)

    response = client.post("/analyze", json={"question": CHINESE_QUESTION})

    assert response.status_code == 200
    assert response.json()["answer"] == "结果：99441"
    assert calls == [
        (
            api_module.DEFAULT_DATABASE_PATH,
            CHINESE_QUESTION,
            _fake_model,
            "zh-CN",
            _fake_narrative_model,
        )
    ]


def test_request_rejects_api_key_and_route_override_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "create_model", _unexpected)
    monkeypatch.setattr(api_module, "create_natural_language_model", _unexpected)
    monkeypatch.setattr(api_module, "answer_question_for_user", _unexpected)

    response = client.post(
        "/analyze",
        json={
            "question": QUESTION,
            "api_key": "secret",
            "route": "sql_only",
            "locale": "zh-CN",
        },
    )

    assert response.status_code == 422


def test_narrative_field_does_not_replace_deterministic_answer_or_evidence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    narrative = NaturalLanguageAnswer("success", "共有 99441 个订单。", None)
    _configure_result(
        monkeypatch,
        _final_result(natural_language_answer=narrative),
    )

    body = client.post("/analyze", json={"question": CHINESE_QUESTION}).json()

    assert body["answer"] == "Result: 99441"
    assert body["natural_language_answer"] == "共有 99441 个订单。"
    assert body["route"] == "sql_only"
    assert body["validation"] == "valid"
    assert body["analysis_tool"] is None
    assert body["evidence"]["generated_sql"] == (
        "SELECT COUNT(*) AS order_count FROM orders"
    )
    assert body["evidence"]["trace"][0]["stage"] == "routing"


def test_response_excludes_secrets_prompts_and_held_out_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_result(monkeypatch, _final_result())

    body = client.post("/analyze", json={"question": QUESTION}).text.lower()

    assert "api_key" not in body
    assert "deepseek_api_key" not in body
    assert "prompt" not in body
    assert "held_out" not in body
    assert "reference_sql" not in body


def test_sql_only_evidence_contains_actual_sql_not_planner_sql(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_result(monkeypatch, _final_result(repaired=True))

    evidence = client.post("/analyze", json={"question": QUESTION}).json()[
        "evidence"
    ]

    assert evidence["generated_sql"] == "SELECT COUNT(*) AS order_count FROM orders"
    assert evidence["repaired_sql"] == "SELECT COUNT(*) AS order_count FROM orders"
    assert evidence["planner_sql"] is None
    assert evidence["python_columns"] == []


def test_provider_initialization_failure_returns_safe_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "create_model",
        lambda: (_ for _ in ()).throw(
            RuntimeError("DEEPSEEK_API_KEY=super-secret traceback")
        ),
    )
    monkeypatch.setattr(api_module, "answer_question_for_user", _unexpected)

    response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis service failed."}
    assert "super-secret" not in response.text
    assert "traceback" not in response.text


def test_unexpected_agent_exception_returns_safe_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "create_model", lambda: _fake_model)
    monkeypatch.setattr(
        api_module,
        "answer_question_for_user",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis service failed."}
    assert "private detail" not in response.text


def test_missing_database_returns_500_before_provider_creation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(api_module, "DEFAULT_DATABASE_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(api_module, "create_model", _unexpected)
    monkeypatch.setattr(api_module, "create_natural_language_model", _unexpected)
    monkeypatch.setattr(api_module, "answer_question_for_user", _unexpected)

    response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis database is unavailable."}


# Mcp server

PROJECT_ROOT = Path(__file__).resolve().parents[1]


SOURCE_ROOT = PROJECT_ROOT / "src"


DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "olist.duckdb"


EXPECTED_TOOL_NAMES = {
    "inspect_schema",
    "run_readonly_sql",
    "get_metric_definition",
}


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


async def _list_tools(server: Any) -> list[Any]:
    async with Client(server) as client:
        return (await client.list_tools()).tools


async def _call_tool(
    server: Any,
    name: str,
    arguments: dict[str, object],
) -> Any:
    async with Client(server) as client:
        return await client.call_tool(name, arguments)


def test_server_registers_only_the_three_expected_tools() -> None:
    server = mcp_server.create_mcp_server(DATABASE_PATH)

    tools = _run(_list_tools(server))

    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES


def test_tool_input_schemas_expose_only_the_allowed_arguments() -> None:
    server = mcp_server.create_mcp_server(DATABASE_PATH)
    tools = {tool.name: tool for tool in _run(_list_tools(server))}

    assert tools["inspect_schema"].input_schema.get("properties") == {}
    assert tools["inspect_schema"].input_schema.get("required", []) == []
    assert set(tools["run_readonly_sql"].input_schema["properties"]) == {
        "sql"
    }
    assert tools["run_readonly_sql"].input_schema["required"] == ["sql"]
    assert set(
        tools["get_metric_definition"].input_schema["properties"]
    ) == {"identifier"}
    assert tools["get_metric_definition"].input_schema["required"] == [
        "identifier"
    ]

    all_schema_text = repr(
        {name: tool.input_schema for name, tool in tools.items()}
    )
    assert "db_path" not in all_schema_text
    assert "database_path" not in all_schema_text
    assert "max_rows" not in all_schema_text


def test_caller_arguments_cannot_change_bound_path_or_row_limit(
    monkeypatch: Any,
) -> None:
    calls: dict[str, object] = {}
    trusted_path = Path("trusted.duckdb")

    def fake_run_readonly_sql_tool(database_path: str | Path) -> Any:
        calls["bound_path"] = Path(database_path)

        def run(sql: str) -> dict[str, object]:
            calls["sql"] = sql
            return {"status": "success"}

        return run

    monkeypatch.setattr(
        mcp_server,
        "run_readonly_sql_tool",
        fake_run_readonly_sql_tool,
    )
    server = mcp_server.create_mcp_server(trusted_path)

    result = _run(
        _call_tool(
            server,
            "run_readonly_sql",
            {
                "sql": "SELECT 1",
                "db_path": "attacker.duckdb",
                "database_path": "attacker-2.duckdb",
                "max_rows": 1_000_000,
            },
        )
    )

    assert result.structured_content == {"status": "success"}
    assert calls == {
        "bound_path": trusted_path,
        "sql": "SELECT 1",
    }


def test_real_stdio_protocol_initialize_list_and_call() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "data_analysis_agent.mcp_server"],
        cwd=PROJECT_ROOT,
        env={
            "PYTHONPATH": str(SOURCE_ROOT),
            mcp_server.DATABASE_PATH_ENV_VAR: str(DATABASE_PATH),
        },
    )

    async def smoke_test() -> None:
        with open(os.devnull, "w", encoding="utf-8") as error_log:
            async with Client(
                stdio_client(server_parameters, errlog=error_log),
                read_timeout_seconds=10,
            ) as client:
                tools = (await client.list_tools()).tools
                assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES
                result = await client.call_tool(
                    "get_metric_definition",
                    {"identifier": "average_review_score"},
                )
                assert result.is_error is False
                assert result.structured_content is not None
                assert result.structured_content["status"] == "success"
                assert result.structured_content["metric"]["identifier"] == (
                    "average_review_score"
                )

    _run(smoke_test())


# Integration tools

def test_schema_tool_arguments_cannot_override_database_path(tmp_path: Path) -> None:
    tool = integration_tools.inspect_schema_tool(tmp_path / "trusted.duckdb")

    with pytest.raises(TypeError):
        tool(database_path=tmp_path / "other.duckdb")  # type: ignore[call-arg]


def test_sql_tool_arguments_cannot_override_path_or_row_limit(
    tmp_path: Path,
) -> None:
    tool = integration_tools.run_readonly_sql_tool(tmp_path / "trusted.duckdb")

    with pytest.raises(TypeError):
        tool(  # type: ignore[call-arg]
            sql="SELECT 1",
            database_path=tmp_path / "other.duckdb",
        )
    with pytest.raises(TypeError):
        tool(sql="SELECT 1", max_rows=10_000)  # type: ignore[call-arg]


def test_sql_result_is_json_safe_for_supported_duckdb_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sql_result = SQLResult(
        executed_sql="SELECT values",
        columns=("amount", "order_date", "created_at"),
        rows=(
            (
                Decimal("123.4500"),
                date(2024, 1, 2),
                datetime(2024, 1, 2, 3, 4, 5),
            ),
        ),
        returned_row_count=1,
        truncated=False,
        status="success",
        error=None,
    )
    monkeypatch.setattr(
        integration_tools,
        "run_readonly_sql",
        lambda database_path, sql, max_rows: sql_result,
    )

    result = integration_tools.run_readonly_sql_tool(
        tmp_path / "trusted.duckdb"
    )("SELECT values")

    assert result == {
        "status": "success",
        "executed_sql": "SELECT values",
        "columns": ["amount", "order_date", "created_at"],
        "rows": [["123.4500", "2024-01-02", "2024-01-02T03:04:05"]],
        "returned_row_count": 1,
        "truncated": False,
        "error": None,
    }
    json.dumps(result, allow_nan=False)


def test_sql_error_contract_preserves_executor_safety_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sql_result = SQLResult(
        executed_sql="DROP TABLE orders",
        columns=(),
        rows=(),
        returned_row_count=0,
        truncated=False,
        status="error",
        error=SQLExecutionError(
            "unsafe_sql",
            "Only SELECT or WITH ... SELECT statements are allowed.",
        ),
    )
    monkeypatch.setattr(
        integration_tools,
        "run_readonly_sql",
        lambda database_path, sql, max_rows: sql_result,
    )

    result = integration_tools.run_readonly_sql_tool(
        tmp_path / "trusted.duckdb"
    )("DROP TABLE orders")

    assert result["status"] == "error"
    assert result["error"] == {
        "code": "unsafe_sql",
        "message": "Only SELECT or WITH ... SELECT statements are allowed.",
    }


def test_metric_identifier_uses_exact_v2_definition() -> None:
    identifier = "average_delivery_duration_days"
    metric = next(
        item for item in METRIC_DEFINITIONS_V2 if item.identifier == identifier
    )

    result = integration_tools.get_metric_definition_tool(identifier)

    expected_metric = {
        field.name: (
            list(value)
            if isinstance((value := getattr(metric, field.name)), tuple)
            else value
        )
        for field in fields(metric)
    }
    assert result == {"status": "success", "metric": expected_metric}
    assert integration_tools.get_metric_definition_tool(
        identifier.upper()
    )["status"] == "not_found"
    assert integration_tools.get_metric_definition_tool(
        f" {identifier}"
    )["status"] == "not_found"
    json.dumps(result, allow_nan=False)


# Streamlit view

def streamlit_view_sql_result() -> SQLResult:
    return SQLResult(
        executed_sql="SELECT executed",
        columns=("value",),
        rows=((1,),),
        returned_row_count=1,
        truncated=False,
        status="success",
        error=None,
    )


def streamlit_view_final_result(
    *,
    route: str = "sql_only",
    repaired: bool = False,
    blocked: bool = False,
    warnings: tuple[str, ...] = (),
    validation_status: str = "valid",
    operation: str = "calculate_growth",
    python_payload: object | None = None,
    natural_language_answer: NaturalLanguageAnswer | None = None,
) -> FinalAnswerResult:
    sql_result = streamlit_view_sql_result()
    sql_answer = (
        QuestionAnswerResult(
            question="test question",
            generated_sql="SELECT generated",
            status="success",
            execution_result=sql_result,
            generation_error=None,
            execution_error=None,
            repaired_sql="SELECT repaired" if repaired else None,
            repair_attempted=repaired,
            repair_error=None,
        )
        if route == "sql_only"
        else None
    )
    plan = (
        PythonAnalysisPlan(
            question="test question",
            python_operation=operation,
            sql="SELECT month, value FROM facts ORDER BY month",
            python_columns=("month", "value"),
            status="success",
            error=None,
        )
        if route == "sql_then_python"
        else None
    )
    result = MultiToolQuestionResult(
        question="REFERENCE SQL MUST NOT APPEAR",
        route_decision=ToolRouteDecision(
            question="test question",
            route=route,
            python_operation=operation if plan is not None else None,
            reason="test",
            status="success",
            error=None,
        ),
        status="success",
        sql_answer_result=sql_answer,
        analysis_plan=plan,
        sql_result=sql_result,
        python_result=(
            PythonAnalysisResult(
                operation=operation,
                status="success",
                result=python_payload,
                error=None,
            )
            if python_payload is not None
            else None
        ),
        error=None,
    )
    return FinalAnswerResult(
        validated_result=ValidatedQuestionResult(
            result=result,
            validation=ResultValidation(
                validation_status,
                (
                    ValidationIssue("test", "warning", "Test warning.")
                    if validation_status == "valid_with_warnings"
                    else ValidationIssue("test", "error", "Test error.")
                ,)
                if validation_status != "valid"
                else (),
            ),
        ),
        synthesis=AnswerSynthesis(
            "blocked" if blocked else "success",
            "Blocked." if blocked else "Result: 1",
            warnings,
        ),
        natural_language_answer=natural_language_answer,
    )


def test_details_do_not_include_reference_or_result_rows() -> None:
    details = extract_analysis_details(streamlit_view_final_result())
    serialized = repr(details)

    assert "REFERENCE SQL" not in serialized
    assert repr(((1,),)) not in serialized


def test_primary_answer_uses_deterministic_fallback_after_model_failure() -> None:
    final = streamlit_view_final_result(
        natural_language_answer=NaturalLanguageAnswer(
            "fallback",
            "Result: 1",
            "model_error: RuntimeError",
        )
    )

    assert primary_answer_text(final) == "Result: 1"


def test_blocked_result_keeps_deterministic_blocked_message() -> None:
    final = streamlit_view_final_result(blocked=True, validation_status="invalid")

    assert synthesis_is_blocked(final) is True
    assert primary_answer_text(final) == "Blocked."


def _growth_result(*points: GrowthPoint) -> GrowthResult:
    return GrowthResult(points=points, period_count=len(points))


def test_growth_chart_extracts_periods_and_values() -> None:
    payload = _growth_result(
        GrowthPoint("2017-01", 10.0, None, None, None),
        GrowthPoint("2017-02", 12.5, 10.0, 2.5, 0.25),
    )

    chart = extract_growth_chart_data(
        streamlit_view_final_result(route="sql_then_python", python_payload=payload)
    )

    assert chart is not None
    assert chart.periods == ("2017-01", "2017-02")
    assert chart.values == (10.0, 12.5)


# Cli

cli_QUESTION = "How many orders are in the dataset?"


GENERATED_SQL = "SELECT COUNT(*) AS order_count FROM orders"


REPAIRED_SQL = "SELECT COUNT(*) AS order_count FROM orders;"


def _database_path(tmp_path: Path) -> Path:
    path = tmp_path / "olist.duckdb"
    path.touch()
    return path


def _input_from(*values: str):
    responses = iter(values)
    return lambda prompt: next(responses)


def _route(
    route: str | None,
    operation: str | None = None,
    *,
    status: str = "success",
    error: ToolRoutingError | None = None,
) -> ToolRouteDecision:
    return ToolRouteDecision(
        question=cli_QUESTION,
        route=route,
        python_operation=operation,
        reason="Test route." if status == "success" else None,
        status=status,
        error=error,
    )


def _sql_execution(
    sql: str = GENERATED_SQL,
    rows: tuple[tuple[object, ...], ...] = ((99_441,),),
) -> SQLResult:
    return SQLResult(
        executed_sql=sql,
        columns=("order_count",),
        rows=rows,
        returned_row_count=len(rows),
        truncated=False,
        status="success",
        error=None,
    )


def _sql_only_result(*, repaired: bool = False) -> MultiToolQuestionResult:
    executed_sql = REPAIRED_SQL if repaired else GENERATED_SQL
    execution = _sql_execution(executed_sql)
    answer = QuestionAnswerResult(
        question=cli_QUESTION,
        generated_sql=GENERATED_SQL,
        status="success",
        execution_result=execution,
        generation_error=None,
        execution_error=None,
        repaired_sql=REPAIRED_SQL if repaired else None,
        repair_attempted=repaired,
        repair_error=None,
    )
    return MultiToolQuestionResult(
        question=cli_QUESTION,
        route_decision=_route("sql_only"),
        status="success",
        sql_answer_result=answer,
        analysis_plan=None,
        sql_result=execution,
        python_result=None,
        error=None,
    )


def _analysis_plan(
    operation: str,
    sql: str,
    columns: tuple[str, ...],
) -> PythonAnalysisPlan:
    return PythonAnalysisPlan(
        question=cli_QUESTION,
        python_operation=operation,
        sql=sql,
        python_columns=columns,
        status="success",
        error=None,
    )


def _python_success_result(
    operation: str,
    sql: str,
    columns: tuple[str, ...],
    python_result: PythonAnalysisResult,
) -> MultiToolQuestionResult:
    sql_result = SQLResult(
        executed_sql=sql,
        columns=columns,
        rows=(),
        returned_row_count=0,
        truncated=False,
        status="success",
        error=None,
    )
    return MultiToolQuestionResult(
        question=cli_QUESTION,
        route_decision=_route("sql_then_python", operation),
        status="success",
        sql_answer_result=None,
        analysis_plan=_analysis_plan(operation, sql, columns),
        sql_result=sql_result,
        python_result=python_result,
        error=None,
    )


def _run_one_result(
    monkeypatch,
    tmp_path: Path,
    result: MultiToolQuestionResult,
) -> str:
    monkeypatch.setattr(
        cli_module,
        "answer_question_with_tools",
        lambda database_path, question, model: result,
    )
    output: list[str] = []
    exit_code = cli_module.run_cli(
        _database_path(tmp_path),
        input_fn=_input_from(cli_QUESTION, "exit"),
        output_fn=output.append,
        model_factory=lambda: object(),
    )
    assert exit_code == 0
    return "\n".join(output)


def test_sql_only_repair_success_prints_repaired_sql(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rendered = _run_one_result(
        monkeypatch,
        tmp_path,
        _sql_only_result(repaired=True),
    )

    assert f"Repaired SQL:\n{REPAIRED_SQL}" in rendered
    assert "Repair attempted: Yes" in rendered
    assert "Result:\n99441" in rendered


def test_calculate_growth_prints_structured_growth_table(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sql = (
        "SELECT month, total_item_transaction_value "
        "FROM monthly_item_values ORDER BY month"
    )
    python_result = PythonAnalysisResult(
        operation="calculate_growth",
        status="success",
        result=GrowthResult(
            points=(
                GrowthPoint("2016-09", 267.36, None, None, None),
                GrowthPoint(
                    "2016-10",
                    49_507.66,
                    267.36,
                    49_240.3,
                    184.17227707959307,
                ),
                GrowthPoint("2016-12", 10.9, None, None, None),
            ),
            period_count=3,
        ),
        error=None,
    )
    result = _python_success_result(
        "calculate_growth",
        sql,
        ("month", "total_item_transaction_value"),
        python_result,
    )

    rendered = _run_one_result(monkeypatch, tmp_path, result)

    assert "Route: SQL → Python" in rendered
    assert "Python analysis: calculate_growth" in rendered
    assert f"SQL:\n{sql}" in rendered
    assert "Python columns: month, total_item_transaction_value" in rendered
    assert "Growth result:" in rendered
    assert "Period | Value | Previous | Absolute Change | Growth Rate" in rendered
    assert "2016-09 | 267.36 | NULL | NULL | NULL" in rendered
    assert (
        "2016-10 | 49507.66 | 267.36 | 49240.3 | 184.17227707959307"
        in rendered
    )
    assert "2016-12 | 10.9 | NULL | NULL | NULL" in rendered
    assert "Period count: 3" in rendered
    assert "20%" not in rendered


@pytest.mark.parametrize(
    "stage",
    [
        "routing_error",
        "planning_error",
        "sql_execution_error",
        "truncated_analysis_input",
        "python_analysis_error",
    ],
)
def test_stage_aware_errors_are_printed(
    stage: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    route = (
        _route(
            None,
            status="error",
            error=ToolRoutingError("unsupported_route", "Unsupported request."),
        )
        if stage == "routing_error"
        else _route("sql_then_python", "describe")
    )
    result = MultiToolQuestionResult(
        question=cli_QUESTION,
        route_decision=route,
        status=stage,
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=None,
        python_result=None,
        error=MultiToolQuestionError(stage, f"Failure at {stage}."),
    )

    rendered = _run_one_result(monkeypatch, tmp_path, result)

    assert f"Error stage: {stage}" in rendered
    assert f"Error: Failure at {stage}." in rendered
    if stage == "truncated_analysis_input":
        assert "Python analysis was NOT executed." in rendered


def test_missing_database_prints_error_and_exits(tmp_path: Path) -> None:
    output: list[str] = []

    exit_code = cli_module.run_cli(
        tmp_path / "missing.duckdb",
        input_fn=_input_from("exit"),
        output_fn=output.append,
        model_factory=lambda: (_ for _ in ()).throw(
            AssertionError("provider must not initialize")
        ),
    )

    assert exit_code == 1
    assert "DuckDB database does not exist" in "\n".join(output)
