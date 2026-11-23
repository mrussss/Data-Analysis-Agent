"""Workflows regression tests."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

import data_analysis_agent.deepseek_provider as provider_module
import data_analysis_agent.final_answer_service as final_answer_service_service_module
import data_analysis_agent.function_calling_spike as spike
import data_analysis_agent.langgraph_workflow as workflow
import data_analysis_agent.multi_tool_service as explicit_workflow
import data_analysis_agent.multi_tool_service as service_module
import data_analysis_agent.validated_question_service as validated_question_service_service_module
from data_analysis_agent.analysis_planner import PythonAnalysisPlan, PythonAnalysisPlanError
from data_analysis_agent.answer_synthesis import AnswerSynthesis
from data_analysis_agent.deepseek_provider import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DeepSeekNaturalLanguageModel,
    DeepSeekTextToSQLModel,
)
from data_analysis_agent.execution_trace import build_execution_trace, ExecutionTrace, TraceStep
from data_analysis_agent.final_answer_service import answer_question_for_user, FinalAnswerResult
from data_analysis_agent.multi_tool_service import (
    answer_question_with_tools,
    MultiToolQuestionError,
    MultiToolQuestionResult,
)
from data_analysis_agent.natural_language_answer import (
    build_natural_language_prompt,
    generate_natural_language_answer,
    NaturalLanguageAnswer,
)
from data_analysis_agent.observability import (
    finalize_observability,
    LLMCallObservation,
    observe_stage,
    record_llm_call,
    RequestObservability,
    StageObservation,
    start_observability_request,
)
from data_analysis_agent.python_analysis import (
    ColumnDescription,
    PythonAnalysisError,
    PythonAnalysisResult,
)
from data_analysis_agent.question_service import QuestionAnswerResult
from data_analysis_agent.result_validation import ResultValidation, ValidationIssue
from data_analysis_agent.schema import ColumnSchema, DatabaseSchema, SchemaObject
from data_analysis_agent.sql_executor import SQLExecutionError, SQLResult
from data_analysis_agent.sql_generator import SQLGenerationError
from data_analysis_agent.sql_repair import SQLRepairError
from data_analysis_agent.tool_router import ToolRouteDecision, ToolRoutingError
from data_analysis_agent.validated_question_service import (
    answer_question_with_validation,
    ValidatedQuestionResult,
)


# Multi tool question service

SCHEMA = DatabaseSchema(
    objects=(
        SchemaObject(
            name="analysis_values",
            object_type="table",
            grain="one row per observation",
            columns=(
                ColumnSchema("price", "DOUBLE", False, False),
                ColumnSchema("freight_value", "DOUBLE", False, False),
                ColumnSchema("label", "VARCHAR", False, False),
            ),
        ),
    )
)


class QueueModel:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs: Iterator[dict[str, object]] = iter(outputs)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return next(self._outputs)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE analysis_values (
                price DOUBLE,
                freight_value DOUBLE,
                label VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO analysis_values VALUES
                (1.0, 2.0, 'a'),
                (2.0, 4.0, 'b'),
                (3.0, 6.0, 'c'),
                (4.0, 8.0, 'd')
            """
        )
    return path


def _sql_only_route() -> dict[str, object]:
    return {
        "status": "success",
        "route": "sql_only",
        "python_operation": None,
        "reason": "A count is SQL-native.",
    }


def _python_route(operation: str) -> dict[str, object]:
    return {
        "status": "success",
        "route": "sql_then_python",
        "python_operation": operation,
        "reason": f"The question requires {operation}.",
    }


def _analysis_plan(sql: str, columns: list[str]) -> dict[str, object]:
    return {
        "status": "success",
        "sql": sql,
        "python_columns": columns,
    }


def _unexpected_call(*args, **kwargs):
    raise AssertionError("This dependency must not be called on the selected route.")


def test_sql_only_reuses_existing_question_pipeline(
    monkeypatch,
    database_path: Path,
) -> None:
    model = QueueModel(
        [
            _sql_only_route(),
            {
                "status": "success",
                "sql": "SELECT COUNT(*) AS row_count FROM analysis_values",
            },
        ]
    )
    monkeypatch.setattr(
        service_module,
        "generate_python_analysis_plan",
        _unexpected_call,
    )
    monkeypatch.setattr(service_module, "run_python_analysis", _unexpected_call)

    result = answer_question_with_tools(
        database_path,
        "How many rows are there?",
        model,
        max_rows=25,
        schema=SCHEMA,
    )

    assert result.status == "success"
    assert result.route_decision.route == "sql_only"
    assert result.sql_answer_result is not None
    assert result.sql_answer_result.status == "success"
    assert result.sql_answer_result.generated_sql == (
        "SELECT COUNT(*) AS row_count FROM analysis_values"
    )
    assert result.sql_result == result.sql_answer_result.execution_result
    assert result.sql_result is not None
    assert result.sql_result.rows == ((4,),)
    assert result.analysis_plan is None
    assert result.python_result is None


def test_describe_route_executes_plan_sql_then_python(
    monkeypatch,
    database_path: Path,
) -> None:
    model = QueueModel(
        [
            _python_route("describe"),
            _analysis_plan(
                "SELECT price FROM analysis_values ORDER BY price",
                ["price"],
            ),
        ]
    )
    monkeypatch.setattr(service_module, "answer_question", _unexpected_call)

    result = answer_question_with_tools(
        database_path,
        "Give me descriptive statistics for price.",
        model,
        schema=SCHEMA,
    )

    assert result.status == "success"
    assert result.route_decision.route == "sql_then_python"
    assert result.route_decision.python_operation == "describe"
    assert result.sql_answer_result is None
    assert result.analysis_plan is not None
    assert result.analysis_plan.python_columns == ("price",)
    assert result.sql_result is not None
    assert result.sql_result.columns == ("price",)
    assert result.sql_result.rows == ((1.0,), (2.0,), (3.0,), (4.0,))
    assert result.python_result is not None
    assert result.python_result.status == "success"
    assert result.python_result.result == (
        ColumnDescription(
            column="price",
            count=4,
            mean=2.5,
            std=pytest.approx(1.2909944487358056),
            min=1.0,
            median=2.5,
            max=4.0,
        ),
    )


def test_planning_failure_stops_before_sql_execution(
    monkeypatch,
    database_path: Path,
) -> None:
    model = QueueModel(
        [
            _python_route("describe"),
            {
                "status": "error",
                "error": "No numeric source column is available.",
            },
        ]
    )
    monkeypatch.setattr(service_module, "run_readonly_sql", _unexpected_call)
    monkeypatch.setattr(service_module, "run_python_analysis", _unexpected_call)

    result = answer_question_with_tools(
        database_path,
        "Describe an unavailable value.",
        model,
        schema=SCHEMA,
    )

    assert result.status == "planning_error"
    assert result.analysis_plan is not None
    assert result.analysis_plan.status == "error"
    assert result.sql_result is None
    assert result.python_result is None


def test_truncated_analysis_input_never_runs_python(
    monkeypatch,
    database_path: Path,
) -> None:
    model = QueueModel(
        [
            _python_route("describe"),
            _analysis_plan("SELECT price FROM analysis_values", ["price"]),
        ]
    )
    monkeypatch.setattr(service_module, "run_python_analysis", _unexpected_call)

    result = answer_question_with_tools(
        database_path,
        "Describe all prices.",
        model,
        analysis_max_rows=2,
        schema=SCHEMA,
    )

    assert result.status == "truncated_analysis_input"
    assert result.sql_result is not None
    assert result.sql_result.returned_row_count == 2
    assert result.sql_result.truncated is True
    assert result.python_result is None
    assert result.error is not None
    assert "statistics over truncated data would be misleading" in result.error.message


def test_analysis_sql_still_passes_through_readonly_safety(
    monkeypatch,
    database_path: Path,
) -> None:
    model = QueueModel(
        [
            _python_route("describe"),
            _analysis_plan("DELETE FROM analysis_values", ["price"]),
        ]
    )
    monkeypatch.setattr(service_module, "run_python_analysis", _unexpected_call)

    result = answer_question_with_tools(
        database_path,
        "Describe price.",
        model,
        schema=SCHEMA,
    )

    assert result.status == "sql_execution_error"
    assert result.sql_result is not None
    assert result.sql_result.error is not None
    assert result.sql_result.error.code == "unsafe_sql"
    assert result.python_result is None


# Langgraph workflow

QUESTION = "Test question"


DATABASE_PATH = Path("test.duckdb")


langgraph_workflow_SCHEMA = DatabaseSchema(objects=())


def _model(prompt: str) -> dict[str, object]:
    raise AssertionError(f"Unexpected model call: {prompt}")


def _route(
    route: str = "sql_only",
    operation: str | None = None,
) -> ToolRouteDecision:
    return ToolRouteDecision(
        question=QUESTION,
        route=route,
        python_operation=operation,
        reason="deterministic test route",
        status="success",
        error=None,
    )


def _routing_error() -> ToolRouteDecision:
    return ToolRouteDecision(
        question=QUESTION,
        route=None,
        python_operation=None,
        reason=None,
        status="error",
        error=ToolRoutingError("unsupported_route", "Unsupported request."),
    )


def _sql_result(
    *,
    status: str = "success",
    truncated: bool = False,
) -> SQLResult:
    return SQLResult(
        executed_sql="SELECT value FROM facts",
        columns=("value",),
        rows=((1.0,), (2.0,)) if status == "success" else (),
        returned_row_count=2 if status == "success" else 0,
        truncated=truncated,
        status=status,
        error=(
            SQLExecutionError("execution_error", "SQL failed.")
            if status == "error"
            else None
        ),
    )


def _sql_answer(
    status: str = "success",
) -> QuestionAnswerResult:
    execution_result = _sql_result(status="error" if status == "execution_error" else "success")
    return QuestionAnswerResult(
        question=QUESTION,
        generated_sql="SELECT value FROM facts",
        status=status,
        execution_result=(
            execution_result if status in {"success", "execution_error", "repair_error"} else None
        ),
        generation_error=(
            SQLGenerationError("cannot_generate", "Cannot generate SQL.")
            if status == "generation_error"
            else None
        ),
        execution_error=(
            SQLExecutionError("execution_error", "SQL failed.")
            if status in {"execution_error", "repair_error"}
            else None
        ),
        repair_attempted=status == "repair_error",
        repair_error=(
            SQLRepairError("cannot_repair", "Cannot repair SQL.")
            if status == "repair_error"
            else None
        ),
    )


def _plan(status: str = "success") -> PythonAnalysisPlan:
    return PythonAnalysisPlan(
        question=QUESTION,
        python_operation="describe",
        sql="SELECT value FROM facts" if status == "success" else None,
        python_columns=("value",) if status == "success" else (),
        status=status,
        error=(
            PythonAnalysisPlanError("invalid_analysis_plan", "Planning failed.")
            if status == "error"
            else None
        ),
    )


def _python_result(status: str = "success") -> PythonAnalysisResult:
    return PythonAnalysisResult(
        operation="describe",
        status=status,
        result=(
            (
                ColumnDescription(
                    column="value",
                    count=2,
                    mean=1.5,
                    std=0.7071067811865476,
                    min=1.0,
                    median=1.5,
                    max=2.0,
                ),
            )
            if status == "success"
            else None
        ),
        error=(
            PythonAnalysisError("insufficient_data", "Python analysis failed.")
            if status == "error"
            else None
        ),
    )


def _unexpected(*args: object, **kwargs: object) -> object:
    raise AssertionError("This dependency must not be called on this graph path.")


def _patch_route(
    monkeypatch: pytest.MonkeyPatch,
    decision: ToolRouteDecision,
) -> None:
    monkeypatch.setattr(workflow, "route_question", lambda question, model: decision)


def _invoke(schema: DatabaseSchema = langgraph_workflow_SCHEMA) -> FinalAnswerResult:
    return workflow.answer_question_with_langgraph(
        DATABASE_PATH,
        QUESTION,
        _model,
        schema=schema,
    )


def test_conditional_route_does_not_enter_unselected_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route(monkeypatch, _route("sql_then_python", "describe"))
    monkeypatch.setattr(workflow, "answer_question", _unexpected)
    monkeypatch.setattr(workflow, "generate_python_analysis_plan", lambda *args: _plan())
    monkeypatch.setattr(workflow, "run_readonly_sql", lambda *args, **kwargs: _sql_result())
    monkeypatch.setattr(workflow, "run_python_analysis", lambda *args, **kwargs: _python_result())

    assert _invoke().validated_result.result.analysis_plan == _plan()


def test_routing_error_skips_all_execution_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route(monkeypatch, _routing_error())
    monkeypatch.setattr(workflow, "answer_question", _unexpected)
    monkeypatch.setattr(workflow, "generate_python_analysis_plan", _unexpected)
    monkeypatch.setattr(workflow, "run_readonly_sql", _unexpected)
    monkeypatch.setattr(workflow, "run_python_analysis", _unexpected)

    result = _invoke()

    assert result.validated_result.result.status == "routing_error"
    assert result.validated_result.validation.status == "invalid"
    assert result.synthesis.status == "blocked"


def test_truncated_analysis_input_skips_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_route(monkeypatch, _route("sql_then_python", "describe"))
    monkeypatch.setattr(workflow, "generate_python_analysis_plan", lambda *args: _plan())
    monkeypatch.setattr(workflow, "run_readonly_sql", lambda *args, **kwargs: _sql_result(truncated=True))
    monkeypatch.setattr(workflow, "run_python_analysis", _unexpected)

    result = _invoke()

    assert result.validated_result.result.status == "truncated_analysis_input"
    assert result.synthesis.status == "blocked"


def test_graph_state_excludes_secrets_prompts_and_evaluation_metadata() -> None:
    fields = set(workflow.AgentGraphState.__annotations__)

    assert "api_key" not in fields
    assert "prompt" not in fields
    assert "held_out_reference" not in fields
    assert "evaluation_metadata" not in fields


def test_sql_only_contract_matches_explicit_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _route()
    answer = _sql_answer()
    monkeypatch.setattr(workflow, "route_question", lambda *args: decision)
    monkeypatch.setattr(workflow, "answer_question", lambda *args, **kwargs: answer)
    monkeypatch.setattr(explicit_workflow, "route_question", lambda *args: decision)
    monkeypatch.setattr(explicit_workflow, "answer_question", lambda *args, **kwargs: answer)

    explicit = explicit_workflow.answer_question_with_tools(
        DATABASE_PATH,
        QUESTION,
        _model,
        schema=langgraph_workflow_SCHEMA,
    )
    graph_result = _invoke().validated_result.result

    assert graph_result == explicit


def test_sql_then_python_contract_matches_explicit_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _route("sql_then_python", "describe")
    plan = _plan()
    sql_result = _sql_result()
    python_result = _python_result()
    monkeypatch.setattr(workflow, "route_question", lambda *args: decision)
    monkeypatch.setattr(workflow, "generate_python_analysis_plan", lambda *args: plan)
    monkeypatch.setattr(workflow, "run_readonly_sql", lambda *args, **kwargs: sql_result)
    monkeypatch.setattr(workflow, "run_python_analysis", lambda *args, **kwargs: python_result)
    monkeypatch.setattr(explicit_workflow, "route_question", lambda *args: decision)
    monkeypatch.setattr(explicit_workflow, "generate_python_analysis_plan", lambda *args: plan)
    monkeypatch.setattr(explicit_workflow, "run_readonly_sql", lambda *args, **kwargs: sql_result)
    monkeypatch.setattr(explicit_workflow, "run_python_analysis", lambda *args, **kwargs: python_result)

    explicit = explicit_workflow.answer_question_with_tools(
        DATABASE_PATH,
        QUESTION,
        _model,
        schema=langgraph_workflow_SCHEMA,
    )
    graph_result = _invoke().validated_result.result

    assert graph_result == explicit


# Final answer service

def _multi_tool_result() -> MultiToolQuestionResult:
    return MultiToolQuestionResult(
        question="test question",
        route_decision=ToolRouteDecision(
            question="test question",
            route="sql_only",
            python_operation=None,
            reason="test",
            status="success",
            error=None,
        ),
        status="success",
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=SQLResult(
            executed_sql="SELECT 1 AS value",
            columns=("value",),
            rows=((1,),),
            returned_row_count=1,
            truncated=False,
            status="success",
            error=None,
        ),
        python_result=None,
        error=None,
    )


def _validated_result(
    status: str = "valid",
    issues: tuple[ValidationIssue, ...] = (),
) -> ValidatedQuestionResult:
    return ValidatedQuestionResult(
        result=_multi_tool_result(),
        validation=ResultValidation(status=status, issues=issues),
    )


def test_validation_service_and_synthesis_are_each_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_result()
    validation_calls = 0
    synthesis_calls = 0

    def fake_validation(*args: object, **kwargs: object) -> ValidatedQuestionResult:
        nonlocal validation_calls
        validation_calls += 1
        return validated

    def fake_synthesis(
        result: ValidatedQuestionResult,
        locale: str = "en",
    ) -> AnswerSynthesis:
        nonlocal synthesis_calls
        synthesis_calls += 1
        assert result is validated
        assert locale == "en"
        return AnswerSynthesis("success", "Result: 1", ())

    monkeypatch.setattr(
        final_answer_service_service_module,
        "answer_question_with_validation",
        fake_validation,
    )
    monkeypatch.setattr(final_answer_service_service_module, "synthesize_answer", fake_synthesis)

    answer_question_for_user("database.duckdb", "question", object())

    assert validation_calls == 1
    assert synthesis_calls == 1


def test_explicit_narrative_adds_one_call_after_one_validation_and_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_result()
    core_model = object()
    validation_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    synthesis_calls: list[tuple[ValidatedQuestionResult, str]] = []
    narrative_prompts: list[str] = []

    def validate(*args: object, **kwargs: object) -> ValidatedQuestionResult:
        validation_calls.append((args, kwargs))
        return validated

    def synthesize(
        result: ValidatedQuestionResult,
        locale: str = "en",
    ) -> AnswerSynthesis:
        synthesis_calls.append((result, locale))
        return AnswerSynthesis("success", "结果：1", ())

    def narrative_model(prompt: str) -> str:
        narrative_prompts.append(prompt)
        return "共有 1 条结果。"

    monkeypatch.setattr(final_answer_service_service_module, "answer_question_with_validation", validate)
    monkeypatch.setattr(final_answer_service_service_module, "synthesize_answer", synthesize)

    final = answer_question_for_user(
        "database.duckdb",
        "问题",
        core_model,
        locale="zh-CN",
        natural_language_model=narrative_model,
    )

    assert len(validation_calls) == 1
    assert validation_calls[0][0] == ("database.duckdb", "问题", core_model)
    assert "natural_language_model" not in validation_calls[0][1]
    assert synthesis_calls == [(validated, "zh-CN")]
    assert len(narrative_prompts) == 1
    assert final.natural_language_answer == NaturalLanguageAnswer(
        "success",
        "共有 1 条结果。",
        None,
    )
    assert final.synthesis.answer == "结果：1"


def test_invalid_result_does_not_call_explicit_narrative_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_result(
        "invalid",
        (ValidationIssue("failed", "error", "Failed validation."),),
    )
    monkeypatch.setattr(
        final_answer_service_service_module,
        "answer_question_with_validation",
        lambda *args, **kwargs: validated,
    )
    monkeypatch.setattr(
        final_answer_service_service_module,
        "synthesize_answer",
        lambda result, locale="en": AnswerSynthesis(
            "blocked",
            "无法生成可靠答案。",
            (),
        ),
    )

    final = answer_question_for_user(
        "database.duckdb",
        "问题",
        object(),
        locale="zh-CN",
        natural_language_model=lambda prompt: pytest.fail(
            "invalid result must not call narrative model"
        ),
    )

    assert final.natural_language_answer == NaturalLanguageAnswer(
        "skipped",
        None,
        None,
    )


# Validated question service

def _result(
    *,
    route: str = "sql_only",
    operation: str | None = None,
    status: str = "success",
) -> MultiToolQuestionResult:
    sql_result = (
        SQLResult(
            executed_sql="SELECT 1 AS value",
            columns=("value",),
            rows=((1,),),
            returned_row_count=1,
            truncated=False,
            status="success",
            error=None,
        )
        if status == "success"
        else None
    )
    return MultiToolQuestionResult(
        question="test question",
        route_decision=ToolRouteDecision(
            question="test question",
            route=route,
            python_operation=operation,
            reason="test",
            status="success",
            error=None,
        ),
        status=status,
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=sql_result,
        python_result=None,
        error=(
            None
            if status == "success"
            else MultiToolQuestionError("routing_error", "unsupported request")
        ),
    )


def test_structured_agent_failure_is_still_validated_and_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _result(status="routing_error")
    validation = ResultValidation(
        "invalid",
        (ValidationIssue("unsuccessful_pipeline", "error", "routing failed"),),
    )
    validated: list[MultiToolQuestionResult] = []
    monkeypatch.setattr(
        validated_question_service_service_module,
        "answer_question_with_tools",
        lambda *args, **kwargs: actual,
    )

    def fake_validate(result: MultiToolQuestionResult) -> ResultValidation:
        validated.append(result)
        return validation

    monkeypatch.setattr(validated_question_service_service_module, "validate_multi_tool_result", fake_validate)

    wrapped = answer_question_with_validation("database.duckdb", "question", object())

    assert validated == [actual]
    assert wrapped.result is actual
    assert wrapped.validation is validation


def test_invalid_validation_does_not_trigger_retry_or_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _result()
    validation = ResultValidation(
        "invalid",
        (ValidationIssue("truncated_result", "error", "Truncated."),),
    )
    execution_calls = 0
    validation_calls = 0

    def fake_answer(*args: object, **kwargs: object) -> MultiToolQuestionResult:
        nonlocal execution_calls
        execution_calls += 1
        return actual

    def fake_validate(result: MultiToolQuestionResult) -> ResultValidation:
        nonlocal validation_calls
        validation_calls += 1
        return validation

    monkeypatch.setattr(validated_question_service_service_module, "answer_question_with_tools", fake_answer)
    monkeypatch.setattr(validated_question_service_service_module, "validate_multi_tool_result", fake_validate)

    wrapped = answer_question_with_validation("database.duckdb", "question", object())

    assert execution_calls == 1
    assert validation_calls == 1
    assert wrapped.validation.status == "invalid"


# Execution trace

def execution_trace_route(
    route: str | None,
    operation: str | None = None,
    *,
    status: str = "success",
    error: ToolRoutingError | None = None,
    reason: str = "test reason",
) -> ToolRouteDecision:
    return ToolRouteDecision(
        question="test question",
        route=route,
        python_operation=operation,
        reason=reason if status == "success" else None,
        status=status,
        error=error,
    )


def execution_trace_sql_result(
    *,
    status: str = "success",
    rows: tuple[tuple[object, ...], ...] = ((1,),),
    columns: tuple[str, ...] = ("value",),
    error: SQLExecutionError | None = None,
) -> SQLResult:
    return SQLResult(
        executed_sql="SELECT executed",
        columns=columns,
        rows=rows,
        returned_row_count=len(rows),
        truncated=False,
        status=status,
        error=error,
    )


def execution_trace_sql_answer(
    *,
    status: str = "success",
    sql_result: SQLResult | None = None,
    generated_sql: str | None = "SELECT generated",
    generation_error: SQLGenerationError | None = None,
    repair_attempted: bool = False,
    repaired_sql: str | None = None,
) -> QuestionAnswerResult:
    return QuestionAnswerResult(
        question="test question",
        generated_sql=generated_sql,
        status=status,
        execution_result=sql_result,
        generation_error=generation_error,
        execution_error=sql_result.error if sql_result is not None else None,
        repaired_sql=repaired_sql,
        repair_attempted=repair_attempted,
        repair_error=None,
    )


def _final(
    result: MultiToolQuestionResult,
    *,
    validation_status: str = "valid",
    issues: tuple[ValidationIssue, ...] = (),
    synthesis_status: str = "success",
    synthesis_answer: str = "Result: 1",
    warnings: tuple[str, ...] = (),
) -> FinalAnswerResult:
    return FinalAnswerResult(
        validated_result=ValidatedQuestionResult(
            result=result,
            validation=ResultValidation(validation_status, issues),
        ),
        synthesis=AnswerSynthesis(synthesis_status, synthesis_answer, warnings),
    )


def _sql_only_result(
    *,
    answer: QuestionAnswerResult | None = None,
    sql_result: SQLResult | None = None,
    status: str = "success",
) -> MultiToolQuestionResult:
    actual_sql = sql_result or execution_trace_sql_result()
    actual_answer = answer or execution_trace_sql_answer(sql_result=actual_sql)
    return MultiToolQuestionResult(
        question="test question",
        route_decision=execution_trace_route("sql_only"),
        status=status,
        sql_answer_result=actual_answer,
        analysis_plan=None,
        sql_result=actual_sql if actual_answer.execution_result is not None else None,
        python_result=None,
        error=(
            None
            if status == "success"
            else MultiToolQuestionError("sql_generation_error", "generation failed")
        ),
    )


def _analysis_result(
    *,
    plan: PythonAnalysisPlan,
    sql_result: SQLResult | None = None,
    python_result: PythonAnalysisResult | None = None,
    status: str = "success",
) -> MultiToolQuestionResult:
    return MultiToolQuestionResult(
        question="test question",
        route_decision=execution_trace_route("sql_then_python", "calculate_growth"),
        status=status,
        sql_answer_result=None,
        analysis_plan=plan,
        sql_result=sql_result,
        python_result=python_result,
        error=(
            None
            if status == "success"
            else MultiToolQuestionError(
                "planning_error" if plan.status == "error" else "sql_execution_error",
                "analysis failed",
            )
        ),
    )


def _stage_names(trace: ExecutionTrace) -> tuple[str, ...]:
    return tuple(step.stage for step in trace.steps)


def _step(trace: ExecutionTrace, stage: str) -> TraceStep:
    return next(step for step in trace.steps if step.stage == stage)


def test_sql_only_successful_repair_adds_repair_step() -> None:
    sql_result = execution_trace_sql_result()
    answer = execution_trace_sql_answer(
        sql_result=sql_result,
        repair_attempted=True,
        repaired_sql="SELECT repaired",
    )
    trace = build_execution_trace(
        _final(_sql_only_result(answer=answer, sql_result=sql_result))
    )

    assert _stage_names(trace) == (
        "routing",
        "sql_generation",
        "sql_repair",
        "sql_execution",
        "validation",
        "answer_synthesis",
    )
    assert _step(trace, "sql_repair").status == "success"
    assert ("repaired_sql", "SELECT repaired") in _step(trace, "sql_repair").details


def test_planning_failure_does_not_add_execution_or_python() -> None:
    plan = PythonAnalysisPlan(
        "test question",
        "calculate_growth",
        None,
        (),
        "error",
        PythonAnalysisPlanError("invalid_model_output", "bad plan"),
    )
    result = _analysis_result(plan=plan, status="planning_error")

    trace = build_execution_trace(
        _final(result, validation_status="invalid", synthesis_status="blocked")
    )

    assert _stage_names(trace) == (
        "routing",
        "planning",
        "validation",
        "answer_synthesis",
    )
    assert _step(trace, "planning").status == "error"


def test_trace_does_not_include_prompt_answer_rows_or_credentials() -> None:
    secret = "DEEPSEEK_API_KEY=super-secret-key"
    sql_result = execution_trace_sql_result(rows=((secret,),))
    answer = execution_trace_sql_answer(sql_result=sql_result, generated_sql="SELECT safe_sql")
    result = _sql_only_result(answer=answer, sql_result=sql_result)
    result = MultiToolQuestionResult(
        question=result.question,
        route_decision=execution_trace_route("sql_only", reason=f"FULL PROMPT {secret}"),
        status=result.status,
        sql_answer_result=result.sql_answer_result,
        analysis_plan=None,
        sql_result=result.sql_result,
        python_result=None,
        error=None,
    )

    trace = build_execution_trace(
        _final(result, synthesis_answer=f"answer contains {secret}")
    )
    serialized = repr(trace)

    assert "FULL PROMPT" not in serialized
    assert "DEEPSEEK_API_KEY" not in serialized
    assert "super-secret-key" not in serialized
    assert "answer contains" not in serialized


def test_trace_builder_does_not_execute_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("trace builder must not execute tools")

    monkeypatch.setattr(
        "data_analysis_agent.sql_executor.run_readonly_sql",
        fail_if_called,
    )
    monkeypatch.setattr(
        "data_analysis_agent.python_analysis.run_python_analysis",
        fail_if_called,
    )
    monkeypatch.setattr(
        "data_analysis_agent.deepseek_provider.DeepSeekTextToSQLModel",
        fail_if_called,
    )

    trace = build_execution_trace(_final(_sql_only_result()))

    assert _stage_names(trace)[0] == "routing"


# Deepseek provider

class FakeCompletions:
    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
        usage: object | None = None,
    ):
        self.content = content
        self.error = error
        self.usage = usage
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=self.usage,
        )


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)


def test_provider_passes_prompt_model_and_json_output() -> None:
    content = '{"status":"success","sql":"SELECT COUNT(*) FROM orders"}'
    completions = FakeCompletions(content=content)
    model = DeepSeekTextToSQLModel(client=FakeClient(completions))

    returned_content = model("Generate JSON SQL for the provided schema.")

    assert returned_content == content
    assert completions.calls == [
        {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Generate JSON SQL for the provided schema.",
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 3072,
            "reasoning_effort": "low",
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    ]
    assert DEEPSEEK_MODEL == "deepseek-v4-flash"


def test_provider_uses_official_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    constructor_arguments: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str, max_retries: int):
            assert max_retries == 0
            constructor_arguments.update(api_key=api_key, base_url=base_url)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(provider_module, "OpenAI", FakeOpenAI)

    DeepSeekTextToSQLModel(env_path=tmp_path / "missing.env")

    assert constructor_arguments["api_key"] == "unit-test-placeholder"
    assert constructor_arguments["base_url"] == DEEPSEEK_BASE_URL
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"


def test_missing_api_key_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is required"):
        DeepSeekTextToSQLModel(env_path=tmp_path / "missing.env")


def test_provider_records_real_usage_from_response() -> None:
    completions = FakeCompletions(
        content='{"status":"success","sql":"SELECT 1"}',
        usage=SimpleNamespace(
            prompt_tokens=101,
            completion_tokens=12,
            total_tokens=113,
        ),
    )
    model = DeepSeekTextToSQLModel(client=FakeClient(completions))

    with start_observability_request() as collector:
        with observe_stage("sql_generation"):
            model("Generate SQL.")
        observation = finalize_observability(
            collector,
            route="sql_only",
            final_status="success",
            validation_status="valid",
        )

    call = observation.llm_calls[0]
    assert call.stage == "sql_generation"
    assert call.model == DEEPSEEK_MODEL
    assert call.prompt_tokens == 101
    assert call.completion_tokens == 12
    assert call.total_tokens == 113
    assert call.status == "success"
    assert call.latency_ms >= 0


def test_provider_does_not_estimate_tokens_when_usage_is_unavailable() -> None:
    model = DeepSeekNaturalLanguageModel(
        client=FakeClient(FakeCompletions(content="Natural answer.", usage=None))
    )

    with start_observability_request() as collector:
        with observe_stage("natural_language_synthesis"):
            model("Summarize.")
        observation = finalize_observability(
            collector,
            route="sql_only",
            final_status="success",
            validation_status="valid",
        )

    call = observation.llm_calls[0]
    assert call.prompt_tokens is None
    assert call.completion_tokens is None
    assert call.total_tokens is None
    assert observation.total_prompt_tokens is None
    assert observation.total_completion_tokens is None
    assert observation.total_tokens is None


# Natural language answer

natural_language_answer_QUESTION = "How many orders are in the dataset?"


SYNTHESIS = AnswerSynthesis("success", "Result: 99441", ())


def test_invalid_validation_skips_model() -> None:
    result = generate_natural_language_answer(
        natural_language_answer_QUESTION,
        "invalid",
        SYNTHESIS,
        lambda prompt: pytest.fail("model must not be called"),
    )

    assert result == NaturalLanguageAnswer("skipped", None, None)


def test_model_exception_falls_back_without_retry() -> None:
    calls = 0

    def model(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("private provider detail")

    result = generate_natural_language_answer(
        natural_language_answer_QUESTION,
        "valid",
        SYNTHESIS,
        model,
    )

    assert result == NaturalLanguageAnswer(
        "fallback",
        SYNTHESIS.answer,
        "model_error: RuntimeError",
    )
    assert calls == 1
    assert "private provider detail" not in result.error


def test_prompt_contains_evidence_only_and_no_causal_invention_constraints() -> None:
    prompt = build_natural_language_prompt(
        natural_language_answer_QUESTION,
        "valid",
        SYNTHESIS,
    )

    assert "Use ONLY the supplied validated evidence." in prompt
    assert "Do not recompute metrics." in prompt
    assert "Do not change numeric values" in prompt
    assert "Do not infer business causes or causal relationships." in prompt


# Observability

def _finalize(collector) -> RequestObservability:
    return finalize_observability(
        collector,
        route="sql_only",
        final_status="success",
        validation_status="valid",
    )


def test_request_isolation_and_stage_order_are_deterministic() -> None:
    with start_observability_request() as first_collector:
        with observe_stage("routing"):
            pass
        with observe_stage("sql_generation"):
            pass
        first = _finalize(first_collector)

    with start_observability_request() as second_collector:
        with observe_stage("planning"):
            pass
        second = _finalize(second_collector)

    assert tuple(stage.stage for stage in first.stages) == (
        "routing",
        "sql_generation",
    )
    assert tuple(stage.stage for stage in second.stages) == ("planning",)
    assert second.request_id != first.request_id


def test_complete_usage_is_aggregated_across_calls() -> None:
    with start_observability_request() as collector:
        record_llm_call(
            LLMCallObservation("routing", "model", 1.0, 10, 2, 12, "success")
        )
        record_llm_call(
            LLMCallObservation(
                "sql_generation",
                "model",
                2.0,
                20,
                3,
                23,
                "success",
            )
        )
        result = _finalize(collector)

    assert result.total_prompt_tokens == 30
    assert result.total_completion_tokens == 5
    assert result.total_tokens == 35


def test_partial_usage_makes_corresponding_aggregates_unavailable() -> None:
    with start_observability_request() as collector:
        record_llm_call(
            LLMCallObservation("routing", "model", 1.0, 10, 2, 12, "success")
        )
        record_llm_call(
            LLMCallObservation(
                "sql_generation",
                "model",
                2.0,
                None,
                None,
                None,
                "success",
            )
        )
        result = _finalize(collector)

    assert result.total_prompt_tokens is None
    assert result.total_completion_tokens is None
    assert result.total_tokens is None


def test_public_contract_cannot_store_prompt_response_or_secret() -> None:
    field_names = {
        field.name
        for contract in (RequestObservability, StageObservation, LLMCallObservation)
        for field in fields(contract)
    }

    assert "prompt" not in field_names
    assert "response" not in field_names
    assert "raw_response" not in field_names
    assert "api_key" not in field_names


def _create_database(tmp_path) -> str:
    database_path = tmp_path / "observability.duckdb"
    with duckdb.connect(str(database_path)):
        pass
    return str(database_path)


def test_repair_pipeline_records_repair_and_both_sql_attempts(tmp_path) -> None:
    database_path = _create_database(tmp_path)

    def model(prompt: str):
        if prompt.startswith("You route"):
            return {
                "status": "success",
                "route": "sql_only",
                "python_operation": None,
                "reason": "SQL is sufficient.",
            }
        if prompt.startswith("You generate"):
            return {"status": "success", "sql": "SELECT missing FROM missing"}
        return {"status": "success", "sql": "SELECT 1 AS value"}

    final = answer_question_for_user(
        database_path,
        "Return one value.",
        model,
        schema=DatabaseSchema(objects=()),
    )

    assert final.observability is not None
    assert tuple(stage.stage for stage in final.observability.stages) == (
        "routing",
        "sql_generation",
        "sql_execution",
        "sql_repair",
        "sql_execution",
        "validation",
        "answer_synthesis",
    )


# Function calling spike

class function_calling_spike_FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        recorded = dict(kwargs)
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            recorded["messages"] = list(messages)
        self.calls.append(recorded)
        if not self._responses:
            raise AssertionError("Unexpected fake Chat Completions request.")
        return self._responses.pop(0)


class function_calling_spike_FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = function_calling_spike_FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _tool_call(
    *,
    call_id: str = "call_sql_1",
    call_type: str = "function",
    name: str = spike.TOOL_NAME,
    arguments: object = '{"sql":"SELECT 1 AS value"}',
) -> object:
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _first_response(
    *,
    finish_reason: str = "tool_calls",
    tool_calls: list[object] | None = None,
) -> tuple[object, object]:
    calls = [_tool_call()] if tool_calls is None else tool_calls
    message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=calls,
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=message,
            )
        ]
    )
    return response, message


def _second_response(content: str = "The query returned one row.") -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _successful_sql_result() -> dict[str, object]:
    return {
        "status": "success",
        "executed_sql": "SELECT 1 AS value",
        "columns": ["value"],
        "rows": [[1]],
        "returned_row_count": 1,
        "truncated": False,
        "error": None,
    }


def _install_fake_adapters(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sql_result: dict[str, object] | None = None,
) -> dict[str, list[object]]:
    calls: dict[str, list[object]] = {
        "schema_bind": [],
        "schema_call": [],
        "sql_bind": [],
        "sql_call": [],
    }

    def fake_inspect_schema_tool(database_path: str | Path) -> Any:
        calls["schema_bind"].append(Path(database_path))

        def inspect() -> dict[str, object]:
            calls["schema_call"].append(True)
            return {
                "tables": [
                    {
                        "name": "orders",
                        "columns": [
                            {"name": "order_id", "data_type": "VARCHAR"}
                        ],
                    }
                ],
                "views": [],
            }

        return inspect

    def fake_run_readonly_sql_tool(database_path: str | Path) -> Any:
        calls["sql_bind"].append(Path(database_path))

        def run(sql: str) -> dict[str, object]:
            calls["sql_call"].append(sql)
            return (
                _successful_sql_result()
                if sql_result is None
                else sql_result
            )

        return run

    monkeypatch.setattr(
        spike,
        "inspect_schema_tool",
        fake_inspect_schema_tool,
    )
    monkeypatch.setattr(
        spike,
        "run_readonly_sql_tool",
        fake_run_readonly_sql_tool,
    )
    return calls


def _run_valid_spike(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[spike.FunctionCallingProof, function_calling_spike_FakeClient, dict[str, list[object]], object]:
    calls = _install_fake_adapters(monkeypatch)
    first_response, assistant_message = _first_response()
    client = function_calling_spike_FakeClient([first_response, _second_response()])

    proof = spike.run_function_calling_spike(
        "How many rows does this query return?",
        Path("trusted.duckdb"),
        client,
    )
    return proof, client, calls, assistant_message


def _assert_invalid_first_response(
    monkeypatch: pytest.MonkeyPatch,
    first_response: object,
    message: str,
) -> None:
    calls = _install_fake_adapters(monkeypatch)
    client = function_calling_spike_FakeClient([first_response])

    with pytest.raises(spike.FunctionCallingProtocolError, match=message):
        spike.run_function_calling_spike(
            "Run a safe query.",
            Path("trusted.duckdb"),
            client,
        )

    assert calls["sql_bind"] == []
    assert calls["sql_call"] == []
    assert len(client.completions.calls) == 1


def test_valid_message_tool_calls_drive_existing_sql_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, calls, _ = _run_valid_spike(monkeypatch)

    assert calls["sql_bind"] == [Path("trusted.duckdb")]
    assert calls["sql_call"] == ["SELECT 1 AS value"]


def test_multiple_tool_calls_fail_before_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, _ = _first_response(tool_calls=[_tool_call(), _tool_call()])
    _assert_invalid_first_response(
        monkeypatch,
        response,
        "exactly one tool call",
    )


def test_invalid_json_arguments_fail_before_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response, _ = _first_response(
        tool_calls=[_tool_call(arguments="not-json")]
    )
    _assert_invalid_first_response(
        monkeypatch,
        response,
        "must be valid JSON",
    )


@pytest.mark.parametrize("injected_name", ["db_path", "database_path", "max_rows"])
def test_path_and_row_limit_injection_fail_before_sql(
    monkeypatch: pytest.MonkeyPatch,
    injected_name: str,
) -> None:
    arguments = json.dumps({"sql": "SELECT 1", injected_name: "attacker"})
    response, _ = _first_response(
        tool_calls=[_tool_call(arguments=arguments)]
    )
    _assert_invalid_first_response(
        monkeypatch,
        response,
        "must contain only the 'sql' field",
    )


def test_second_request_preserves_assistant_tool_call_and_adds_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, client, _, assistant_message = _run_valid_spike(monkeypatch)

    second_request = client.completions.calls[1]
    messages = second_request["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 4
    assert messages[2] is assistant_message
    assert messages[2].tool_calls[0].id == "call_sql_1"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_sql_1"
    assert json.loads(messages[3]["content"]) == _successful_sql_result()
    assert second_request["tool_choice"] == "none"
    assert second_request["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
