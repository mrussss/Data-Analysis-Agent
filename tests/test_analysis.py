"""Analysis regression tests."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import pytest

from data_analysis_agent.answer_synthesis import synthesize_answer
from data_analysis_agent.multi_tool_service import MultiToolQuestionError, MultiToolQuestionResult
from data_analysis_agent.python_analysis import (
    CorrelationResult,
    GrowthPoint,
    GrowthResult,
    PythonAnalysisError,
    PythonAnalysisRequest,
    PythonAnalysisResult,
    run_python_analysis,
)
from data_analysis_agent.result_validation import (
    ResultValidation,
    validate_multi_tool_result,
    ValidationIssue,
)
from data_analysis_agent.sql_executor import SQLResult
from data_analysis_agent.tool_router import ToolRouteDecision
from data_analysis_agent.validated_question_service import ValidatedQuestionResult


# Python analysis

def test_describe_multiple_numeric_columns() -> None:
    result = run_python_analysis(
        columns=("x", "y"),
        rows=((1, 2), (2, 4), (3, 8)),
        request=PythonAnalysisRequest("describe", ("x", "y")),
    )

    assert result.status == "success"
    assert isinstance(result.result, tuple)
    assert tuple(item.column for item in result.result) == ("x", "y")
    assert result.result[0].mean == pytest.approx(2.0)
    assert result.result[1].median == pytest.approx(4.0)


def test_describe_ignores_null_values() -> None:
    result = run_python_analysis(
        columns=("value",),
        rows=((1,), (None,), (3,)),
        request=PythonAnalysisRequest("describe", ("value",)),
    )

    assert isinstance(result.result, tuple)
    assert result.result[0].count == 2
    assert result.result[0].mean == pytest.approx(2.0)
    assert result.result[0].std == pytest.approx(2**0.5)


def test_describe_rejects_non_numeric_column() -> None:
    result = run_python_analysis(
        columns=("label",),
        rows=(("one",), ("two",)),
        request=PythonAnalysisRequest("describe", ("label",)),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "non_numeric_column"


def test_correlation_uses_only_paired_non_null_rows() -> None:
    result = run_python_analysis(
        columns=("x", "y"),
        rows=((1, 2), (2, None), (None, 6), (3, 6)),
        request=PythonAnalysisRequest("correlation", ("x", "y")),
    )

    assert isinstance(result.result, CorrelationResult)
    assert result.result.paired_count == 2
    assert result.result.correlation == pytest.approx(1.0)


def test_correlation_rejects_non_numeric_column() -> None:
    result = run_python_analysis(
        columns=("x", "label"),
        rows=((1, "one"), (2, "two")),
        request=PythonAnalysisRequest("correlation", ("x", "label")),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "non_numeric_column"


def test_correlation_rejects_zero_variance() -> None:
    result = run_python_analysis(
        columns=("x", "y"),
        rows=((1, 2), (1, 4), (1, 6)),
        request=PythonAnalysisRequest("correlation", ("x", "y")),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "zero_variance"


def test_unsupported_operation_returns_structured_error() -> None:
    result = run_python_analysis(
        columns=("value",),
        rows=((1,), (2,)),
        request=PythonAnalysisRequest("regression", ("value",)),
    )

    assert result == PythonAnalysisResult(
        operation="regression",
        status="error",
        result=None,
        error=PythonAnalysisError(
            code="unsupported_operation",
            message="Unsupported Python analysis operation: regression",
        ),
    )


def test_malformed_table_returns_invalid_argument() -> None:
    result = run_python_analysis(
        columns=("x", "y"),
        rows=((1,), (2, 4)),
        request=PythonAnalysisRequest("describe", ("x",)),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_argument"


def test_calculate_growth_returns_complete_ordered_growth_series() -> None:
    result = run_python_analysis(
        columns=("month", "transaction_value"),
        rows=(("2018-01", 100.0), ("2018-02", 120.0), ("2018-03", 90.0)),
        request=PythonAnalysisRequest(
            "calculate_growth",
            ("month", "transaction_value"),
        ),
    )

    assert result.status == "success"
    assert result.error is None
    assert isinstance(result.result, GrowthResult)
    assert result.result.period_count == 3
    assert result.result.points == (
        GrowthPoint("2018-01", 100.0, None, None, None),
        GrowthPoint("2018-02", 120.0, 100.0, 20.0, pytest.approx(0.2)),
        GrowthPoint("2018-03", 90.0, 120.0, -30.0, pytest.approx(-0.25)),
    )


def test_calculate_growth_resets_comparison_after_calendar_month_gap() -> None:
    result = run_python_analysis(
        columns=("month", "value"),
        rows=(
            ("2016-09", 100),
            ("2016-10", 120),
            ("2016-12", 90),
            ("2017-01", 100),
        ),
        request=PythonAnalysisRequest("calculate_growth", ("month", "value")),
    )

    assert result.status == "success"
    assert isinstance(result.result, GrowthResult)
    assert result.result.points == (
        GrowthPoint("2016-09", 100.0, None, None, None),
        GrowthPoint("2016-10", 120.0, 100.0, 20.0, pytest.approx(0.2)),
        GrowthPoint("2016-12", 90.0, None, None, None),
        GrowthPoint(
            "2017-01",
            100.0,
            90.0,
            10.0,
            pytest.approx(10 / 90),
        ),
    )


def test_calculate_growth_treats_year_boundary_as_consecutive_months() -> None:
    result = run_python_analysis(
        columns=("month", "value"),
        rows=(("2025-12", 80), ("2026-01", 100)),
        request=PythonAnalysisRequest("calculate_growth", ("month", "value")),
    )

    assert isinstance(result.result, GrowthResult)
    assert result.result.points[1] == GrowthPoint(
        "2026-01",
        100.0,
        80.0,
        20.0,
        pytest.approx(0.25),
    )


def test_calculate_growth_treats_date_year_boundary_as_consecutive_months() -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=((date(2025, 12, 1), 80), (date(2026, 1, 1), 100)),
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert isinstance(result.result, GrowthResult)
    assert result.result.points[1].previous_value == 80.0
    assert result.result.points[1].growth_rate == pytest.approx(0.25)


def test_calculate_growth_does_not_infer_frequency_for_date_periods() -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=((date(2026, 1, 5), 100), (date(2026, 1, 12), 120)),
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert isinstance(result.result, GrowthResult)
    assert result.result.points[1] == GrowthPoint(
        date(2026, 1, 12),
        120.0,
        100.0,
        20.0,
        pytest.approx(0.2),
    )


def test_calculate_growth_supports_datetime_periods() -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=(
            (datetime(2018, 2, 1, 12), 110),
            (datetime(2018, 1, 1, 12), 100),
        ),
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert result.status == "success"
    assert isinstance(result.result, GrowthResult)
    assert result.result.points[1].growth_rate == pytest.approx(0.1)


def test_calculate_growth_ignores_null_metric_observation_without_zero_fill() -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=(
            ("2026-01", 100),
            ("2026-02", None),
            ("2026-03", 120),
            ("2026-04", 150),
        ),
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert result.status == "success"
    assert isinstance(result.result, GrowthResult)
    assert result.result.period_count == 3
    assert result.result.points == (
        GrowthPoint("2026-01", 100.0, None, None, None),
        GrowthPoint("2026-03", 120.0, None, None, None),
        GrowthPoint("2026-04", 150.0, 120.0, 30.0, pytest.approx(0.25)),
    )


def test_calculate_growth_keeps_absolute_change_when_previous_value_is_zero() -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=(("2018-01", 0), ("2018-02", 25)),
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert isinstance(result.result, GrowthResult)
    second_point = result.result.points[1]
    assert second_point.previous_value == 0.0
    assert second_point.absolute_change == 25.0
    assert second_point.growth_rate is None


@pytest.mark.parametrize(
    "periods",
    [
        (("January 2018", 100), ("February 2018", 120)),
        (("2018-1", 100), ("2018-02", 120)),
        (("0000-01", 100), ("0000-02", 120)),
        (("2018-01", 100), (date(2018, 2, 1), 120)),
        (("2018-01", 100), ("2018-01", 120)),
    ],
)
def test_calculate_growth_rejects_unreliable_period_representation(periods) -> None:
    result = run_python_analysis(
        columns=("period", "value"),
        rows=periods,
        request=PythonAnalysisRequest("calculate_growth", ("period", "value")),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_period_column"


# Result validation

def _sql_result(
    columns: tuple[str, ...] = ("value",),
    rows: tuple[tuple[object, ...], ...] = ((42,),),
    *,
    truncated: bool = False,
) -> SQLResult:
    return SQLResult(
        executed_sql="SELECT value",
        columns=columns,
        rows=rows,
        returned_row_count=len(rows),
        truncated=truncated,
        status="success",
        error=None,
    )


def _multi_tool_result(
    *,
    route: str = "sql_only",
    operation: str | None = None,
    sql_result: SQLResult | None = None,
    python_result: PythonAnalysisResult | None = None,
    status: str = "success",
) -> MultiToolQuestionResult:
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
        sql_result=_sql_result() if sql_result is None and status == "success" else sql_result,
        python_result=python_result,
        error=(
            None
            if status == "success"
            else MultiToolQuestionError("sql_execution_error", "query failed")
        ),
    )


def _growth_result(
    points: tuple[GrowthPoint, ...] | None = None,
    *,
    period_count: int | None = None,
) -> GrowthResult:
    actual_points = points or (
        GrowthPoint("2020-01", 10.0, None, None, None),
        GrowthPoint("2020-02", 15.0, 10.0, 5.0, 0.5),
    )
    return GrowthResult(
        points=actual_points,
        period_count=len(actual_points) if period_count is None else period_count,
    )


def _python_actual(
    operation: str,
    payload: object,
) -> MultiToolQuestionResult:
    return _multi_tool_result(
        route="sql_then_python",
        operation=operation,
        sql_result=_sql_result(("period", "value"), (("2020-01", 10.0),)),
        python_result=PythonAnalysisResult(
            operation=operation,
            status="success",
            result=payload,
            error=None,
        ),
    )


def _issue_codes(result: MultiToolQuestionResult) -> tuple[str, ...]:
    return tuple(
        issue.code for issue in validate_multi_tool_result(result).issues
    )


def test_successful_sql_only_without_sql_result_is_invalid() -> None:
    result = replace(_multi_tool_result(), sql_result=None)

    assert "missing_sql_result" in _issue_codes(result)


def test_empty_sql_result_is_warning() -> None:
    result = _multi_tool_result(sql_result=_sql_result(rows=()))

    validation = validate_multi_tool_result(result)

    assert validation.status == "valid_with_warnings"
    assert _issue_codes(result) == ("empty_result",)
    assert validation.issues[0].severity == "warning"


def test_truncated_sql_result_is_invalid() -> None:
    result = _multi_tool_result(sql_result=_sql_result(truncated=True))

    validation = validate_multi_tool_result(result)

    assert validation.status == "invalid"
    assert "truncated_result" in _issue_codes(result)


def test_duplicate_sql_columns_are_invalid() -> None:
    result = _multi_tool_result(
        sql_result=_sql_result(("value", "value"), ((1, 2),))
    )

    assert "duplicate_columns" in _issue_codes(result)


def test_sql_row_width_mismatch_is_invalid() -> None:
    result = _multi_tool_result(
        sql_result=_sql_result(("first", "second"), ((1,),))
    )

    assert "row_width_mismatch" in _issue_codes(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_sql_numeric_is_invalid(value: float) -> None:
    result = _multi_tool_result(sql_result=_sql_result(rows=((value,),)))

    assert "non_finite_sql_value" in _issue_codes(result)


def test_growth_period_count_mismatch_is_invalid() -> None:
    result = _python_actual(
        "calculate_growth",
        _growth_result(period_count=1),
    )

    assert "growth_period_count_mismatch" in _issue_codes(result)


@pytest.mark.parametrize(
    "periods",
    [
        ("2020-02", "2020-01"),
        (date(2020, 1, 1), date(2020, 1, 1)),
    ],
)
def test_growth_unordered_or_duplicate_periods_are_invalid(
    periods: tuple[object, object],
) -> None:
    points = (
        GrowthPoint(periods[0], 10.0, None, None, None),
        GrowthPoint(periods[1], 11.0, 10.0, 1.0, 0.1),
    )
    result = _python_actual("calculate_growth", _growth_result(points))

    assert "unordered_growth_periods" in _issue_codes(result)


def test_calculate_growth_with_wrong_payload_is_invalid() -> None:
    result = _python_actual(
        "calculate_growth",
        CorrelationResult("x", "y", 0.5, 3),
    )

    assert "python_payload_mismatch" in _issue_codes(result)


def test_python_operation_mismatch_is_invalid() -> None:
    result = _multi_tool_result(
        route="sql_then_python",
        operation="calculate_growth",
        sql_result=_sql_result(),
        python_result=PythonAnalysisResult(
            operation="correlation",
            status="success",
            result=CorrelationResult("x", "y", 0.5, 3),
            error=None,
        ),
    )

    assert "python_operation_mismatch" in _issue_codes(result)


# Answer synthesis

def answer_synthesis_sql_result(
    columns: tuple[str, ...] = ("value",),
    rows: tuple[tuple[object, ...], ...] = ((99441,),),
    *,
    executed_sql: str = "SELECT value",
) -> SQLResult:
    return SQLResult(
        executed_sql=executed_sql,
        columns=columns,
        rows=rows,
        returned_row_count=len(rows),
        truncated=False,
        status="success",
        error=None,
    )


def answer_synthesis_multi_tool_result(
    *,
    route: str = "sql_only",
    operation: str | None = None,
    sql_result: SQLResult | None = None,
    python_payload: object | None = None,
    status: str = "success",
    question: str = "test question",
) -> MultiToolQuestionResult:
    python_result = (
        PythonAnalysisResult(
            operation=operation or "unknown",
            status="success",
            result=python_payload,
            error=None,
        )
        if route == "sql_then_python"
        else None
    )
    return MultiToolQuestionResult(
        question=question,
        route_decision=ToolRouteDecision(
            question=question,
            route=route,
            python_operation=operation,
            reason="internal router reason",
            status="success",
            error=None,
        ),
        status=status,
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=sql_result or answer_synthesis_sql_result(),
        python_result=python_result,
        error=(
            None
            if status == "success"
            else MultiToolQuestionError("sql_execution_error", "query failed")
        ),
    )


def _validated(
    result: MultiToolQuestionResult,
    *,
    status: str = "valid",
    issues: tuple[ValidationIssue, ...] = (),
) -> ValidatedQuestionResult:
    return ValidatedQuestionResult(
        result=result,
        validation=ResultValidation(status=status, issues=issues),
    )


def answer_synthesis_growth_result() -> GrowthResult:
    return GrowthResult(
        points=(
            GrowthPoint("2020-01", 10.0, None, None, None),
            GrowthPoint("2020-02", 15.0, 10.0, 5.0, 0.5),
            GrowthPoint("2020-04", 12.0, None, None, None),
        ),
        period_count=3,
    )


def test_invalid_validation_blocks_answer() -> None:
    result = answer_synthesis_multi_tool_result(status="sql_execution_error")
    validation = _validated(
        result,
        status="invalid",
        issues=(ValidationIssue("unsuccessful_pipeline", "error", "Failed."),),
    )

    synthesis = synthesize_answer(validation)

    assert synthesis.status == "blocked"
    assert "cannot be generated" in synthesis.answer
    assert "sql_execution_error" in synthesis.answer
    assert "query failed" in synthesis.answer


def test_sql_scalar_ratio_is_not_converted_to_percentage() -> None:
    result = answer_synthesis_multi_tool_result(sql_result=answer_synthesis_sql_result(rows=((0.123,),)))

    answer = synthesize_answer(_validated(result)).answer

    assert answer == "Result: 0.123"
    assert "%" not in answer


def test_sql_multi_row_is_a_deterministic_table() -> None:
    result = answer_synthesis_multi_tool_result(
        sql_result=answer_synthesis_sql_result(
            ("order_status", "order_count"),
            (("delivered", 96478), ("shipped", 1107)),
        )
    )

    answer = synthesize_answer(_validated(result)).answer

    assert answer == (
        "order_status | order_count\n"
        "delivered | 96478\n"
        "shipped | 1107"
    )


def test_unknown_python_payload_is_blocked_without_repr() -> None:
    payload = object()
    result = answer_synthesis_multi_tool_result(
        route="sql_then_python",
        operation="unknown",
        python_payload=payload,
    )

    synthesis = synthesize_answer(_validated(result))

    assert synthesis.status == "blocked"
    assert synthesis.answer == "Unsupported result payload for answer synthesis."
    assert repr(payload) not in synthesis.answer


def test_synthesis_does_not_call_llm_database_or_python_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthesis must not execute external tools")

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

    synthesis = synthesize_answer(_validated(answer_synthesis_multi_tool_result()))

    assert synthesis.status == "success"


def test_answer_does_not_expose_internal_metadata() -> None:
    result = answer_synthesis_multi_tool_result(
        question="HELD-OUT METADATA SECRET",
        sql_result=answer_synthesis_sql_result(
            executed_sql="SELECT 'REFERENCE SQL SECRET' AS value",
        ),
    )

    answer = synthesize_answer(_validated(result)).answer

    assert "HELD-OUT" not in answer
    assert "REFERENCE SQL" not in answer
    assert "internal router reason" not in answer
    assert "Prompt" not in answer
    assert ".env" not in answer


def test_zh_growth_preserves_null_and_raw_ratio() -> None:
    result = answer_synthesis_multi_tool_result(
        route="sql_then_python",
        operation="calculate_growth",
        python_payload=answer_synthesis_growth_result(),
    )

    answer = synthesize_answer(_validated(result), locale="zh-CN").answer

    assert "2020-01 | 10.0 | NULL | NULL | NULL" in answer
    assert "2020-02 | 15.0 | 10.0 | 5.0 | 0.5" in answer
    assert "%" not in answer
