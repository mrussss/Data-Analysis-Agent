"""Evaluation regression tests."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

import data_analysis_agent.held_out_runner as runner_module
import scripts.evaluate_gold_set as evaluation_module
from data_analysis_agent import GOLD_QUESTIONS as gold_questions_GOLD_QUESTIONS
from data_analysis_agent.gold_questions import GOLD_QUESTIONS, LabelAlias, RankingComparison
from data_analysis_agent.held_out_evaluator import (
    evaluate_held_out_case,
    HeldOutCaseEvaluation,
    summarize_held_out_evaluations,
)
from data_analysis_agent.held_out_runner import format_held_out_report, run_held_out_evaluation
from data_analysis_agent.multi_tool_service import MultiToolQuestionError, MultiToolQuestionResult
from data_analysis_agent.multi_tool_test_questions import (
    MULTI_TOOL_TEST_QUESTIONS,
    MultiToolTestQuestion,
)
from data_analysis_agent.python_analysis import GrowthPoint, GrowthResult, PythonAnalysisResult
from data_analysis_agent.question_service import QuestionAnswerResult
from data_analysis_agent.sql_executor import run_readonly_sql, SQLExecutionError, SQLResult
from data_analysis_agent.sql_generator import SQLGenerationError
from data_analysis_agent.tool_router import ToolRouteDecision, ToolRoutingError
from scripts.baseline_queries import BASELINE_QUERIES
from scripts.build_duckdb import build_database
from scripts.evaluate_gold_set import (
    evaluate_gold_set,
    evaluate_result,
    results_match,
    ResultSnapshot,
)


# Evaluate gold set

def _sql_result(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> SQLResult:
    return SQLResult(
        executed_sql="SELECT test_result",
        columns=columns,
        rows=rows,
        returned_row_count=len(rows),
        truncated=False,
        status="success",
        error=None,
    )


def _answer_result(
    execution_result: SQLResult | None,
    *,
    status="success",
    generated_sql="SELECT test_result",
    generation_error=None,
    repair_attempted=False,
) -> QuestionAnswerResult:
    return QuestionAnswerResult(
        question="test question",
        generated_sql=generated_sql,
        status=status,
        execution_result=execution_result,
        generation_error=generation_error,
        execution_error=None,
        repair_attempted=repair_attempted,
    )


def test_results_match_integer_exactly_and_float_with_tolerance() -> None:
    expected_integer = ResultSnapshot(("value",), ((99_441,),))
    assert results_match(ResultSnapshot(("count",), ((99_441,),)), expected_integer)
    assert not results_match(ResultSnapshot(("count",), ((99_442,),)), expected_integer)

    expected_float = ResultSnapshot(("value",), ((Decimal("103.1234"),),))
    assert results_match(ResultSnapshot(("average",), ((103.13,),)), expected_float)
    assert not results_match(ResultSnapshot(("average",), ((103.14,),)), expected_float)


def test_results_match_reference_projection_and_ignore_row_order() -> None:
    expected = ResultSnapshot(
        ("state", "customer_count", "extra_metric"),
        (("SP", 10, Decimal("20.00")), ("RJ", 5, Decimal("8.00"))),
    )
    actual = ResultSnapshot(
        ("region", "count"),
        (("RJ", 5), ("SP", 10)),
    )

    assert results_match(actual, expected)


def test_order_sensitive_results_require_the_same_row_order() -> None:
    expected = ResultSnapshot(
        ("state", "customer_count"),
        (("SP", 100), ("RJ", 80)),
    )
    same_order = ResultSnapshot(
        ("region", "count"),
        (("SP", 100), ("RJ", 80)),
    )
    reversed_order = ResultSnapshot(
        ("region", "count"),
        (("RJ", 80), ("SP", 100)),
    )

    assert results_match(same_order, expected, order_sensitive=True)
    assert not results_match(reversed_order, expected, order_sensitive=True)
    assert results_match(reversed_order, expected, order_sensitive=False)


def test_order_sensitive_results_preserve_numeric_tolerance() -> None:
    expected = ResultSnapshot(
        ("state", "value"),
        (("SP", Decimal("100.004")), ("RJ", Decimal("80.004"))),
    )
    actual = ResultSnapshot(
        ("state", "value"),
        (("SP", 100.00), ("RJ", 80.00)),
    )

    assert results_match(actual, expected, order_sensitive=True)


def test_results_match_equivalent_date_and_datetime_values() -> None:
    expected = ResultSnapshot(
        ("month", "count"),
        ((date(2018, 1, 1), 10),),
    )
    actual = ResultSnapshot(
        ("order_month", "order_count"),
        ((datetime(2018, 1, 1), 10),),
    )

    assert results_match(actual, expected)


def test_month_normalization_does_not_affect_ordinary_strings() -> None:
    actual = ResultSnapshot(("delivery_status",), (("after_estimate",),))
    expected = ResultSnapshot(("delivery_status",), (("delayed",),))

    assert not results_match(actual, expected)


def test_tie_aware_ranking_allows_only_tied_group_reordering() -> None:
    ranking = RankingComparison("seller_count", "descending")
    expected = ResultSnapshot(
        ("seller_state", "seller_count"),
        (("SP", 100), ("RJ", 50), ("MS", 5), ("RN", 5)),
    )
    tied_rows_reordered = ResultSnapshot(
        ("state", "count"),
        (("SP", 100), ("RJ", 50), ("RN", 5), ("MS", 5)),
    )
    primary_ranking_broken = ResultSnapshot(
        ("state", "count"),
        (("SP", 100), ("RN", 5), ("RJ", 50), ("MS", 5)),
    )

    assert results_match(tied_rows_reordered, expected, ranking=ranking)
    assert not results_match(primary_ranking_broken, expected, ranking=ranking)


def test_tie_aware_ranking_rejects_missing_or_extra_rows() -> None:
    ranking = RankingComparison("seller_count", "descending")
    expected = ResultSnapshot(
        ("seller_state", "seller_count"),
        (("SP", 100), ("RJ", 50), ("MS", 5), ("RN", 5)),
    )
    missing = ResultSnapshot(
        ("state", "count"),
        (("SP", 100), ("RJ", 50), ("MS", 5)),
    )
    extra = ResultSnapshot(
        ("state", "count"),
        (("SP", 100), ("RJ", 50), ("MS", 5), ("RN", 5), ("AC", 1)),
    )

    assert not results_match(missing, expected, ranking=ranking)
    assert not results_match(extra, expected, ranking=ranking)


def test_declared_label_aliases_require_matching_numeric_values() -> None:
    aliases = (
        LabelAlias("delivery_status", "delayed", ("after_estimate",)),
        LabelAlias(
            "delivery_status",
            "on_time_or_early",
            ("on_or_before_estimate",),
        ),
    )
    expected = ResultSnapshot(
        ("delivery_status", "average_review_score"),
        (("delayed", 2.5665), ("on_time_or_early", 4.2936)),
    )
    equivalent = ResultSnapshot(
        ("timing", "score"),
        (("after_estimate", 2.56655), ("on_or_before_estimate", 4.29358)),
    )
    unknown_label = ResultSnapshot(
        ("timing", "score"),
        (("late_delivery", 2.56655), ("on_or_before_estimate", 4.29358)),
    )
    wrong_value = ResultSnapshot(
        ("timing", "score"),
        (("after_estimate", 3.0), ("on_or_before_estimate", 4.29358)),
    )

    assert results_match(equivalent, expected, label_aliases=aliases)
    assert not results_match(unknown_label, expected, label_aliases=aliases)
    assert not results_match(wrong_value, expected, label_aliases=aliases)


def test_evaluate_result_accepts_correct_answer_and_correct_rejection() -> None:
    answerable_question = GOLD_QUESTIONS[0]
    expected = ResultSnapshot(("total_order_count",), ((99_441,),))
    correct_answer = _answer_result(_sql_result(("order_count",), ((99_441,),)))
    answer_record = evaluate_result(
        answerable_question,
        correct_answer,
        expected,
        "total_order_count",
    )

    unanswerable_question = GOLD_QUESTIONS[-1]
    rejection = _answer_result(
        None,
        status="generation_error",
        generated_sql=None,
        generation_error=SQLGenerationError(
            "cannot_generate",
            "The dataset has no refund events.",
        ),
    )
    rejection_record = evaluate_result(
        unanswerable_question,
        rejection,
        None,
        "rejection",
    )

    assert answer_record.passed is True
    assert rejection_record.passed is True
    assert rejection_record.failure_reason is None


def test_evaluate_result_classifies_semantic_failure_and_missing_rejection() -> None:
    wrong_answer = evaluate_result(
        GOLD_QUESTIONS[0],
        _answer_result(_sql_result(("order_count",), ((10,),))),
        ResultSnapshot(("total_order_count",), ((99_441,),)),
        "total_order_count",
    )
    should_reject = evaluate_result(
        GOLD_QUESTIONS[-1],
        _answer_result(_sql_result(("refund_rate",), ((0.0,),))),
        None,
        "rejection",
    )

    assert wrong_answer.failure_reason == "semantic_wrong_answer"
    assert should_reject.failure_reason == "should_have_rejected"


def test_evaluate_gold_set_uses_existing_answer_and_reference_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "olist.duckdb"
    database_path.touch()
    question = GOLD_QUESTIONS[0]
    model = object()
    answer_calls: list[tuple[Path, str, object]] = []

    def fake_reference_executor(path, sql):
        assert path == database_path
        assert "COUNT(*)" in sql
        return _sql_result(("total_order_count",), ((99_441,),))

    def fake_answer(path, text, received_model):
        answer_calls.append((path, text, received_model))
        return _answer_result(_sql_result(("order_count",), ((99_441,),)))

    monkeypatch.setattr(evaluation_module, "run_readonly_sql", fake_reference_executor)
    monkeypatch.setattr(evaluation_module, "answer_question", fake_answer)

    records = evaluate_gold_set(database_path, model, questions=(question,))

    assert records[0].passed is True
    assert answer_calls == [(database_path, question.question, model)]


# Gold questions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data" / "raw"


GRAIN_SENSITIVE_BASELINES = {
    "average_payment_value_per_order",
    "orders_with_multiple_items_percentage",
    "orders_with_multiple_payment_records_percentage",
}


@pytest.fixture(scope="module")
def database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gold_questions") / "olist.duckdb"
    build_database(DATA_DIR, path)
    return path


def test_answerable_questions_reference_exactly_one_sql_source() -> None:
    for question in gold_questions_GOLD_QUESTIONS:
        has_baseline = question.baseline_key is not None
        has_reference_sql = question.reference_sql is not None
        if question.answerable:
            assert has_baseline ^ has_reference_sql
            assert question.unanswerable_reason is None
        else:
            assert not has_baseline
            assert not has_reference_sql
            assert question.unanswerable_reason is not None
            assert question.unanswerable_reason.strip()


def test_all_answerable_queries_execute_successfully(database_path: Path) -> None:
    baselines = {query.key: query.sql for query in BASELINE_QUERIES}

    for question in gold_questions_GOLD_QUESTIONS:
        if not question.answerable:
            continue
        sql = (
            baselines[question.baseline_key]
            if question.baseline_key is not None
            else question.reference_sql
        )
        assert sql is not None
        result = run_readonly_sql(database_path, sql)
        assert result.status == "success", (question.id, result.error)
        assert result.returned_row_count > 0
        assert result.truncated is False


def test_contains_required_unanswerable_questions() -> None:
    unanswerable = {
        question.question: question
        for question in gold_questions_GOLD_QUESTIONS
        if not question.answerable
    }

    assert len(unanswerable) >= 2
    assert "What is Olist's gross profit margin?" in unanswerable
    assert "What is the refund rate?" in unanswerable


def test_contains_required_grain_sensitive_questions() -> None:
    grain_questions = {
        question.baseline_key: question
        for question in gold_questions_GOLD_QUESTIONS
        if question.category == "grain_sensitive"
    }

    assert set(grain_questions) == GRAIN_SENSITIVE_BASELINES
    assert "payment-record grain" in grain_questions[
        "average_payment_value_per_order"
    ].expected_grain
    assert "order-item grain" in grain_questions[
        "orders_with_multiple_items_percentage"
    ].expected_grain
    assert "payment-record grain" in grain_questions[
        "orders_with_multiple_payment_records_percentage"
    ].expected_grain
    assert all("order grain" in question.expected_grain for question in grain_questions.values())


# Held out evaluator

@pytest.fixture()
def held_out_evaluator_database_path(tmp_path: Path) -> Path:
    path = tmp_path / "held-out.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE scalar_values(value DOUBLE)")
        connection.execute("INSERT INTO scalar_values VALUES (42.0)")
        connection.execute("CREATE TABLE ranked_values(label VARCHAR, value DOUBLE)")
        connection.execute(
            "INSERT INTO ranked_values VALUES ('A', 3.0), ('B', 2.0), ('C', 1.0)"
        )
        connection.execute("CREATE TABLE monthly_values(period DATE, value DOUBLE)")
        connection.execute(
            "INSERT INTO monthly_values VALUES "
            "(DATE '2020-01-01', 10.0), "
            "(DATE '2020-02-01', 15.0), "
            "(DATE '2020-04-01', 12.0)"
        )
    return path


def _sql_question(
    *,
    reference_sql: str = "SELECT value FROM scalar_values",
    expected_grain: str = "single value",
) -> MultiToolTestQuestion:
    return MultiToolTestQuestion(
        id="MTQ-TEST-SQL",
        question="Return the test value.",
        category="sql_only",
        expected_disposition="answer",
        expected_route="sql_only",
        expected_python_operation=None,
        metric_definition="Test metric.",
        expected_grain=expected_grain,
        expected_tables=("scalar_values",),
        reference_sql=reference_sql,
    )


def _growth_question() -> MultiToolTestQuestion:
    return MultiToolTestQuestion(
        id="MTQ-TEST-GROWTH",
        question="How did the value change month over month?",
        category="calculate_growth",
        expected_disposition="answer",
        expected_route="sql_then_python",
        expected_python_operation="calculate_growth",
        metric_definition="Monthly test value followed by period-over-period growth.",
        expected_grain="one value per observed month",
        expected_tables=("monthly_values",),
        reference_sql=(
            "SELECT period AS month, value FROM monthly_values ORDER BY period"
        ),
        python_columns=("month", "value"),
    )


def _reject_question(category: str = "capability_unsupported") -> MultiToolTestQuestion:
    return MultiToolTestQuestion(
        id="MTQ-TEST-REJECT",
        question="Predict a future value.",
        category=category,
        expected_disposition="reject",
        expected_route=None,
        expected_python_operation=None,
        metric_definition="Unsupported test request.",
        expected_grain="one predicted value",
        expected_tables=(),
        unanswerable_reason="The current tools cannot make forecasts.",
    )


def _route(
    route: str | None,
    operation: str | None,
    *,
    status: str = "success",
    error: ToolRoutingError | None = None,
) -> ToolRouteDecision:
    return ToolRouteDecision(
        question="test question",
        route=route,
        python_operation=operation,
        reason="test decision" if status == "success" else None,
        status=status,
        error=error,
    )


def held_out_evaluator_sql_result(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
    *,
    status: str = "success",
    error: SQLExecutionError | None = None,
    truncated: bool = False,
) -> SQLResult:
    return SQLResult(
        executed_sql="SELECT test",
        columns=columns,
        rows=rows,
        returned_row_count=len(rows),
        truncated=truncated,
        status=status,
        error=error,
    )


def _successful_sql_actual(
    sql_result: SQLResult,
    *,
    route: str = "sql_only",
    operation: str | None = None,
    repaired: bool = False,
) -> MultiToolQuestionResult:
    sql_answer = QuestionAnswerResult(
        question="test question",
        generated_sql="SELECT generated",
        status="success",
        execution_result=sql_result,
        generation_error=None,
        execution_error=None,
        repaired_sql="SELECT repaired" if repaired else None,
        repair_attempted=repaired,
    )
    return MultiToolQuestionResult(
        question="test question",
        route_decision=_route(route, operation),
        status="success",
        sql_answer_result=sql_answer,
        analysis_plan=None,
        sql_result=sql_result,
        python_result=None,
        error=None,
    )


def _successful_growth_actual(
    growth: GrowthResult,
    *,
    operation: str = "calculate_growth",
) -> MultiToolQuestionResult:
    sql_result = held_out_evaluator_sql_result(
        ("period", "value"),
        (
            (date(2020, 1, 1), 10.0),
            (date(2020, 2, 1), 15.0),
            (date(2020, 4, 1), 12.0),
        ),
    )
    return MultiToolQuestionResult(
        question="test question",
        route_decision=_route("sql_then_python", operation),
        status="success",
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=sql_result,
        python_result=PythonAnalysisResult(
            operation=operation,
            status="success",
            result=growth,
            error=None,
        ),
        error=None,
    )


def _expected_growth() -> GrowthResult:
    return GrowthResult(
        points=(
            GrowthPoint("2020-01", 10.0, None, None, None),
            GrowthPoint("2020-02", 15.0, 10.0, 5.0, 0.5),
            GrowthPoint("2020-04", 12.0, None, None, None),
        ),
        period_count=3,
    )


def test_sql_only_uses_successful_repaired_execution_result(held_out_evaluator_database_path: Path) -> None:
    actual = _successful_sql_actual(
        held_out_evaluator_sql_result(("different_alias",), ((42.0,),)),
        repaired=True,
    )
    actual = MultiToolQuestionResult(
        question=actual.question,
        route_decision=actual.route_decision,
        status=actual.status,
        sql_answer_result=actual.sql_answer_result,
        analysis_plan=None,
        sql_result=held_out_evaluator_sql_result(
            (),
            (),
            status="error",
            error=SQLExecutionError("invalid_sql", "initial failure"),
        ),
        python_result=None,
        error=None,
    )

    evaluation = evaluate_held_out_case(_sql_question(), actual, held_out_evaluator_database_path)

    assert evaluation.passed is True
    assert evaluation.semantic_correct is True


def test_numeric_meaningful_mismatch_fails(held_out_evaluator_database_path: Path) -> None:
    evaluation = evaluate_held_out_case(
        _sql_question(),
        _successful_sql_actual(held_out_evaluator_sql_result(("value",), ((43.0,),))),
        held_out_evaluator_database_path,
    )

    assert evaluation.passed is False
    assert evaluation.failure_reason == "value mismatch at row 1 column 1"


def test_route_mismatch_fails_even_when_values_match(held_out_evaluator_database_path: Path) -> None:
    actual = _successful_sql_actual(
        held_out_evaluator_sql_result(("value",), ((42.0,),)),
        route="sql_then_python",
        operation="calculate_growth",
    )

    evaluation = evaluate_held_out_case(_sql_question(), actual, held_out_evaluator_database_path)

    assert evaluation.semantic_correct is True
    assert evaluation.route_correct is False
    assert evaluation.passed is False
    assert evaluation.failure_reason == (
        "route mismatch: expected sql_only, got sql_then_python"
    )


def test_growth_wrong_operation_fails(held_out_evaluator_database_path: Path) -> None:
    evaluation = evaluate_held_out_case(
        _growth_question(),
        _successful_growth_actual(_expected_growth(), operation="correlation"),
        held_out_evaluator_database_path,
    )

    assert evaluation.operation_correct is False
    assert evaluation.passed is False
    assert evaluation.failure_reason == (
        "operation mismatch: expected calculate_growth, got correlation"
    )


def test_growth_none_fields_and_month_representations_compare_correctly(
    held_out_evaluator_database_path: Path,
) -> None:
    actual_growth = GrowthResult(
        points=(
            GrowthPoint(date(2020, 1, 1), 10.0, None, None, None),
            GrowthPoint(datetime(2020, 2, 1), 15.0, 10.0, 5.0, 0.5),
            GrowthPoint("2020-04-01", 12.0, None, None, None),
        ),
        period_count=3,
    )

    evaluation = evaluate_held_out_case(
        _growth_question(),
        _successful_growth_actual(actual_growth),
        held_out_evaluator_database_path,
    )

    assert evaluation.semantic_correct is True


def test_controlled_sql_generation_rejection_passes(held_out_evaluator_database_path: Path) -> None:
    sql_answer = QuestionAnswerResult(
        question="test question",
        generated_sql=None,
        status="generation_error",
        execution_result=None,
        generation_error=SQLGenerationError("cannot_generate", "missing data"),
        execution_error=None,
    )
    actual = MultiToolQuestionResult(
        question="test question",
        route_decision=_route("sql_only", None),
        status="sql_generation_error",
        sql_answer_result=sql_answer,
        analysis_plan=None,
        sql_result=None,
        python_result=None,
        error=MultiToolQuestionError("sql_generation_error", "missing data"),
    )

    evaluation = evaluate_held_out_case(
        _reject_question("data_unanswerable"),
        actual,
        held_out_evaluator_database_path,
    )

    assert evaluation.actual_disposition == "reject"
    assert evaluation.passed is True


@pytest.mark.parametrize("error_code", ["invalid_model_output", "model_error"])
def test_model_error_is_not_a_controlled_rejection(
    held_out_evaluator_database_path: Path,
    error_code: str,
) -> None:
    actual = MultiToolQuestionResult(
        question="test question",
        route_decision=_route(
            None,
            None,
            status="error",
            error=ToolRoutingError(error_code, "provider unavailable"),
        ),
        status="routing_error",
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=None,
        python_result=None,
        error=MultiToolQuestionError("routing_error", "provider unavailable"),
    )

    evaluation = evaluate_held_out_case(_reject_question(), actual, held_out_evaluator_database_path)

    assert evaluation.actual_disposition == "unknown"
    assert evaluation.passed is False
    assert evaluation.failure_reason == (
        "model/infrastructure error is not a semantic rejection"
    )


def test_summary_counts_denominators_and_accuracies() -> None:
    evaluations = (
        HeldOutCaseEvaluation(
            "one", "sql_only", "answer", "answer", True,
            "sql_only", "sql_only", True, None, None, True, True, True, None,
        ),
        HeldOutCaseEvaluation(
            "two", "calculate_growth", "answer", "answer", True,
            "sql_then_python", "sql_then_python", True,
            "calculate_growth", "correlation", False, True, False,
            "operation mismatch",
        ),
        HeldOutCaseEvaluation(
            "three", "data_unanswerable", "reject", "reject", True,
            None, None, None, None, None, None, None, True, None,
        ),
        HeldOutCaseEvaluation(
            "four", "capability_unsupported", "reject", "unknown", False,
            None, None, None, None, None, None, None, False,
            "infrastructure error",
        ),
    )

    summary = summarize_held_out_evaluations(evaluations)

    assert (summary.total, summary.passed, summary.failed) == (4, 2, 2)
    assert (summary.disposition_correct, summary.disposition_accuracy) == (3, 0.75)
    assert (summary.route_evaluated, summary.route_correct, summary.route_accuracy) == (
        2,
        2,
        1.0,
    )
    assert (
        summary.operation_evaluated,
        summary.operation_correct,
        summary.operation_accuracy,
    ) == (2, 1, 0.5)
    assert (
        summary.semantic_evaluated,
        summary.semantic_correct,
        summary.semantic_accuracy,
    ) == (2, 2, 1.0)
    assert (summary.sql_only_passed, summary.sql_only_total) == (1, 1)
    assert (summary.calculate_growth_passed, summary.calculate_growth_total) == (0, 1)
    assert (summary.data_unanswerable_passed, summary.data_unanswerable_total) == (1, 1)
    assert (
        summary.capability_unsupported_passed,
        summary.capability_unsupported_total,
    ) == (0, 1)


# Held out runner

def _question(number: int, *, category: str = "sql_only") -> MultiToolTestQuestion:
    answerable = category in {"sql_only", "calculate_growth"}
    growth = category == "calculate_growth"
    return MultiToolTestQuestion(
        id=f"MTQ-TEST-{number:03d}",
        question=f"User-only question {number}?",
        category=category,
        expected_disposition="answer" if answerable else "reject",
        expected_route=("sql_then_python" if growth else "sql_only")
        if answerable
        else None,
        expected_python_operation="calculate_growth" if growth else None,
        metric_definition=f"SECRET METRIC {number}",
        expected_grain=f"SECRET GRAIN {number}",
        expected_tables=(f"secret_table_{number}",) if answerable else (),
        reference_sql=f"SELECT {number} AS secret_reference_{number}"
        if answerable
        else None,
        python_columns=("month", "value") if growth else (),
        unanswerable_reason=None if answerable else f"SECRET REASON {number}",
        notes=f"SECRET NOTES {number}",
    )


def _actual(question: str, *, route: str = "sql_only") -> MultiToolQuestionResult:
    operation = "calculate_growth" if route == "sql_then_python" else None
    return MultiToolQuestionResult(
        question=question,
        route_decision=ToolRouteDecision(
            question=question,
            route=route,
            python_operation=operation,
            reason="scripted",
            status="success",
            error=None,
        ),
        status="success",
        sql_answer_result=None,
        analysis_plan=None,
        sql_result=None,
        python_result=None,
        error=None,
    )


def _evaluation(
    question: MultiToolTestQuestion,
    actual: MultiToolQuestionResult,
    *,
    passed: bool = True,
) -> HeldOutCaseEvaluation:
    expected_answer = question.expected_disposition == "answer"
    return HeldOutCaseEvaluation(
        question_id=question.id,
        category=question.category,
        expected_disposition=question.expected_disposition,
        actual_disposition="answer" if expected_answer else "reject",
        disposition_correct=True,
        expected_route=question.expected_route,
        actual_route=actual.route_decision.route,
        route_correct=passed if expected_answer else None,
        expected_python_operation=question.expected_python_operation,
        actual_python_operation=actual.route_decision.python_operation,
        operation_correct=passed if expected_answer else None,
        semantic_correct=passed if expected_answer else None,
        passed=passed,
        failure_reason=None if passed else "scripted failure",
    )


def _install_scripted_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer_calls: list[str],
    evaluation_calls: list[str],
) -> None:
    def fake_answer(
        database_path: str | Path,
        question: str,
        model: object,
    ) -> MultiToolQuestionResult:
        del database_path, model
        answer_calls.append(question)
        return _actual(question)

    def fake_evaluate(
        question: MultiToolTestQuestion,
        actual: MultiToolQuestionResult,
        database_path: str | Path,
    ) -> HeldOutCaseEvaluation:
        del database_path
        evaluation_calls.append(question.id)
        return _evaluation(question, actual)

    monkeypatch.setattr(runner_module, "answer_question_with_tools", fake_answer)
    monkeypatch.setattr(runner_module, "evaluate_held_out_case", fake_evaluate)


def test_runner_passes_only_question_text_to_model_facing_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = (_question(1), _question(2, category="capability_unsupported"))
    observed: list[tuple[object, ...]] = []

    def fake_answer(*args: object, **kwargs: object) -> MultiToolQuestionResult:
        observed.append((*args, kwargs))
        return _actual(str(args[1]))

    monkeypatch.setattr(runner_module, "answer_question_with_tools", fake_answer)
    monkeypatch.setattr(
        runner_module,
        "evaluate_held_out_case",
        lambda question, actual, database_path: _evaluation(question, actual),
    )

    model = object()
    run_held_out_evaluation("database.duckdb", model, questions)

    assert len(observed) == 2
    for call, question in zip(observed, questions, strict=True):
        assert call[:3] == ("database.duckdb", question.question, model)
        assert call[3] == {}
        serialized_call = repr(call)
        if question.reference_sql is not None:
            assert question.reference_sql not in serialized_call
        assert question.metric_definition not in serialized_call
        assert question.expected_grain not in serialized_call
        assert question.notes not in serialized_call
        assert repr(question.expected_tables) not in serialized_call
        assert repr(question.expected_route) not in serialized_call


@pytest.mark.parametrize("exception_stage", ["agent", "evaluation"])
def test_unhandled_case_exception_is_recorded_without_retry_and_run_continues(
    monkeypatch: pytest.MonkeyPatch,
    exception_stage: str,
) -> None:
    questions = (_question(1), _question(2))
    answer_counts = {question.question: 0 for question in questions}
    evaluated: list[str] = []

    def fake_answer(
        database_path: str | Path,
        question: str,
        model: object,
    ) -> MultiToolQuestionResult:
        del database_path, model
        answer_counts[question] += 1
        if exception_stage == "agent" and question == questions[0].question:
            raise RuntimeError("scripted agent failure")
        return _actual(question)

    def fake_evaluate(
        question: MultiToolTestQuestion,
        actual: MultiToolQuestionResult,
        database_path: str | Path,
    ) -> HeldOutCaseEvaluation:
        del database_path
        evaluated.append(question.id)
        if exception_stage == "evaluation" and question.id == questions[0].id:
            raise RuntimeError("scripted evaluation failure")
        return _evaluation(question, actual)

    monkeypatch.setattr(runner_module, "answer_question_with_tools", fake_answer)
    monkeypatch.setattr(runner_module, "evaluate_held_out_case", fake_evaluate)

    result = run_held_out_evaluation("unused.duckdb", object(), questions)

    assert answer_counts == {question.question: 1 for question in questions}
    assert result.summary.total == 2
    assert result.summary.failed == 1
    assert result.run_failures[0].question_id == questions[0].id
    assert result.run_failures[0].stage == exception_stage
    assert result.evaluations[0].actual_disposition == "unknown"
    assert result.evaluations[1].passed is True


def test_report_contains_summary_and_case_results_without_frozen_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = (_question(1), _question(2))
    monkeypatch.setattr(
        runner_module,
        "answer_question_with_tools",
        lambda database_path, question, model: _actual(question),
    )
    monkeypatch.setattr(
        runner_module,
        "evaluate_held_out_case",
        lambda question, actual, database_path: _evaluation(
            question,
            actual,
            passed=question.id == questions[0].id,
        ),
    )

    report = format_held_out_report(
        run_held_out_evaluation("unused.duckdb", object(), questions)
    )

    assert "Total: 2" in report
    assert "Passed: 1" in report
    assert "Failed: 1" in report
    assert "Overall pass rate: 50.00%" in report
    assert "Disposition accuracy: 100.00%" in report
    assert "Route accuracy: 50.00%" in report
    assert "Operation accuracy: 50.00%" in report
    assert "Semantic accuracy: 50.00%" in report
    assert "SQL_ONLY: 1 / 2" in report
    assert "Result: PASS" in report
    assert "Result: FAIL" in report
    for question in questions:
        assert f"ID: {question.id}" in report
        assert question.reference_sql not in report
        assert question.metric_definition not in report
        assert question.notes not in report


def test_report_does_not_include_model_secret_or_environment_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _question(1)
    answer_calls: list[str] = []
    evaluation_calls: list[str] = []
    _install_scripted_dependencies(
        monkeypatch,
        answer_calls=answer_calls,
        evaluation_calls=evaluation_calls,
    )
    secret_model = {"DEEPSEEK_API_KEY": "super-secret-test-key"}

    report = format_held_out_report(
        run_held_out_evaluation("unused.duckdb", secret_model, (question,))
    )

    assert "DEEPSEEK_API_KEY" not in report
    assert "super-secret-test-key" not in report
    assert ".env" not in report


# Multi tool test questions

FORBIDDEN_PROMPT_EXAMPLES = {
    "How many orders are in the dataset?",
    "Give me descriptive statistics for payment values.",
    "What is the Pearson correlation between item price and freight value?",
    "What was total item transaction value by month?",
    "How did total item transaction value change month over month?",
}


def _normalized_question(text: str) -> str:
    return " ".join(text.lower().split())


def test_all_questions_have_complete_human_audited_semantics() -> None:
    for question in MULTI_TOOL_TEST_QUESTIONS:
        assert question.metric_definition.strip()
        assert question.expected_grain.strip()
        assert question.notes.startswith("ALIGNED:")


def test_all_reference_sql_is_safe_and_executes_without_truncation(
    database_path: Path,
) -> None:
    for question in MULTI_TOOL_TEST_QUESTIONS:
        if question.reference_sql is None:
            continue

        result = run_readonly_sql(
            database_path,
            question.reference_sql,
            max_rows=1_000,
        )

        assert result.status == "success", (question.id, result.error)
        assert result.returned_row_count > 0, question.id
        assert result.truncated is False, question.id


def test_top_n_reference_has_deterministic_order_and_exact_row_count(
    database_path: Path,
) -> None:
    question = next(
        question
        for question in MULTI_TOOL_TEST_QUESTIONS
        if question.id == "MTQ-006"
    )
    assert question.reference_sql is not None
    normalized_sql = " ".join(question.reference_sql.split())

    assert "item_count DESC" in normalized_sql
    assert "sellers.seller_state ASC" in normalized_sql
    assert "sellers.seller_city ASC" in normalized_sql
    assert "LIMIT 7" in normalized_sql

    result = run_readonly_sql(database_path, question.reference_sql)

    assert result.status == "success"
    assert result.returned_row_count == 7
    ranking = tuple((row[2], row[1], row[0]) for row in result.rows)
    assert ranking == tuple(
        sorted(ranking, key=lambda item: (-item[0], item[1], item[2]))
    )


def test_questions_do_not_duplicate_gold_or_frozen_prompt_examples() -> None:
    old_questions = {
        _normalized_question(question.question) for question in GOLD_QUESTIONS
    }
    forbidden_questions = {
        _normalized_question(question) for question in FORBIDDEN_PROMPT_EXAMPLES
    }
    new_questions = {
        _normalized_question(question.question)
        for question in MULTI_TOOL_TEST_QUESTIONS
    }

    assert new_questions.isdisjoint(old_questions)
    assert new_questions.isdisjoint(forbidden_questions)
