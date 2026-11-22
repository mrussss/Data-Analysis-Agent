"""Minimal interactive CLI for the multi-tool question workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from data_analysis_agent.deepseek_provider import DeepSeekTextToSQLModel
from data_analysis_agent.multi_tool_service import (
    MultiToolQuestionResult,
    answer_question_with_tools,
)
from data_analysis_agent.python_analysis import (
    ColumnDescription,
    CorrelationResult,
    GrowthResult,
)
from data_analysis_agent.sql_executor import SQLResult
from data_analysis_agent.sql_generator import TextToSQLModel


DEFAULT_DATABASE_PATH = Path("data/processed/olist.duckdb")
EXIT_COMMANDS = frozenset({"exit", "quit"})

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]
ModelFactory = Callable[[], TextToSQLModel]


def _format_rows(result: SQLResult) -> list[str]:
    if not result.rows:
        return ["(no rows)"]
    if len(result.rows) == 1 and len(result.rows[0]) == 1:
        return [str(result.rows[0][0])]

    lines = [" | ".join(result.columns)] if result.columns else []
    lines.extend(
        " | ".join("NULL" if value is None else str(value) for value in row)
        for row in result.rows
    )
    if result.truncated:
        lines.append("(results truncated)")
    return lines


def _print_error(
    result: MultiToolQuestionResult,
    output_fn: OutputFunction,
) -> None:
    output_fn(f"Error stage: {result.status}")
    message = result.error.message if result.error is not None else "Unknown error."
    output_fn(f"Error: {message}")
    if result.status == "truncated_analysis_input":
        output_fn("Python analysis was NOT executed.")


def _print_sql_only(
    result: MultiToolQuestionResult,
    output_fn: OutputFunction,
) -> None:
    output_fn("Route: SQL only")
    answer = result.sql_answer_result
    if answer is None:
        _print_error(result, output_fn)
        return

    if answer.generated_sql is not None:
        output_fn("Generated SQL:")
        output_fn(answer.generated_sql)

    if answer.repaired_sql is not None:
        output_fn("Repaired SQL:")
        output_fn(answer.repaired_sql)

    if result.status == "success" and answer.execution_result is not None:
        output_fn("Result:")
        for line in _format_rows(answer.execution_result):
            output_fn(line)
    else:
        _print_error(result, output_fn)

    output_fn(f"Repair attempted: {'Yes' if answer.repair_attempted else 'No'}")


def _print_descriptions(
    descriptions: tuple[ColumnDescription, ...],
    output_fn: OutputFunction,
) -> None:
    for index, description in enumerate(descriptions):
        if index:
            output_fn("")
        if len(descriptions) == 1:
            output_fn(description.column.replace("_", " ").capitalize())
        else:
            output_fn(f"Column: {description.column}")
        output_fn(f"Count: {description.count}")
        output_fn(f"Mean: {description.mean}")
        output_fn(f"Std: {description.std}")
        output_fn(f"Min: {description.min}")
        output_fn(f"Median: {description.median}")
        output_fn(f"Max: {description.max}")


def _print_growth_result(
    growth_result: GrowthResult,
    output_fn: OutputFunction,
) -> None:
    output_fn("Growth result:")
    output_fn("Period | Value | Previous | Absolute Change | Growth Rate")
    for point in growth_result.points:
        values = (
            point.period,
            point.value,
            point.previous_value,
            point.absolute_change,
            point.growth_rate,
        )
        output_fn(
            " | ".join(
                "NULL" if value is None else str(value) for value in values
            )
        )
    output_fn(f"Period count: {growth_result.period_count}")


def _print_python_analysis(
    result: MultiToolQuestionResult,
    output_fn: OutputFunction,
) -> None:
    output_fn("Route: SQL → Python")
    plan = result.analysis_plan
    if plan is not None:
        output_fn(f"Python analysis: {plan.python_operation}")
        if plan.sql is not None:
            output_fn("SQL:")
            output_fn(plan.sql)
        if plan.python_columns:
            output_fn(f"Python columns: {', '.join(plan.python_columns)}")

    if result.status != "success":
        _print_error(result, output_fn)
        return

    python_result = result.python_result
    if python_result is None or python_result.result is None:
        output_fn("Error stage: python_analysis_error")
        output_fn("Error: Python analysis returned no result.")
        return

    payload = python_result.result
    if (
        isinstance(payload, tuple)
        and payload
        and all(isinstance(item, ColumnDescription) for item in payload)
    ):
        _print_descriptions(payload, output_fn)
    elif isinstance(payload, CorrelationResult):
        output_fn(f"Correlation: {payload.correlation}")
        output_fn(f"Paired rows: {payload.paired_count}")
    elif isinstance(payload, GrowthResult):
        _print_growth_result(payload, output_fn)
    else:
        output_fn("Error stage: python_analysis_error")
        output_fn(
            "Error: Unsupported Python analysis result type: "
            f"{type(payload).__name__}"
        )


def _print_answer(
    result: MultiToolQuestionResult,
    output_fn: OutputFunction,
) -> None:
    if result.route_decision.route == "sql_only":
        _print_sql_only(result, output_fn)
    elif result.route_decision.route == "sql_then_python":
        _print_python_analysis(result, output_fn)
    else:
        _print_error(result, output_fn)


def run_cli(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    model_factory: ModelFactory = DeepSeekTextToSQLModel,
) -> int:
    """Run the interactive question loop using multi-tool orchestration."""

    path = Path(database_path)
    if not path.is_file():
        output_fn(f"Error: DuckDB database does not exist: {path}")
        return 1

    try:
        model = model_factory()
    except Exception as exc:
        output_fn(f"Error: Unable to initialize DeepSeek Provider: {exc}")
        return 1

    output_fn("Data Analysis Agent")
    output_fn("Type a question, or 'exit' to quit.")
    output_fn("")

    while True:
        try:
            question = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return 0

        try:
            result = answer_question_with_tools(path, question, model)
        except Exception as exc:
            output_fn(f"Error: {exc}")
            continue

        _print_answer(result, output_fn)
        output_fn("")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .model_budget import BudgetedModel
    from .routing import run_task

    parser = argparse.ArgumentParser(description="Trusted business queries and evidence-led investigation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("query", "investigate", "auto", "eval"):
        command = commands.add_parser(name)
        command.add_argument("--database", type=Path)
        command.add_argument("--online", action="store_true")
        if name == "eval":
            command.add_argument("--split", choices=("dev", "test"), default="test")
            command.add_argument("--case", help="Case ID or comma-separated Mode B case IDs")
            command.add_argument("--mode-b", action="store_true", help="Compare Mode B policies")
            command.add_argument("--budget", type=int, choices=(3, 5, 7), default=5)
            command.add_argument("--bike", action="store_true", help="Run UCI cross-domain smoke checks")
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("question")
            command.add_argument("--domain", choices=("olist", "bike"), default="olist")
            if name in {"auto", "investigate"}:
                command.add_argument("--baseline-db", type=Path)
                command.add_argument("--budget", type=int, choices=range(1, 8), default=7)
    commands.add_parser("legacy")
    args = parser.parse_args(argv)
    if args.command == "legacy":
        return run_cli(model_factory=BudgetedModel)
    domain = getattr(args, "domain", "olist")
    database = args.database or Path(f"data/processed/{domain}.duckdb")
    try:
        if not database.is_file():
            raise ValueError(f"Missing database {database}. Run python -m scripts.prepare_data {domain}")
        model = BudgetedModel() if args.online else None
        if args.command == "eval":
            from .benchmark import evaluate, evaluate_mode_b

            if args.mode_b or model:
                payload = evaluate_mode_b(database, split=args.split,
                                          case_ids=tuple(args.case.split(",")) if args.case else (),
                                          budget=args.budget, online=model, bike=args.bike)
            else:
                payload = evaluate(database, split=args.split, case_id=args.case, bike=args.bike)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                pending = args.output.with_suffix(".tmp")
                pending.write_text(json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":")) + "\n")
                pending.replace(args.output)
                summary = {"output": str(args.output), "summary": payload["summary"]}
                if "skipped_or_partial" in payload:
                    summary["skipped_or_partial"] = [
                        {"case": item["case"], "reason": item["reason"]}
                        for item in payload["skipped_or_partial"]
                    ]
                    summary["online_usage"] = payload["online_usage"]
                payload = summary
        else:
            baseline = getattr(args, "baseline_db", None)
            if baseline and not baseline.is_file():
                raise ValueError(f"Missing baseline database {baseline}")
            payload = run_task(database, args.question, domain, command=args.command,
                               baseline=baseline, budget=getattr(args, "budget", 7), model=model)
        print(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
        return int(payload.get("status") == "refused" or payload.get("stop_reason") == "planner_error")
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
