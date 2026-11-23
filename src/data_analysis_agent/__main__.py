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


def _business_investigation(database, args, model):
    from dataclasses import asdict
    from datetime import date
    from functools import partial
    import json

    from .business_tools import BusinessGoal
    from .investigation import investigate

    if not args.before or not args.current or args.baseline_db:
        raise ValueError("Use --before START END and --current START END in one business snapshot")
    before, current = (tuple(date.fromisoformat(day) for day in window)
                       for window in (args.before, args.current))
    if not before[0] < before[1] <= current[0] < current[1]:
        raise ValueError("Business windows must be chronologically ordered and disjoint")
    if args.domain == "olist":
        metric = args.metric or "item_lines"
        if (metric not in {"orders", "item_lines", "gmv", "freight_value", "late_delivery_rate"}
                or args.events or (metric == "orders" and (args.entity or args.dimension in {"category", "seller_state"}))
                or (metric == "late_delivery_rate" and (args.entity or args.dimension not in {"state", "weekday", "status"}))
                or (args.entity and args.dimension == "category")):
            raise ValueError("Metric, dimension, or event source is incompatible with the Olist investigation")
    elif args.domain == "retail":
        metric = args.metric or "units"
        if metric != "units" or not args.entity or args.dimension:
            raise ValueError("Retail investigations need an entity and use units as the comparison metric")
    else:
        raise ValueError("Business investigations support Olist and the fictional retail panel")
    events = ()
    if args.events:
        events = json.loads(args.events.read_text())
        fields = {"id", "subject", "date", "scope", "text", "source"}
        if (not isinstance(events, list) or any(not isinstance(item, dict) or not fields <= item.keys()
                or any(not isinstance(item[key], str) or not item[key] for key in fields) for item in events)):
            raise ValueError("Event archive must be a list of records with id, subject, date, scope, text, and source")
        if len({item["id"] for item in events}) != len(events):
            raise ValueError("Event IDs must be unique")
        for item in events:
            date.fromisoformat(item["date"])
    goal = BusinessGoal(args.question, args.domain, before, current, args.entity, metric, args.dimension)
    state = investigate((database, database), goal, policy="llm_adaptive" if model else "rule_adaptive",
                        budget=args.budget, planner=partial(model, planning=True) if model else None,
                        events=events)
    return asdict(state)


def _write_evaluation(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


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
            suite = command.add_mutually_exclusive_group()
            suite.add_argument("--mode-b", action="store_true", help="Compare delivery benchmark policies")
            suite.add_argument("--business", action="store_true", help="Run the business investigation suite")
            command.add_argument("--deep", action="store_true", help="Use the Mode B deep retail suite")
            command.add_argument("--audit", action="store_true", help="Audit and freeze the deep suite offline")
            command.add_argument("--prior-results", type=Path, action="append", default=[],
                                 help="Subtract prior business online usage from this experiment's budget")
            command.add_argument("--reliability-retest", action="store_true",
                                 help="Re-run specified, previously attempted business test cases separately")
            command.add_argument("--budget", type=int, choices=(3, 5, 7), default=5)
            command.add_argument("--bike", action="store_true", help="Run UCI cross-domain smoke checks")
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("question")
            command.add_argument("--domain", choices=("olist", "bike", "retail") if name == "investigate"
                                 else ("olist", "bike"), default="olist")
            if name == "investigate":
                command.add_argument("--business", action="store_true", help="Use cross-metric business tools")
                command.add_argument("--before", nargs=2, metavar=("START", "END"))
                command.add_argument("--current", nargs=2, metavar=("START", "END"))
                command.add_argument("--entity", help="Olist category or fictional retail merchant")
                command.add_argument("--metric")
                command.add_argument("--dimension", choices=("state", "weekday", "status", "category", "seller_state"))
                command.add_argument("--events", type=Path, help="Local retail event JSON archive")
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
        if args.command == "eval" and (args.deep or args.audit) and not args.business:
            raise ValueError("--deep and --audit require --business")
        business_online = args.command == "eval" and args.business and args.online
        if business_online:
            from .business_benchmark import BUSINESS_MAX_REQUESTS, BUSINESS_MAX_TOKENS

            model = BudgetedModel(max_requests=BUSINESS_MAX_REQUESTS, max_tokens=BUSINESS_MAX_TOKENS)
        else:
            model = BudgetedModel() if args.online else None
        if args.command == "eval":
            from .benchmark import evaluate, evaluate_mode_b

            if args.business:
                from .business_benchmark import DEEP_STUDY, audit_deep, evaluate_business, online_budget
                from .business_cases import DEEP_DEV, DEEP_ONLINE_ORDER, DEV, ONLINE_PAIRS

                if args.audit:
                    if not args.deep or args.online or not args.output or args.output.exists():
                        raise ValueError("Deep audit requires --deep, offline execution, and a new --output file")
                    _write_evaluation(args.output, audit_deep())
                    print(json.dumps({"study_id": DEEP_STUDY, "output": str(args.output)}))
                    return 0
                dev_cases = DEEP_DEV if args.deep else DEV
                online_order = DEEP_ONLINE_ORDER if args.deep else ONLINE_PAIRS
                study_id = DEEP_STUDY if args.deep else None

                if args.bike or (args.prior_results and not model):
                    raise ValueError("Business evaluation does not use --bike and prior usage requires --online")
                if args.reliability_retest and (not model or args.split != "test" or not args.case):
                    raise ValueError("Reliability retests require --online --split test and explicit --case")
                selected = tuple(args.case.split(",")) if args.case else ()
                if not model:
                    payload = evaluate_business(database, split=args.split, case_ids=selected, budget=args.budget, deep=args.deep)
                else:
                    if not args.output:
                        raise ValueError("Online business evaluation requires --output to preserve the result")
                    with online_budget(model, args.prior_results) as (previous, seen, begin_test, journal):
                        if args.output.exists():
                            raise ValueError("Use a new --output path to preserve the existing online result")
                        if args.split == "test":
                            completed_dev = {(row["case"], row["arm"]) for record in previous
                                             if record.get("split") == "dev" and record.get("study_id") == study_id
                                             and (record.get("low_probe") or {}).get("outcome") == "accepted"
                                             and record.get("model") == model.name
                                             for row in record["runs"]
                                             if row["execution"] == "online_model" and row["completed"]}
                            expected_dev = {(case.id, arm) for case in dev_cases
                                            for arm in ("llm_frozen", "llm_adaptive")}
                            if not expected_dev <= completed_dev:
                                raise ValueError("Finish both Low model arms on all development cases for this suite first")
                            if args.reliability_retest:
                                if set(selected) - seen:
                                    raise ValueError("Reliability retests require previously attempted cases")
                            else:
                                selected = selected or tuple(case for case in online_order if case not in seen)
                                if not selected or set(selected) & seen:
                                    raise ValueError("Previously attempted cases require --reliability-retest")
                        payload = evaluate_business(database, split=args.split, case_ids=selected,
                                                    budget=args.budget, model=model, begin_test=begin_test, deep=args.deep)
                        payload.update(budget_journal_id=journal["id"], cumulative_online_usage=journal["usage"],
                                       reliability_retest=args.reliability_retest)
                        _write_evaluation(args.output, payload)
            elif args.mode_b or model:
                payload = evaluate_mode_b(database, split=args.split,
                                          case_ids=tuple(args.case.split(",")) if args.case else (),
                                          budget=args.budget, online=model, bike=args.bike)
            else:
                payload = evaluate(database, split=args.split, case_id=args.case, bike=args.bike)
            if args.output:
                if not business_online:
                    _write_evaluation(args.output, payload)
                summary = {"output": str(args.output), "summary": payload["summary"]}
                if "skipped_or_partial" in payload:
                    summary["skipped_or_partial"] = [
                        {"case": item["case"], "reason": item["reason"]}
                        for item in payload["skipped_or_partial"]
                    ]
                    summary["online_usage"] = payload["online_usage"]
                    if args.business:
                        summary["low_probe"] = payload["low_probe"]
                        if model:
                            summary["cumulative_online_usage"] = payload["cumulative_online_usage"]
                payload = summary
        elif args.command == "investigate" and args.business:
            payload = _business_investigation(database, args, model)
        else:
            if domain == "retail" or any(getattr(args, name, None) is not None
                                        for name in ("before", "current", "entity", "metric", "dimension", "events")):
                raise ValueError("Business arguments require investigate --business")
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
