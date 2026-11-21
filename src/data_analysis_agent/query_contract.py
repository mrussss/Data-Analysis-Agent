"""Freeze supported business intent before compiling any SQL."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .execution_trace import TraceStep
from .metric_catalog import METRIC_DEFINITIONS_V2
from .observability import observed_stage
from .result_validation import validate_sql_result
from .sql_executor import SQLResult, run_readonly_sql

OLIST = """
SELECT o.order_id, o.order_purchase_timestamp AS ts, o.order_status AS status,
       c.customer_state AS state,
       isodow(o.order_purchase_timestamp)::VARCHAR AS weekday,
       {measure_columns}
       CASE WHEN o.order_delivered_customer_date IS NOT NULL
                 AND o.order_estimated_delivery_date IS NOT NULL
            THEN (o.order_delivered_customer_date > o.order_estimated_delivery_date)::INT
       END AS late,
       date_diff('second', o.order_purchase_timestamp, o.order_delivered_customer_date)
         / 86400.0 AS delivery_days
FROM orders o LEFT JOIN customers c USING (customer_id)
{measure_join}
"""
BIKE = """
SELECT dteday::TIMESTAMP + hr * INTERVAL '1 hour' AS ts,
       cnt AS rentals, workingday::VARCHAR AS workingday,
       weathersit::VARCHAR AS weather, season::VARCHAR AS season
FROM bike_hours
"""
ORDER_MEASURES = {
    "gmv": ("COALESCE(i.item_transaction_value, 0) AS gmv,",
            "LEFT JOIN order_item_summary i USING (order_id)"),
    "gmv_with_freight": ("COALESCE(i.item_transaction_value, 0) AS gmv, "
                         "COALESCE(i.freight_value, 0) AS freight,",
                         "LEFT JOIN order_item_summary i USING (order_id)"),
    "payments": ("COALESCE(p.payment_value, 0) AS payments,",
                 "LEFT JOIN order_payment_summary p USING (order_id)"),
    "average_items_per_order": ("COALESCE(i.item_count, 0) AS items,",
                                "LEFT JOIN order_item_summary i USING (order_id)"),
}
UPSTREAM = {metric.identifier: metric.definition for metric in METRIC_DEFINITIONS_V2}


@dataclass(frozen=True)
class Metric:
    domain: str
    expression: str
    unit: str
    definition: str
    mean: bool = False


METRICS = {
    "orders": Metric("olist", "1", "orders", "Count orders at order grain, all statuses by default."),
    "gmv": Metric("olist", "gmv", "BRL", "Sum item prices excluding freight, by purchase date."),
    "gmv_with_freight": Metric("olist", "gmv + freight", "BRL", "Sum item prices plus freight."),
    "payments": Metric("olist", "payments", "BRL", "Sum payment records at order grain."),
    "average_items_per_order": Metric("olist", "items", "items",
                                      UPSTREAM["average_items_per_order"], True),
    "average_delivery_duration_days": Metric(
        "olist", "delivery_days", "days", UPSTREAM["average_delivery_duration_days"], True
    ),
    "late_delivery_rate": Metric(
        "olist", "late", "fraction", "Orders delivered after estimate / orders with both dates.", True
    ),
    "rentals": Metric("bike", "rentals", "rentals", "Sum of all reported hourly rentals."),
    "average_hourly_rentals": Metric(
        "bike", "rentals", "rentals/hour", "Average rentals per recorded hour.", True
    ),
}
DIMENSIONS = {"olist": ("state", "weekday", "status"), "bike": ("workingday", "weather", "season")}
STATUSES = frozenset(("delivered", "shipped", "canceled", "unavailable", "invoiced",
                      "processing", "created", "approved"))
ALIASES = {
    "order count": "orders", "订单数": "orders", "商品成交额": "gmv",
    "含运费成交额": "gmv_with_freight", "支付总额": "payments", "租车量": "rentals",
    "晚到率": "late_delivery_rate",
    **{label: metric.identifier for metric in METRIC_DEFINITIONS_V2
       if metric.identifier != "average_review_score"
       for label in (metric.label.lower(), metric.zh_label, *metric.zh_aliases)},
}
WORD = r"[a-z0-9_]+"
DAY = r"\d{4}-\d{2}-\d{2}"
QUESTION = re.compile(
    rf"(?:show |what is |查询 )?(?P<metric>[\w ]+?) from (?P<start>{DAY}) to (?P<end>{DAY})"
    rf"(?: where (?P<filters>{WORD}={WORD}(?: and {WORD}={WORD})*))?"
    r"(?: by (?P<grain>day|month))?[?.]?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryContract:
    metric: str
    start: date
    end: date
    filters: tuple[tuple[str, str], ...] = ()
    grain: str = "total"

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", tuple((key, value) for key, value in self.filters))
        spec = METRICS.get(self.metric)
        if spec is None or self.start >= self.end or self.grain not in ("total", "day", "month"):
            raise ValueError("Unsupported metric, empty time window, or invalid grain")
        if len({name for name, _ in self.filters}) != len(self.filters):
            raise ValueError("Conflicting filters")
        for name, value in self.filters:
            if name not in DIMENSIONS[spec.domain] or not re.fullmatch(WORD, value, re.I):
                raise ValueError(f"Unsupported filter: {name}={value}")
            if name == "status" and value not in STATUSES:
                raise ValueError(f"Unknown order status: {value}")

    @property
    def spec(self) -> Metric:
        return METRICS[self.metric]


def parse_question(question: str, domain: str) -> QueryContract:
    match = QUESTION.fullmatch(question.strip())
    if match is None:
        raise ValueError("Use '<metric> from YYYY-MM-DD to YYYY-MM-DD [where field=value] [by day|month]'")
    fields = match.groupdict()
    metric = fields["metric"].strip().lower()
    filters = []
    for condition in (re.split(" and ", fields["filters"], flags=re.I) if fields["filters"] else ()):
        field, value = condition.split("=")
        field = field.lower()
        filters.append((field, value.upper() if field == "state" else value.lower()))
    contract = QueryContract(
        ALIASES.get(metric, metric),
        date.fromisoformat(fields["start"]), date.fromisoformat(fields["end"]),
        tuple(filters),
        fields["grain"].lower() if fields["grain"] else "total",
    )
    if contract.spec.domain != domain:
        raise ValueError("Metric unavailable in this domain")
    return contract


def aggregate_sql(contract: QueryContract, dimensions: tuple[str, ...] = (), *,
                  sufficient_statistics: bool = False, cte: bool = False) -> str:
    """Compile trusted fragments at the domain's fact grain. Dates are half-open."""
    spec = contract.spec
    if set(dimensions) - set(DIMENSIONS[spec.domain]):
        raise ValueError("Unknown dimension")
    measure_columns, measure_join = ORDER_MEASURES.get(contract.metric, ("", ""))
    source = (OLIST.format(measure_columns=measure_columns, measure_join=measure_join)
              if spec.domain == "olist" else BIKE)
    keys = [f"COALESCE({dim}, '(unknown)') AS {dim}" for dim in dimensions]
    if contract.grain != "total":
        keys.insert(0, f"date_trunc('{contract.grain}', ts)::DATE AS period")
    expression = spec.expression
    measures = ([f"SUM({expression}) AS numerator", f"COUNT({expression}) AS n"]
                if sufficient_statistics else
                [f"{'AVG' if spec.mean else 'SUM'}({expression}) AS value"])
    predicates = [f"ts >= DATE '{contract.start}'", f"ts < DATE '{contract.end}'"]
    predicates.extend(f"{key} = '{value}'" for key, value in contract.filters)
    positions = ", ".join(map(str, range(1, len(keys) + 1)))
    grouping = f" GROUP BY {positions} ORDER BY {positions}" if keys else ""
    prefix = f"WITH facts AS ({source}) " if cte else ""
    relation = "facts" if cte else f"({source}) AS facts"
    return (f"{prefix}SELECT {', '.join(keys + measures)} FROM {relation} "
            f"WHERE {' AND '.join(predicates)}{grouping}")


@dataclass(frozen=True)
class ContractAnswer:
    status: str
    contract: QueryContract | None
    result: SQLResult | None
    trace: tuple[TraceStep, ...]
    error: str | None = None


@observed_stage("query_contract")
def answer_contract(database, question: str, domain: str, candidate: str | None = None,
                    repair=None) -> ContractAnswer:
    """Optional untrusted proposal gets one repair against the original immutable contract."""
    try:
        contract = parse_question(question, domain)
    except ValueError as exc:
        return ContractAnswer("refused", None, None, (), str(exc))
    canonical = aggregate_sql(contract)
    allowed = {canonical, aggregate_sql(contract, cte=True)}
    sql = candidate if candidate is not None else canonical
    steps: list[TraceStep] = []
    result = None
    for attempt in range(2):
        issue = "SQL is outside the two approved compiler forms"
        if sql in allowed:
            result = run_readonly_sql(database, sql)
            issues = validate_sql_result(result)
            issue = next((item.message for item in issues if item.severity == "error"), "")
            if not issue and not any(row[-1] is not None for row in result.rows):
                issue = "No usable observations in the requested population"
        steps.append(TraceStep("validation" if attempt == 0 else "sql_repair",
                               "error" if issue else "success", issue or "Contract satisfied",
                               (("sql", sql),)))
        if not issue:
            return ContractAnswer("success", contract, result, tuple(steps))
        if attempt == 0 and candidate is not None:
            sql = repair(contract, issue) if repair else canonical
        else:
            break
    return ContractAnswer("refused", contract, None, tuple(steps), issue)
