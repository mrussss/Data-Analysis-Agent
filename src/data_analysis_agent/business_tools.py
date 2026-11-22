"""Bounded metric and event observations for Mode B business investigations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .investigation import Comparison, Evidence, measure
from .query_contract import QueryContract
from .sql_executor import run_readonly_sql


@dataclass(frozen=True)
class BusinessGoal:
    question: str
    domain: str
    before: tuple[date, date]
    current: tuple[date, date]
    entity: str | None
    metric: str
    dimension: str | None = None


@dataclass(frozen=True)
class BusinessEvidence(Evidence):
    columns: tuple[str, ...] = ()
    rows: tuple[tuple, ...] = ()
    facts: tuple[str, ...] = ()


OLIST_TOOLS = {
    "breakdown": "Largest absolute before/after change of the requested additive metric by requested dimension.",
    "reviews": "Negative review rate (score <= 2) and coverage, latest review at order grain.",
    "delivery": "Late delivery rate among category orders with both timestamps.",
    "review_delivery": "Negative review rates for late versus on-time reviewed orders with delivery dates.",
    "freight": "Freight charged in total and per item line.",
    "sellers": "Distinct sellers with recorded sales and lines per seller.",
    "products": "Distinct products with recorded sales and lines per product.",
}
RETAIL_TOOLS = {
    "events": "Search dated public archive by entity (exact subject plus names containing it) within the baseline/current windows.",
    "timeline": "Daily target units and first sustained break versus baseline.",
    "scope": "Largest absolute changes by SKU and region, for units and complaint counts.",
    "stock": "In-stock panel share, days with recorded stock, and unit change on rows that became unavailable.",
    "stock_core": "Inventory and unit change for the core SKU only.",
    "stock_plus": "Inventory and unit change for the plus SKU only.",
    "pricing": "Average SKU-region-day price and promotion share.",
    "access": "Visits and units per visit.",
    "peer": "Unit change for the recorded market peer.",
    "complaints": "Complaint count and complaints per sold unit.",
    "service": "Recorded late shipments per sold unit.",
}


def catalog(goal: BusinessGoal) -> dict[str, str]:
    tools = OLIST_TOOLS if goal.domain == "olist" else RETAIL_TOOLS
    return {key: value for key, value in tools.items() if key != "breakdown" or goal.dimension}


def change(before, after, *, absolute=None) -> str:
    if before is None or after is None:
        return "missing"
    before, after = float(before), float(after)
    threshold = absolute if absolute is not None else abs(before) * 0.08
    delta = after - before
    if abs(delta) <= threshold + 1e-10:
        return "stable"
    return "up" if delta > 0 else "down"


def _literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _period(source, column, goal):
    before, current = (_literal(value) for value in (goal.before[0], goal.current[0]))
    before_end, current_end = (_literal(value) for value in (goal.before[1], goal.current[1]))
    return (f"SELECT *, CASE WHEN {column} >= {current} AND {column} < {current_end} THEN 1 ELSE 0 END AS period "
            f"FROM ({source}) AS source WHERE ({column} >= {before} AND {column} < {before_end}) "
            f"OR ({column} >= {current} AND {column} < {current_end})")


def _olist_sql(goal, action):
    metrics = {"orders": "COUNT(*)", "item_lines": "COUNT(*)", "gmv": "SUM(price)",
               "freight_value": "SUM(freight_value)"}
    groups = {"weekday": "isodow(ts)::VARCHAR", "state": "customer_state", "category": "category",
              "seller_state": "seller_state", "status": "order_status"}
    at_order_grain = goal.metric == "orders" and not goal.entity and action in {"headline", "breakdown"}
    items = "" if at_order_grain else "JOIN items i USING(order_id)"
    category = "" if not goal.entity else f"WHERE p.product_category_name_english = {_literal(goal.entity)}"
    ctes = f"""WITH items AS (
      SELECT order_id, product_id, seller_id, price, freight_value, p.product_category_name_english AS category,
             seller_state FROM order_items JOIN products_with_category_translation p USING(product_id)
      JOIN sellers USING(seller_id) {category}
    ), cohort AS (
      {_period(f'''SELECT o.order_id, order_purchase_timestamp AS ts, order_status, customer_state,
             (order_delivered_customer_date > order_estimated_delivery_date)::INT AS late
             {", i.* EXCLUDE (order_id)" if items else ""}
             FROM orders o JOIN customers USING(customer_id) {items}''', 'ts', goal)}
    )"""
    if action == "breakdown":
        statement = (f"SELECT period, COALESCE({groups[goal.dimension]}, '(unknown)') AS key, {metrics[goal.metric]} AS value "
                     "FROM cohort GROUP BY 1, 2 ORDER BY 1, 2")
    elif action == "headline":
        statement = ("SELECT period, COUNT(*) AS orders FROM cohort GROUP BY 1 ORDER BY 1" if at_order_grain
                     else """SELECT period, COUNT(*) AS item_lines, SUM(price) AS gmv, AVG(price) AS avg_price,
                        COUNT(DISTINCT order_id) AS category_orders,
                        COUNT(*)::DOUBLE / COUNT(DISTINCT order_id) AS lines_per_order,
                        SUM(freight_value) AS freight_value FROM cohort GROUP BY 1 ORDER BY 1""")
    elif action == "delivery":
        statement = """SELECT period, COUNT(late) AS eligible_orders, COUNT(*) AS category_orders, AVG(late) AS late_rate
            FROM (SELECT DISTINCT period, order_id, late FROM cohort) GROUP BY 1 ORDER BY 1"""
    elif action in {"reviews", "review_delivery"}:
        ctes += """, latest AS (
          SELECT order_id, review_score FROM order_reviews
          QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC NULLS LAST,
            review_creation_date DESC NULLS LAST, review_id, review_score) = 1
        ), reviewed AS (
          SELECT * FROM (SELECT DISTINCT period, order_id, late FROM cohort) LEFT JOIN latest USING(order_id)
        )"""
        if action == "reviews":
            statement = """SELECT period, COUNT(review_score) AS reviewed_orders, COUNT(*) AS category_orders,
                COUNT(review_score)::DOUBLE / COUNT(*) AS coverage, AVG((review_score <= 2)::INT) AS negative_rate
                FROM reviewed GROUP BY 1 ORDER BY 1"""
        else:
            statement = """SELECT period, late, COUNT(review_score) AS reviewed_orders,
                AVG((review_score <= 2)::INT) AS negative_rate FROM reviewed
                WHERE late IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""
    else:
        expressions = {
            "freight": "SUM(freight_value) AS freight_total, AVG(freight_value) AS freight_per_line",
            "sellers": "COUNT(DISTINCT seller_id) AS active_sellers, COUNT(*)::DOUBLE / COUNT(DISTINCT seller_id) AS lines_per_seller",
            "products": "COUNT(DISTINCT product_id) AS active_products, COUNT(*)::DOUBLE / COUNT(DISTINCT product_id) AS lines_per_product",
        }
        statement = f"SELECT period, {expressions[action]} FROM cohort GROUP BY 1 ORDER BY 1"
    return f"{ctes} {statement}"


def _retail_sql(goal, action):
    entity = "market peer" if action == "peer" else goal.entity
    predicate = f"entity = {_literal(entity)}"
    if action in {"stock_core", "stock_plus"}:
        predicate += f" AND sku = {_literal(action.split('_')[1])}"
        action = "stock"
    source = _period(f"SELECT * FROM retail WHERE {predicate}", "day", goal)
    if action == "timeline":
        statement = "SELECT period, day, SUM(units) AS units FROM panel GROUP BY 1, 2 ORDER BY 1, 2"
    elif action == "scope":
        statement = """SELECT period, 'sku' AS axis, sku AS key, SUM(units) AS units, SUM(complaints) AS complaints
            FROM panel GROUP BY 1, 2, 3 UNION ALL
            SELECT period, 'region', region, SUM(units), SUM(complaints) FROM panel GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""
    elif action == "stock":
        statement = """SELECT period, AVG(available) AS availability, COUNT(available) AS recorded,
            COUNT(*) AS panel_rows, SUM(units) AS units, SUM(CASE WHEN switched THEN units END) AS switched_units
            FROM (SELECT *, MIN(available) FILTER (WHERE period = 0) OVER (PARTITION BY sku,region) = 1
              AND MAX(available) FILTER (WHERE period = 1) OVER (PARTITION BY sku,region) = 0 AS switched
              FROM panel) GROUP BY 1 ORDER BY 1"""
    else:
        measures = {
            "headline": "SUM(units) AS units",
            "peer": "SUM(units) AS units",
            "pricing": "SUM(price * units)::DOUBLE / NULLIF(SUM(units), 0) AS price, "
                       "SUM(promo * units)::DOUBLE / NULLIF(SUM(units), 0) AS promo_rate",
            "access": "SUM(visits) AS visits, SUM(units)::DOUBLE / NULLIF(SUM(visits), 0) AS conversion",
            "complaints": "SUM(complaints) AS complaints, SUM(complaints)::DOUBLE / NULLIF(SUM(units), 0) AS complaint_rate",
            "service": "SUM(late) AS late_orders, SUM(late)::DOUBLE / NULLIF(SUM(units), 0) AS late_rate",
        }
        statement = f"SELECT period, {measures[action]} FROM panel GROUP BY 1 ORDER BY 1"
    return f"WITH panel AS ({source}) {statement}"


def _top(deltas):
    return sorted(deltas.items(), key=lambda pair: (-abs(pair[1]), str(pair[0])))[0]


def _facts(action, columns, rows):
    if action == "breakdown":
        if not rows:
            return ()
        key, delta = _top({k: sum((value or 0) * (1 if p else -1) for p, group, value in rows if group == k)
                           for _, k, _ in rows})
        return (f"top={key}:{change(0, delta)}",)
    if action == "scope":
        findings = []
        for axis, label, index in (("sku", "sku", 3), ("region", "region", 3), ("sku", "complaints_sku", 4)):
            values = {key: sum(row[index] * (1 if row[0] else -1) for row in rows if row[1:3] == (axis, key))
                      for _, dim, key, *_ in rows if dim == axis}
            if values:
                key, delta = _top(values)
                if delta and sum(abs(value) == abs(delta) for value in values.values()) == 1:
                    findings.append(f"{label}={key}")
        return tuple(findings)
    if action == "timeline":
        baseline = [value for period, _, value in rows if period == 0]
        current = [(when, value) for period, when, value in rows if period == 1]
        if not baseline:
            return ()
        mean = sum(baseline) / len(baseline)
        for first, second in zip(current, current[1:]):
            direction = change(mean, first[1])
            if second[0] - first[0] == timedelta(days=1) and direction != "stable" and direction == change(mean, second[1]):
                return (f"onset={first[0]}",)
        return ("onset=none",)
    if action == "review_delivery":
        values = {(period, late): rate for period, late, n, rate in rows if n >= 10}
        if (1, 0) not in values or (1, 1) not in values:
            return ("late_gap=missing",)
        direction = change(values[1, 0], values[1, 1], absolute=0.025)
        return (f"late_gap={dict(up='positive', down='negative', stable='stable')[direction]}",)
    periods = {row[0]: dict(zip(columns[1:], row[1:])) for row in rows}
    if not {0, 1} <= periods.keys():
        return ()
    earlier, later = periods[0], periods[1]
    rates = {"negative_rate", "late_rate", "complaint_rate", "availability", "promo_rate", "coverage", "conversion"}
    facts = [f"{name}={change(value, later[name], absolute=0.025 if name in rates else None)}"
             for name, value in earlier.items()]
    if action in {"stock", "stock_core", "stock_plus"}:
        loss = earlier["units"] - later["units"]
        if loss > 0 and (earlier["switched_units"] or 0) >= loss * 0.8 and later["switched_units"] == 0:
            facts.append("stockout_sales_loss=present")
    return tuple(facts)


def observe(database: Path, goal: BusinessGoal, action: str, evidence_id: str, events=()) -> BusinessEvidence:
    if action not in {"headline", *catalog(goal)}:
        raise ValueError("Unknown action")
    if goal.metric == "late_delivery_rate" and action in {"headline", "breakdown"}:
        comparison = Comparison(QueryContract(goal.metric, *goal.before), QueryContract(goal.metric, *goal.current),
                                (goal.dimension,))
        group = (goal.dimension,) if action == "breakdown" else ()
        evidence = measure((database, database), comparison, group, evidence_id)
        findings = ()
        if evidence.segments and group:
            rows = evidence.segments
            contributions = {row.key[0]: (row.within or 0) + (row.mix or 0) for row in rows}
            key, delta = _top(contributions)
            within, mix = (sum(getattr(row, effect) or 0 for row in rows) for effect in ("within", "mix"))
            dominant = "within" if abs(within) >= abs(mix) else "mix"
            findings = (f"{group[0]}:top={key}:{change(0, delta)}", f"{group[0]}:effect={dominant}")
        return BusinessEvidence(evidence.id, (action,), evidence.sql, evidence.segments,
                                error=evidence.error, facts=findings)
    if action == "events":
        columns = ("id", "subject", "date", "scope", "text", "source")
        rows = tuple(tuple(event[key] for key in columns) for event in events
                     if goal.entity.casefold() in event["subject"].casefold()
                     and goal.before[0] <= date.fromisoformat(event["date"]) < goal.current[1])
        return BusinessEvidence(evidence_id, (action,), ("", ""), (), columns=columns, rows=rows)
    statement = (_olist_sql if goal.domain == "olist" else _retail_sql)(goal, action)
    result = run_readonly_sql(database, statement, max_rows=300)
    error = result.error.code if result.error else "truncated" if result.truncated else None
    facts = () if error else _facts(action, result.columns, result.rows)
    prefix = goal.dimension if action == "breakdown" else action
    return BusinessEvidence(evidence_id, (action,), (statement, ""), (), error, columns=result.columns,
                            rows=result.rows, facts=tuple(f"{prefix}:{item}" for item in facts))
