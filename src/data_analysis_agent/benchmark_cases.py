"""Frozen task definitions and independent Olist interventions and gold queries."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import shutil

import duckdb


WINDOW = ("2018-03-01", "2018-07-01")
NOW = " from 2018-03-01 to 2018-07-01"
WHERE = "o.order_purchase_timestamp >= DATE '2018-03-01' AND o.order_purchase_timestamp < DATE '2018-07-01'"
ITEMS = f"FROM order_items i JOIN orders o USING(order_id) WHERE {WHERE}"
ORDERS = f"FROM orders o WHERE {WHERE}"
ELIGIBLE = ("order_delivered_customer_date IS NOT NULL "
            "AND order_estimated_delivery_date IS NOT NULL")
LATE = "(order_delivered_customer_date > order_estimated_delivery_date)::INT"

@dataclass(frozen=True)
class GoldQuery:
    id: str
    question: str
    gold: str | None
    proposal: str
    retry: str | None = None


GOLD_TEST = (
    GoldQuery("orders", "orders" + NOW, f"SELECT COUNT(*) {ORDERS}", f"SELECT COUNT(*) {ORDERS}"),
    GoldQuery("monthly", "orders" + NOW + " by month",
              f"SELECT order_purchase_timestamp::DATE - "
              f"(DAY(order_purchase_timestamp)::INT - 1) AS month, COUNT(*) {ORDERS} "
              "GROUP BY 1 ORDER BY 1", f"SELECT COUNT(*) {ORDERS}"),
    GoldQuery("fanout", "gmv" + NOW, f"SELECT SUM(i.price) {ITEMS}",
              f"SELECT SUM(i.price) FROM orders o JOIN order_items i USING(order_id) "
              f"JOIN order_payments p USING(order_id) WHERE {WHERE}"),
    GoldQuery("all_orders_denominator", "average_items_per_order" + NOW,
              f"SELECT (SELECT COUNT(*) {ITEMS}) / (SELECT COUNT(*) {ORDERS})",
              f"SELECT (SELECT COUNT(*) {ITEMS}) / (SELECT COUNT(DISTINCT i.order_id) {ITEMS})"),
    GoldQuery("freight", "gmv_with_freight" + NOW, f"SELECT SUM(i.price + i.freight_value) {ITEMS}",
              f"SELECT SUM(i.price) {ITEMS}"),
    GoldQuery("payments", "payments" + NOW,
              f"SELECT SUM(p.payment_value) FROM order_payments p JOIN orders o USING(order_id) WHERE {WHERE}",
              f"SELECT SUM(p.payment_value) FROM order_payments p JOIN orders o USING(order_id) WHERE {WHERE}"),
    GoldQuery("repair_drift", "orders" + NOW + " where status=delivered",
              f"SELECT COUNT(*) {ORDERS} AND order_status = 'delivered'",
              f"SELECT COUNT(*) {ORDERS} AND status = 'delivered'",
              "SELECT COUNT(*) FROM orders"),
    GoldQuery("window", "orders" + NOW, f"SELECT COUNT(*) {ORDERS}",
              f"SELECT COUNT(*) FROM orders o WHERE {WHERE.replace('< DATE', '<= DATE')}"),
    GoldQuery("conflict", "gmv excluding and including freight" + NOW, None,
              f"SELECT SUM(i.price + i.freight_value) {ITEMS}"),
    GoldQuery("unavailable", "net profit" + NOW, None, "SELECT SUM(price) FROM order_items"),
)
GOLD_DEV = (
    GoldQuery("dev_daily", "orders from 2018-06-01 to 2018-06-04 by day",
              "SELECT order_purchase_timestamp::DATE, COUNT(*) FROM orders "
              "WHERE order_purchase_timestamp::DATE BETWEEN '2018-06-01' AND '2018-06-03' "
              "GROUP BY 1 ORDER BY 1",
              "SELECT order_purchase_timestamp::DATE, COUNT(*) FROM orders "
              "WHERE order_purchase_timestamp::DATE BETWEEN '2018-06-01' AND '2018-06-03' "
              "GROUP BY 1 ORDER BY 1"),
)


@dataclass(frozen=True)
class Case:
    id: str
    before: tuple[str, str]
    current: tuple[str, str]
    # Actions below are scorer data. Runtime receives the comparison and snapshots only.
    edits: tuple[tuple[str, int | str, int, int], ...] = ()
    dimension: str | None = None
    control: str | None = None

    @property
    def question(self):
        before, earlier_end = self.before
        start, end = self.current
        return (f"why late_delivery_rate from {start} to {end} versus {before} to {earlier_end}"
                + (f" by {self.dimension}" if self.dimension else ""))


INV_TEST = (
    Case("single_factor", WINDOW, WINDOW, (("SP", 0, 1, 900),)),
    Case("conditional", WINDOW, WINDOW, (("SP", 3, 1, 1250),)),
    Case("offset", WINDOW, WINDOW, (("SP", 1, 1, 150), ("RJ", 2, -1, 150))),
    Case("composition", WINDOW, WINDOW, (("SP", 0, 0, 7500),)),
    Case("linked_proxy", WINDOW, WINDOW, (("SP", 5, 2, 1100),)),
    Case("normal", WINDOW, WINDOW, control="no_material_change"),
    Case("missing", ("2015-01-01", "2015-02-01"), WINDOW, control="insufficient_evidence"),
    Case("historical_fixed", ("2018-03-01", "2018-04-01"), ("2018-04-01", "2018-05-01"),
         dimension="state"),
    Case("historical_open", ("2018-04-01", "2018-05-01"), ("2018-05-01", "2018-06-01")),
)
INV_DEV = (Case("dev_north", ("2018-02-01", "2018-03-01"), ("2018-02-01", "2018-03-01"),
                (("MG", 0, 1, 140),)),)


def counts(database: Path, window: tuple[str, str], state: str | None = None) -> tuple[int, int]:
    with duckdb.connect(str(database), read_only=True) as connection:
        return connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM({LATE}), 0) FROM orders "
            "JOIN customers USING(customer_id) "
            f"WHERE order_purchase_timestamp >= ? AND order_purchase_timestamp < ? AND {ELIGIBLE}"
            + (" AND customer_state = ?" if state else ""),
            (*window, state) if state else window,
        ).fetchone()



def historical_groups(database: Path, case: Case) -> dict[str, list[list[tuple]]]:
    columns = {"state": "customer_state", "weekday": "ISODOW(order_purchase_timestamp)",
               "status": "order_status"}
    dimensions = (case.dimension,) if case.dimension else tuple(columns)
    output = {}
    with duckdb.connect(str(database), read_only=True) as connection:
        for width in (1, 2):
            for action in combinations(dimensions, width):
                keys = ", ".join(f"COALESCE(({columns[name]})::VARCHAR, '(unknown)')" for name in action)
                statement = (
                    f"SELECT {keys}, COUNT(*), SUM({LATE}) FROM orders "
                    "LEFT JOIN customers USING(customer_id) "
                    f"WHERE order_purchase_timestamp >= ? AND order_purchase_timestamp < ? AND {ELIGIBLE} "
                    "GROUP BY ALL ORDER BY ALL"
                )
                output["+".join(action)] = [
                    connection.execute(statement, window).fetchall() for window in (case.before, case.current)
                ]
    return output

def snapshot(database: Path, case: Case, target: Path) -> tuple[Path, dict]:
    """Copy only for controlled experiments and return the scorer-only manifest."""
    if not case.edits:
        return database, {"baseline": counts(database, case.before),
                          "current": counts(database, case.current), "edits": [],
                          "historical_groups": historical_groups(database, case) if not case.control else {}}
    shutil.copyfile(database, target)
    edits = []
    with duckdb.connect(str(target)) as connection:
        for state, weekday, direction, limit in case.edits:
            params = [*case.current, state]
            day_filter = ""
            if weekday:
                day_filter = "AND isodow(o.order_purchase_timestamp) = ?"
                params.append(weekday)
            frame = f"""FROM orders o JOIN customers c USING(customer_id)
                WHERE o.order_purchase_timestamp >= ? AND o.order_purchase_timestamp < ?
                AND c.customer_state = ? AND {ELIGIBLE} {day_filter}"""
            if direction:
                frame += f" AND {LATE} = {0 if direction > 0 else 1}"
            connection.execute(
                f"CREATE OR REPLACE TEMP TABLE selected AS SELECT o.order_id, {LATE} AS was_late "
                f"{frame} ORDER BY md5(o.order_id || 'frozen-2026-v1'), o.order_id LIMIT {limit}",
                params,
            )
            n, was_late = connection.execute("SELECT COUNT(*), SUM(was_late) FROM selected").fetchone()
            if not n:
                raise ValueError(f"No eligible intervention rows in {case.id}")
            if direction:
                assign = ("order_delivered_customer_date = order_estimated_delivery_date "
                          + ("+ INTERVAL '1 day'" if direction > 0 else ""))
                if direction == 2:
                    assign += ", order_status = 'shipped'"
            else:
                # Move selected observations outside the measured window, keeping
                # their per-order delivery delays and relationships intact.
                assign = ", ".join(f"{column} = {column} + INTERVAL '2 years'" for column in (
                    "order_purchase_timestamp", "order_delivered_customer_date",
                    "order_estimated_delivery_date", "order_approved_at", "order_delivered_carrier_date",
                ))
            connection.execute(f"UPDATE orders SET {assign} WHERE order_id IN (SELECT order_id FROM selected)")
            edits.append({"state": state, "weekday": weekday, "direction": direction,
                          "matched": n, "original_late": was_late,
                          "late_delta": (n if direction > 0 else -n) if direction else -was_late,
                          "n_delta": 0 if direction else -n})
    before, original, current = (counts(database, case.before), counts(database, case.current),
                                  counts(target, case.current))
    expected = (original[0] + sum(edit["n_delta"] for edit in edits),
                original[1] + sum(edit["late_delta"] for edit in edits))
    if current != expected:
        raise ValueError(f"Intervention {case.id} failed its count conservation check")
    for edit in edits:
        if edit["direction"] == 0:
            n, late = counts(database, case.current, edit["state"])
            m, after_late = counts(target, case.current, edit["state"])
            np, mp = original[0], current[0]
            center = (original[1] / np + current[1] / mp) / 2
            segment_midpoint = (late / n + after_late / m) / 2 if m else late / n
            edit["mix_contribution"] = (m / mp - n / np) * (segment_midpoint - center)
        if edit["direction"] == 2:
            edit["induced_proxy"] = {"status": "shipped", "relabeled_rows": edit["matched"]}
    return target, {"baseline": before, "unmodified_current": original,
                    "current": current, "edits": edits}
