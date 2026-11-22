"""Data pipeline regression tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from data_analysis_agent.schema import DatabaseSchema, inspect_schema
from scripts.build_duckdb import build_database, DEFAULT_OUTPUT_PATH, TABLE_SPECS
from scripts.check_join_grain import analyze_join_grain, render_markdown
from scripts.profile_dataset import (
    calculate_relation_coverage,
    parse_datetime_series,
    validate_required_columns,
)
from scripts.stage_database import ArtifactStagingError, stage_database


# Build duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("olist_duckdb") / "olist.duckdb"
    build_database(DATA_DIR, path)
    return path


@pytest.fixture()
def connection(database_path: Path):
    with duckdb.connect(str(database_path), read_only=True) as database:
        yield database


def test_table_row_counts_match_the_audited_csv_version(connection) -> None:
    for spec in TABLE_SPECS:
        actual = connection.execute(
            f"SELECT COUNT(*) FROM {spec.table_name}"
        ).fetchone()[0]
        assert actual == spec.expected_row_count


def test_primary_key_assumptions_remain_unique(connection) -> None:
    for spec in TABLE_SPECS:
        key_columns = ", ".join(spec.primary_key)
        duplicate_groups = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT {key_columns}, COUNT(*) AS row_count
                FROM {spec.table_name}
                GROUP BY {key_columns}
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        assert duplicate_groups == 0


def test_core_joins_execute_without_orphaning_child_rows(connection) -> None:
    joins = (
        ("orders", "customers", "customer_id"),
        ("order_items", "orders", "order_id"),
        ("order_payments", "orders", "order_id"),
        ("order_reviews", "orders", "order_id"),
        ("order_items", "products", "product_id"),
        ("order_items", "sellers", "seller_id"),
    )
    for child, parent, key in joins:
        missing = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {child} AS child
            LEFT JOIN {parent} AS parent USING ({key})
            WHERE parent.{key} IS NULL
            """
        ).fetchone()[0]
        assert missing == 0


def test_timestamp_columns_are_typed_and_queryable(connection) -> None:
    types = dict(
        connection.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'orders'
            """
        ).fetchall()
    )
    assert types["order_purchase_timestamp"] == "TIMESTAMP"
    assert types["order_delivered_customer_date"] == "TIMESTAMP"
    date_range = connection.execute(
        """
        SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp)
        FROM orders
        """
    ).fetchone()
    assert date_range[0] < date_range[1]


def test_order_level_views_prevent_item_payment_amount_amplification(connection) -> None:
    source_totals = connection.execute(
        """
        SELECT
            (SELECT SUM(price) FROM order_items),
            (SELECT SUM(freight_value) FROM order_items),
            (SELECT SUM(payment_value) FROM order_payments)
        """
    ).fetchone()
    safe_totals = connection.execute(
        """
        SELECT
            SUM(COALESCE(items.item_transaction_value, 0)),
            SUM(COALESCE(items.freight_value, 0)),
            SUM(COALESCE(payments.payment_value, 0))
        FROM order_item_summary AS items
        FULL OUTER JOIN order_payment_summary AS payments USING (order_id)
        """
    ).fetchone()
    assert safe_totals == source_totals

    common_source_item_total, naive_item_total = connection.execute(
        """
        SELECT
            (
                SELECT SUM(summary.item_transaction_value)
                FROM order_item_summary AS summary
                JOIN order_payment_summary AS payments USING (order_id)
            ),
            (
                SELECT SUM(items.price)
                FROM order_items AS items
                JOIN order_payments AS payments USING (order_id)
            )
        """
    ).fetchone()
    assert naive_item_total > common_source_item_total


def test_category_translation_preserves_both_language_columns(connection) -> None:
    columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'products_with_category_translation'
            """
        ).fetchall()
    }
    assert "product_category_name" in columns
    assert "product_category_name_english" in columns


def test_default_database_output_is_ignored_and_untracked() -> None:
    relative_path = str(DEFAULT_OUTPUT_PATH)
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert ignored.returncode == 0
    assert tracked.returncode != 0


# Check join grain

def test_join_grain_quantifies_amplification_and_conserves_preaggregated_totals() -> None:
    order_items = pd.DataFrame(
        {
            "order_id": ["O-1", "O-1", "O-2", "O-3"],
            "price": ["10.00", "20.00", "5.00", "7.00"],
            "freight_value": ["1.00", "2.00", "0.00", "0.00"],
        }
    )
    order_payments = pd.DataFrame(
        {
            "order_id": ["O-1", "O-1", "O-2", "O-4"],
            "payment_value": ["15.00", "18.00", "5.00", "9.00"],
        }
    )

    result = analyze_join_grain(order_items, order_payments)

    assert result.common_order_count == 2
    assert result.item_only_order_count == 1
    assert result.payment_only_order_count == 1
    assert result.orders_with_multiple_items == 1
    assert result.orders_with_multiple_payments == 1
    assert result.orders_with_multiple_items_and_payments == 1
    assert result.common_item_row_count == 3
    assert result.common_payment_row_count == 3
    assert result.naive_join_row_count == 5
    assert result.common_item_total_cents == 3_800
    assert result.naive_item_total_cents == 7_100
    assert result.baseline_item_total_cents == 4_500
    assert result.preaggregated_item_total_cents == 4_500
    assert result.common_payment_total_cents == 3_800
    assert result.naive_payment_total_cents == 7_100
    assert result.baseline_payment_total_cents == 4_700
    assert result.preaggregated_payment_total_cents == 4_700
    assert result.amounts_conserved_after_preaggregation is True
    assert "outer 订单集合预聚合后全量金额守恒 | 通过" in render_markdown(result)


def test_join_grain_rejects_invalid_money() -> None:
    order_items = pd.DataFrame(
        {"order_id": ["O-1"], "price": ["invalid"], "freight_value": ["1.00"]}
    )
    order_payments = pd.DataFrame(
        {"order_id": ["O-1"], "payment_value": ["1.00"]}
    )

    with pytest.raises(ValueError, match="price 包含 1 个无效金额"):
        analyze_join_grain(order_items, order_payments)


# Inspect schema

@pytest.fixture(scope="module")
def schema(database_path: Path) -> DatabaseSchema:
    return inspect_schema(database_path)


def _objects_by_name(schema: DatabaseSchema):
    return {obj.name: obj for obj in schema.objects}


def _columns_by_name(schema: DatabaseSchema, object_name: str):
    obj = _objects_by_name(schema)[object_name]
    return {column.name: column for column in obj.columns}


def test_inspect_schema_reads_columns_types_and_constraints(schema) -> None:
    order_columns = _columns_by_name(schema, "orders")
    item_columns = _columns_by_name(schema, "order_items")
    summary_columns = _columns_by_name(schema, "order_item_summary")

    assert order_columns["order_id"].data_type == "VARCHAR"
    assert order_columns["order_id"].nullable is False
    assert order_columns["order_id"].primary_key is True
    assert order_columns["order_purchase_timestamp"].data_type == "TIMESTAMP"
    assert item_columns["price"].data_type == "DECIMAL(18,2)"
    assert summary_columns["item_count"].data_type == "BIGINT"
    assert summary_columns["order_id"].primary_key is False


def test_inspect_schema_adds_only_known_grain_metadata(schema) -> None:
    objects = _objects_by_name(schema)

    assert objects["orders"].grain == "one row per order"
    assert objects["order_items"].grain == "one row per order item"
    assert objects["order_payments"].grain == "one row per payment record"
    assert objects["order_item_summary"].grain == "one row per order"
    assert objects["order_payment_summary"].grain == "one row per order"
    assert objects["products"].grain is None


def test_inspect_schema_reports_missing_database(tmp_path) -> None:
    missing_path = tmp_path / "missing.duckdb"

    with pytest.raises(
        FileNotFoundError,
        match=r"DuckDB database does not exist: .*missing\.duckdb",
    ):
        inspect_schema(missing_path)


# Profile dataset

def test_validate_required_columns_reports_missing_and_unexpected_fields() -> None:
    tables = {
        "orders": pd.DataFrame(
            {"order_id": ["O-1"], "unexpected": ["value"]}
        )
    }
    required = {
        "orders": ("order_id", "customer_id"),
        "customers": ("customer_id",),
    }

    errors = validate_required_columns(tables, required)

    assert errors == [
        "orders 缺少字段: customer_id",
        "orders 存在未审阅字段: unexpected",
        "缺少表: customers",
    ]


def test_parse_datetime_series_counts_invalid_values_and_range() -> None:
    values = pd.Series(
        [
            "2016-09-04 21:15:19",
            "2018-10-17 17:30:18",
            "2018/10/17 17:30:18",
            None,
        ]
    )

    result = parse_datetime_series(values, date_format="%Y-%m-%d %H:%M:%S")

    assert result.non_null_count == 3
    assert result.valid_count == 2
    assert result.invalid_count == 1
    assert result.minimum == pd.Timestamp("2016-09-04 21:15:19")
    assert result.maximum == pd.Timestamp("2018-10-17 17:30:18")


def test_calculate_relation_coverage_preserves_string_keys_and_orphans() -> None:
    child = pd.DataFrame({"zip_prefix": ["01037", "01046", "99999", None]})
    parent = pd.DataFrame({"zip_prefix": ["01037", "01046", "01046"]})

    result = calculate_relation_coverage(
        child, "zip_prefix", parent, "zip_prefix"
    )

    assert result.child_non_null_rows == 3
    assert result.child_missing_rows == 1
    assert result.matched_rows == 2
    assert result.row_coverage == 2 / 3
    assert result.child_unique_keys == 3
    assert result.matched_unique_keys == 2
    assert result.unique_key_coverage == 2 / 3
    assert result.parent_duplicate_keys == 1
    assert result.parent_unique_keys == 2
    assert result.referenced_parent_unique_keys == 2
    assert result.parent_key_coverage == 1.0
    assert result.orphan_key_samples == ("99999",)


# Stage database

ARTIFACT_CONTENT = b"verified-duckdb-artifact"


ARTIFACT_URL = "https://example.invalid/data-v1/olist.duckdb"


def _write_manifest(
    path: Path,
    *,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "data-v1",
                "url": ARTIFACT_URL,
                "size_bytes": (
                    len(ARTIFACT_CONTENT) if size_bytes is None else size_bytes
                ),
                "sha256": (
                    hashlib.sha256(ARTIFACT_CONTENT).hexdigest()
                    if sha256 is None
                    else sha256
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def _local_download(url: str, destination: Path) -> None:
    assert url == ARTIFACT_URL
    destination.write_bytes(ARTIFACT_CONTENT)


def test_valid_download_is_verified_and_atomically_published(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json")
    target_path = tmp_path / "data" / "processed" / "olist.duckdb"

    downloaded = stage_database(
        manifest_path,
        target_path,
        downloader=_local_download,
    )

    assert downloaded is True
    assert target_path.read_bytes() == ARTIFACT_CONTENT
    assert not list(target_path.parent.glob("*.download"))


def test_sha256_mismatch_fails_without_publishing_target(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json", sha256="0" * 64)
    target_path = tmp_path / "data" / "olist.duckdb"

    with pytest.raises(ArtifactStagingError, match="SHA256 mismatch"):
        stage_database(manifest_path, target_path, downloader=_local_download)

    assert not target_path.exists()
    assert not list(target_path.parent.glob("*.download"))


def test_size_mismatch_fails_without_publishing_target(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        size_bytes=len(ARTIFACT_CONTENT) + 1,
    )
    target_path = tmp_path / "data" / "olist.duckdb"

    with pytest.raises(ArtifactStagingError, match="size mismatch"):
        stage_database(manifest_path, target_path, downloader=_local_download)

    assert not target_path.exists()
    assert not list(target_path.parent.glob("*.download"))


def test_download_failure_leaves_no_target_or_temporary_file(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json")
    target_path = tmp_path / "data" / "olist.duckdb"

    def failed_download(url: str, destination: Path) -> None:
        raise OSError("network unavailable")

    with pytest.raises(ArtifactStagingError, match="download failed"):
        stage_database(manifest_path, target_path, downloader=failed_download)

    assert not target_path.exists()
    assert not list(target_path.parent.glob("*.download"))
