from __future__ import annotations

import json
from pathlib import Path

from data_analysis_agent.csv_analysis import analyze_csv
from scripts.evaluate_insightbench import audit


def step(sql, question="How many records are usable?"):
    return {"sql": sql, "question": question, "why": "Check coverage.", "columns": ["occurred_at"],
            "check": "Compare with total rows."}


def review(*, next_step=None, insight=None, gaps=()):
    return {"sufficient": not gaps, "answered": [], "gaps": list(gaps), "explanations": [],
            "next_step": next_step, "insights": [insight] if insight else []}


def insight(statement="There are {0} usable records.", evidence="E2"):
    return {"statement": statement, "cells": [{"evidence": evidence, "row": 0, "column": "valid"}],
            "scope": "All input rows.", "limitations": [], "kind": "descriptive"}


def make_csv(tmp_path: Path) -> Path:
    path = tmp_path / "unknown.csv"
    path.write_text('occurred_at,"invoice total",channel\n2026-01-01,10,online\nbad,20,store\n,30,online\n')
    return path


def test_csv_review_changes_next_query_based_on_data(tmp_path):
    csv = make_csv(tmp_path)
    prompts = []
    first = step("SELECT count(*) AS total, count(TRY_CAST(occurred_at AS TIMESTAMP)) AS valid FROM source_data")
    preplanned = step("SELECT channel, count(*) AS n FROM source_data GROUP BY channel", "Which channel is bigger?")
    updated = step("SELECT count(*) AS valid FROM source_data WHERE TRY_CAST(occurred_at AS TIMESTAMP) IS NOT NULL")

    def plan(prompt):
        context = json.loads(prompt.split("\n", 1)[1])
        prompts.append(context)
        if context["phase"] == "initial":
            return {"steps": [first, preplanned]}
        if context["phase"] == "review" and len(context["evidence"]) == 1:
            total, valid = context["evidence"][0]["rows"][0]
            assert total == 3 and valid == 1
            return review(next_step=updated, gaps=["What is the count with a valid timestamp?"])
        return review(insight=insight(evidence="E1" if context["phase"] == "final" else "E2"))

    dynamic = analyze_csv(csv, "Describe channel coverage and dates.", planner=plan)
    fixed = analyze_csv(csv, "Describe channel coverage and dates.", planner=plan, strategy="fixed")
    assert prompts[0] == prompts[3]
    columns = {column["name"]: column for column in dynamic["profile"]["columns"]}
    assert columns["occurred_at"]["valid_time"] == 1
    assert columns["occurred_at"]["missing_rate"] == 0.3333
    assert columns["invoice total"]["mean"] == 20
    assert columns["channel"]["kind"] == "category"
    assert dynamic["evidence"][1]["step"]["sql"] == updated["sql"]
    assert fixed["evidence"][1]["step"]["sql"] == preplanned["sql"]
    assert dynamic["trace"][2]["after"]["gaps"]
    assert dynamic["insights"][0]["statement"] == "There are 1 usable records."
    assert audit(dynamic)["passed_value_checks"] == 1


def test_csv_blocks_mutation_and_external_files_and_recovers(tmp_path):
    csv = make_csv(tmp_path)
    external = tmp_path / "outside.csv"
    external.write_text("private\n77\n")
    actions = iter([
        step("DELETE FROM source_data"),
        step(f"SELECT * FROM read_csv_auto('{external}')"),
        step("SELECT count(*) AS valid FROM source_data"),
    ])

    def plan(prompt):
        context = json.loads(prompt.split("\n", 1)[1])
        if context["phase"] == "initial":
            return {"steps": [next(actions)]}
        if len(context["evidence"]) < 3:
            assert context["evidence"][-1]["error"]
            return review(next_step=next(actions), gaps=["Which rows are measurable?"])
        return review(insight=insight(evidence="E3"))

    report = analyze_csv(csv, "Check data coverage.", planner=plan)
    assert [item["status"] for item in report["evidence"]] == ["error", "error", "success"]
    assert report["evidence"][-1]["rows"] == [[3]]
    assert report["insights"][0]["facts"][0]["value"] == 3


def test_csv_rejects_fake_number_then_retains_partial_report(tmp_path):
    def plan(prompt):
        context = json.loads(prompt.split("\n", 1)[1])
        if context["phase"] == "initial":
            return {"steps": [step("SELECT count(*) AS valid FROM source_data")]}
        statement = "Of 30 records, {0} are usable." if "correction" not in context else "There are {0} rows."
        return review(insight=insight(statement, "E1"), gaps=["Closure reason is unknown."])

    report = analyze_csv(make_csv(tmp_path), "Explain why closure is slow.", planner=plan, strategy="fixed")
    assert report["trace"][2] == {"phase": "final", "error": "unverified_values", "retry": True}
    assert not report["assessment"]["sufficient"]
    assert report["assessment"]["gaps"] == ["Closure reason is unknown."]
    assert report["insights"][0]["statement"] == "There are 3 rows."
    report["insights"][0]["facts"][0]["value"] = 4
    assert audit(report)["passed_value_checks"] == 0


def test_csv_rejects_nonexistent_evidence(tmp_path):
    def plan(prompt):
        context = json.loads(prompt.split("\n", 1)[1])
        if context["phase"] == "initial":
            return {"steps": [step("SELECT count(*) AS valid FROM source_data")]}
        return review(insight=insight(evidence="E99"))

    report = analyze_csv(make_csv(tmp_path), "Count the records.", planner=plan)
    assert report["error"] == "invalid_evidence_cell"
    assert not report["insights"]
