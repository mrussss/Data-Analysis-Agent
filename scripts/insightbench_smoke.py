"""Scripted planner replies for end-to-end integration, never an online arm."""
from __future__ import annotations

import json


def _step(question, columns, sql):
    return {"question": question, "why": "Measure a part of the business goal.",
            "columns": columns, "check": "Inspect group sizes and comparison scope.", "sql": sql}


CATEGORY = _step("How many incidents are in each category?", ["category"],
                 "SELECT category, count(*) AS incidents FROM source_data GROUP BY category ORDER BY incidents DESC, category")
ASSIGNMENT = _step("How many incidents are assigned per category?", ["category", "assigned_to"],
                   "SELECT category, count(assigned_to) AS assigned FROM source_data GROUP BY category ORDER BY assigned DESC, category")
CONCENTRATION = _step("How concentrated are assignments in the largest category?", ["category", "assigned_to"],
                      """SELECT category, assigned_to, count(*) AS incidents FROM source_data
                         WHERE category = (SELECT category FROM source_data GROUP BY category ORDER BY count(*) DESC, category LIMIT 1)
                         GROUP BY category, assigned_to ORDER BY incidents DESC, assigned_to""")
MONTH = "date_trunc('month', TRY_CAST(opened_at AS TIMESTAMP))"
DURATION = "date_diff('second', TRY_CAST(opened_at AS TIMESTAMP), TRY_CAST(closed_at AS TIMESTAMP)) / 86400.0"
TIMING = _step("How does observed resolution time vary by opening month?", ["opened_at", "closed_at"],
               f"""SELECT {MONTH} AS month, count(*) AS total, count({DURATION}) AS resolved,
                          round(avg({DURATION}), 2) AS days
                   FROM source_data GROUP BY month ORDER BY month""")
PRIORITY = _step("How does observed duration vary by priority?", ["priority", "opened_at", "closed_at"],
                 f"""SELECT priority, count(*) AS total, count({DURATION}) AS resolved,
                            round(avg({DURATION}), 2) AS days FROM source_data
                     GROUP BY priority ORDER BY priority""")
MISSING = _step("How many rows lack a usable closure timestamp?", ["closed_at"],
                """SELECT count(*) FILTER (WHERE TRY_CAST(closed_at AS TIMESTAMP) IS NULL) AS missing,
                          count(*) AS total FROM source_data""")


def _insight(statement, cells, scope, limitations=()):
    return {"statement": statement,
            "cells": [{"evidence": evidence, "row": row, "column": column} for evidence, row, column in cells],
            "scope": scope, "limitations": list(limitations), "kind": "descriptive"}


def plan(prompt: str) -> dict:
    context = json.loads(prompt.split("\n", 1)[1])
    case_one = context["goal"].startswith("Find the discrepancy")
    if context["phase"] == "initial":
        return {"steps": [CATEGORY, ASSIGNMENT] if case_one else [TIMING, PRIORITY]}
    evidence = context["evidence"]
    first = evidence[0]
    if case_one:
        insight = _insight("{0} has the most incidents across categories, with {1} recorded incidents.",
                           [("E1", 0, "category"), ("E1", 0, "incidents")], "All rows in the CSV.")
        gaps = []
    else:
        index = next(i for i, data in enumerate(first["rows"]) if data[3] is not None)
        insight = _insight("Incidents opened in {0} took {1} days on average among {2} records with valid timestamps.",
                           [("E1", index, "month"), ("E1", index, "days"), ("E1", index, "resolved")],
                           "Only records with valid opening and closure times.",
                           ["Open or invalid records do not contribute to the mean."])
        gaps = ["The cause of the time pattern remains unresolved."]
    response = {"sufficient": False, "answered": [{"question": first["step"]["question"],
                                                  "finding": "Computed grouped observations.", "evidence": ["E1"]}],
                "gaps": gaps, "explanations": []}
    if context["phase"] == "review" and len(evidence) == 1:
        observed = first["rows"]
        if case_one and observed[0][1] > 2 * observed[1][1]:
            response.update(gaps=["Are assignments concentrated within the largest category?"],
                            next_step=CONCENTRATION)
        elif not case_one and any(row[1] != row[2] for row in observed):
            response.update(gaps=["How many rows lack a usable closing time?"], next_step=MISSING)
        else:
            response.update(gaps=["How do comparison groups differ?"], next_step=ASSIGNMENT if case_one else PRIORITY)
        return response
    if len(evidence) > 1:
        second = evidence[1]
        if "missing" in second["columns"]:
            missing = _insight("{0} of {1} incidents lack a usable closure timestamp.",
                               [("E2", 0, "missing"), ("E2", 0, "total")], "All rows in the CSV.")
        elif "assigned_to" in second["columns"]:
            missing = _insight("Within {0}, the assignee with the most incidents has {1}.",
                               [("E2", 0, "category"), ("E2", 0, "incidents")], "Largest category only.")
        elif case_one:
            missing = _insight("{0} has {1} incidents with an assigned owner.",
                               [("E2", 0, "category"), ("E2", 0, "assigned")], "Category with the most incidents that have a recorded assignee.")
        else:
            missing = _insight("Priority {0} incidents average {1} days among {2} records with usable timestamps.",
                               [("E2", 0, "priority"), ("E2", 0, "days"), ("E2", 0, "resolved")], "First listed priority.")
        if not case_one and context["phase"] == "review" and len(evidence) == 2 and "missing" in second["columns"]:
            response.update(next_step=PRIORITY)
            return response
        response["insights"] = [insight, missing]
    else:
        response["insights"] = [insight]
    response["sufficient"] = not gaps
    return response
