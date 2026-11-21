"""Minimal Streamlit workbench for the Data Analysis Agent."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from data_analysis_agent.routing import run_task
from data_analysis_agent.streamlit_view import DEFAULT_DATABASE_PATH


EXAMPLES = {
    "olist": (
        "orders from 2018-03-01 to 2018-04-01",
        "gmv from 2018-03-01 to 2018-06-01 by month",
        "why late_delivery_rate from 2018-05-01 to 2018-06-01 versus 2018-04-01 to 2018-05-01",
    ),
    "bike": (
        "rentals from 2012-06-01 to 2012-07-01",
        "rentals from 2012-06-01 to 2012-07-01 where workingday=1",
        "why average_hourly_rentals from 2012-07-01 to 2012-08-01 versus 2012-06-01 to 2012-07-01",
    ),
}


def _select_example(question: str) -> None:
    st.session_state.question_input = question


def main() -> None:
    st.title("数据分析助手")
    domain = st.selectbox("数据源", ("olist", "bike"))
    database = DEFAULT_DATABASE_PATH if domain == "olist" else Path("data/processed/bike.duckdb")
    for column, example in zip(st.columns(3), EXAMPLES[domain], strict=True):
        column.button(example, on_click=_select_example, args=(example,), width="stretch")
    question = st.text_area("业务问题", key="question_input")
    if not st.button("开始分析"):
        return
    if not database.is_file():
        st.error(f"请先运行 python -m scripts.prepare_data {domain}")
        return
    try:
        with st.spinner("正在分析…"):
            payload = run_task(database, question, domain)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if payload.get("error"):
        st.warning(payload["error"])
        return
    st.caption(f"执行路径 · {payload['route']}")
    if payload["route"] == "query":
        result = payload["result"]
        st.dataframe([dict(zip(result["columns"], row)) for row in result["rows"]])
        st.write(payload["metric_definition"]["definition"])
    else:
        st.write(payload["stop_reason"])
        if payload["facts"]:
            st.write(payload["facts"])
        if payload["statistical_attributions"]:
            st.dataframe(payload["statistical_attributions"])
        st.caption(payload["interpretation"])
    with st.expander("查看 SQL、校验与证据"):
        st.json(json.loads(json.dumps(payload, default=str)), expanded=False)


if __name__ == "__main__":
    main()
