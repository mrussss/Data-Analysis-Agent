import json

import duckdb
import pytest

from data_analysis_agent.benchmark import effective_replans, score_investigation
from data_analysis_agent.benchmark_cases import Case, counts, snapshot
from data_analysis_agent.investigation import investigate, parse_comparison, report
from data_analysis_agent.routing import route_question


WINDOW = ("2018-01-01", "2018-01-08")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "orders.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE customers AS
                SELECT 'SP' AS customer_id, 'SP' AS customer_state UNION ALL SELECT 'RJ', 'RJ';
            CREATE TABLE orders AS
                SELECT CAST(i AS VARCHAR) AS order_id, IF((i // 7) % 2 = 0, 'SP', 'RJ') AS customer_id,
                       DATE '2018-01-01' + (i % 7)::INTEGER AS order_purchase_timestamp,
                       'delivered' AS order_status,
                       DATE '2018-01-13' + (i % 5 = 0)::INT AS order_delivered_customer_date,
                       DATE '2018-01-13' AS order_estimated_delivery_date,
                       DATE '2018-01-08' AS order_approved_at,
                       DATE '2018-01-08' AS order_delivered_carrier_date
                FROM range(1400) AS t(i);
        """)
    return path


def choice(dimensions, *, evidence="E1", cohort=None, expect=None,
           next_action=None, revises=None):
    return {
        "action": {"dimensions": dimensions, "cohort": cohort or {}}, "expect": expect,
        "evidence_ids": [evidence], "question": "Which population accounts for the change?",
        "next_action": next_action, "revises": revises,
    }


def test_model_changes_executed_action_after_refutation_without_seeing_manifest(source, tmp_path):
    case = Case("hidden-mechanism", WINDOW, WINDOW, (("SP", 3, 1, 65),))
    observed, manifest = snapshot(source, case, tmp_path / "observed.duckdb")
    assert counts(source, WINDOW) == (1400, 280)
    assert manifest["current"] == (1400, 345)
    goal = parse_comparison(case.question, "olist")
    planned = {"dimensions": ["status"]}
    inputs = []

    def planner(payload):
        data = json.loads(payload)
        inputs.append(data)
        assert "rule_choice" not in data and "hidden-mechanism" not in payload and "matched" not in payload
        if len(inputs) == 1:
            return choice(["state"], expect="stable", next_action=planned)
        if len(inputs) == 2:
            assert data["tested_hypotheses"][0]["state"] == "refuted"
            return choice(["weekday"], cohort={"state": "SP"}, evidence="E2", revises="H1")
        return {"stop": "sufficient_evidence", "evidence_ids": ["E3"],
                "question": "Can the localized shift account for the observed change?"}

    adaptive = investigate((source, observed), goal, policy="llm_adaptive", budget=6, planner=planner)
    assert [e.action for e in adaptive.evidence] == [(), ("state",), ("weekday",)]
    assert adaptive.evidence[-1].cohort == (("state", "SP"),)
    assert "state = 'SP'" in adaptive.evidence[-1].sql[0]
    assert adaptive.model_hypotheses[-1].segment == ("3",)
    assert adaptive.model_hypotheses[-1].state == "observed"
    assert effective_replans(adaptive) == (1, 1)
    scored = score_investigation(adaptive, manifest, None)
    assert scored["correct"] and scored["top1"] and scored["supported_claims"] == scored["claims"]
    assert "gold" not in report(adaptive)
    assert adaptive.stop_reason == "sufficient_evidence"
    assert adaptive.remaining_budget == 3


def test_frozen_only_sees_headline_and_own_actions(source):
    data = []

    def planner(payload):
        data.append(json.loads(payload))
        assert not data[-1]["tested_hypotheses"] and not data[-1]["statistical_candidates"]
        assert [e["id"] for e in data[-1]["evidence"]] == ["E1"]
        if len(data) == 1:
            return choice(["state"])
        assert len(data[-1]["past_decisions"]) == 1
        return choice([], cohort={"state": "RJ"})

    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    state = investigate((source, source), goal, policy="llm_frozen", budget=3, planner=planner)
    assert [step["evidence"] for step in state.trace] == ["E1", "E2", "E3"]
    assert state.evidence[-1].cohort == (("state", "RJ"),)
    assert effective_replans(state) == (0, 0)


@pytest.mark.parametrize("invalid", [
    choice(["state"], evidence="E999"),
    choice([], cohort={"state": "SP' OR TRUE --"}),
])
def test_invalid_model_action_stops_without_a_rule_fallback(source, invalid):
    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    state = investigate((source, source), goal, policy="llm_adaptive",
                        budget=3, planner=lambda _: invalid)
    assert len(state.evidence) == 1
    assert state.stop_reason == "planner_error"
    assert not state.reported


@pytest.mark.parametrize("case", [
    Case("normal", WINDOW, WINDOW, control="no_material_change"),
    Case("empty", ("2017-01-01", "2017-02-01"), WINDOW, control="insufficient_evidence"),
])
def test_unsupported_attribution_stops_without_claims(source, case):
    state = investigate((source, source), parse_comparison(case.question, "olist"))
    assert state.stop_reason == case.control
    assert state.reported == []


def test_route_uses_task_requirements():
    assert route_question("orders from 2018-01-01 to 2018-01-08", "olist") == "query"
    open_question = Case("test", WINDOW, WINDOW).question
    assert route_question(open_question, "olist") == "adaptive"
    assert route_question(open_question + " by weekday", "olist") == "workflow"


def test_repeated_action_can_be_corrected_in_the_existing_decision_budget(source):
    goal = parse_comparison(Case("normal", WINDOW, WINDOW).question, "olist")
    decisions = iter([choice(["state"]), choice(["state"]), choice(["weekday"])])
    attempts = []

    def planner(payload):
        attempts.append(json.loads(payload))
        return next(decisions)

    result = investigate((source, source), goal, policy="llm_adaptive", budget=4, planner=planner)
    assert [item.action for item in result.evidence] == [(), ("state",), ("weekday",)]
    assert attempts[-1]["previous_decision_error"] == "repeated_action"
    assert result.stop_reason == "decision_budget_exhausted"
    assert result.model_error is None
    assert len(attempts) == 3 and result.remaining_budget == 1


@pytest.mark.parametrize(("content", "finish", "status"), [
    ("   ", "stop", "empty_output"),
    ('{"action":', "length", "truncated_output"),
    ("not JSON", "stop", "invalid_json"),
    ('```json\n{"evidence_ids":["E1"]}\n```', "stop", "json"),
])
def test_planner_diagnoses_provider_output_without_storing_raw_content(monkeypatch, content, finish, status):
    from types import SimpleNamespace as NS

    from data_analysis_agent import model_budget

    provider_arguments = {}

    def create(**kwargs):
        provider_arguments.update(kwargs)
        return NS(choices=[NS(finish_reason=finish, message=NS(content=content, reasoning_content=""))],
                  usage=NS(prompt_tokens=40, completion_tokens=10, total_tokens=50))

    monkeypatch.setenv("LLM_ROUTE_ORDER", "")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-test")
    monkeypatch.delenv("MODEL_SPEND_CAP_USD", raising=False)
    monkeypatch.setattr(model_budget, "OpenAI",
                        lambda **_: NS(chat=NS(completions=NS(create=create))))
    model = model_budget.BudgetedModel()
    if status == "json":
        assert model.plan("{}") == {"evidence_ids": ["E1"]}
    else:
        with pytest.raises(model_budget.PlannerResponseError, match=status):
            model.plan("{}")
    record, = model.planner_calls
    assert record["response_status"] == status
    assert (record["finish_reason"], record["content_chars"], record["total_tokens"]) == (finish, len(content), 50)
    assert not {"content", "prompt", "headers"} & record.keys()
    assert provider_arguments["extra_body"] == {"thinking": {"type": "enabled"}}
    assert provider_arguments["max_tokens"] == 3072 and provider_arguments["reasoning_effort"] == "low"


def test_business_frozen_plan_is_isolated_and_uses_shared_event_search(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "after_onset")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    payloads = []

    def plan(prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        payloads.append(payload)
        assert not {"verdict", "changes", "findings", "case"} & set(payload["goal"])
        assert case.id not in json.dumps(payload)
        if len(payloads) == 1:
            assert [item["id"] for item in payload["evidence"]] == ["E1"]
            assert {"stock", "events", "timeline"} <= payload["tools"].keys()
            return {"actions": ["events", "timeline"], "reason": "Check event and onset dates."}
        assert [item["id"] for item in payload["evidence"]] == ["E1", "E2", "E3"]
        return {"verdict": "insufficient", "claims": [
            {"evidence": id, "fact": fact} for id, fact in zip(("E1", "E3", "E2"), case.findings)
        ]}

    state = investigate((database, database), case.goal, budget=5, policy="llm_frozen", planner=plan, events=events)
    assert len(payloads) == 2
    assert [step["visible_evidence"] for step in state.trace[1:]] == [["E1"], ["E1"]]
    assert score(state, case)["correct"]


def test_business_refutation_changes_executed_path(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "recall_scope")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    actions = iter(("events", "stock_core", "scope", "complaints"))

    def plan(prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if "verdicts" in payload:
            return {"verdict": "plausible", "claims": [
                {"evidence": item["id"], "fact": fact} for item in payload["evidence"] for fact in item["facts"]
            ]}
        action = next(actions)
        decision = {"action": action, "question": "Can this evidence distinguish the explanations?"}
        if action == "stock_core":
            decision.update(expect="stock_core:availability=down", next="pricing")
        if action == "scope":
            assert payload["previous"]["refutation"] == {"evidence": "E3", "actual": "stock_core:availability=stable"}
        return decision

    state = investigate((database, database), case.goal, budget=5, policy="llm_adaptive", planner=plan, events=events)
    result = score(state, case)
    assert result["correct"] and result["changes_after_refutation"] == 1
    assert result["refutations"] == 1 and state.remaining_budget == 0
    fixed = investigate((database, database), case.goal, budget=5, policy="workflow", events=events)
    assert all(item.action != ("stock_core",) for item in fixed.evidence)
    assert score(fixed, case)["correct"]


def test_business_scoring_detects_false_cause_despite_real_event(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import RETAIL_TEST, build_retail

    case = next(item for item in RETAIL_TEST if item.id == "stock_vs_bad_press")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    state = investigate((database, database), case.goal, budget=5, events=events)
    evidence = next(item for item in state.evidence if item.action == ("stock",))
    assert evidence.rows[0][1:4] == (1.0, 56, 56) and evidence.rows[1][1:4] == (0.5, 56, 56)
    assert evidence.rows[0][-1] == 1400 and evidence.rows[1][-1] == 0
    assert score(state, case)["correct"]
    state.verdict = "causal"
    state.claims.append({"evidence": "E999", "fact": "events:press=matched"})
    result = score(state, case)
    assert result["event_retrieval"] and result["event_matching"]
    assert result["wrong_attribution"] and not result["correct"]
    assert result["unsupported_claims"] == [("E999", "events:press=matched")]


def test_business_frozen_answers_are_reachable_and_splits_disjoint(tmp_path):
    from pathlib import Path

    from data_analysis_agent.business_cases import DEV, TEST, build_retail
    from data_analysis_agent.business_investigation import event_findings
    from data_analysis_agent.business_tools import observe

    assert {item.goal for item in TEST}.isdisjoint(item.goal for item in DEV)
    assert [sum(item.kind == group for item in TEST) for group in "ABCD"] == [8, 8, 8, 6]
    olist = Path("data/processed/olist.duckdb")
    if not olist.is_file():
        pytest.skip("Build the audited Olist database to verify all observational gold")
    for case in (*TEST, *DEV):
        database, events = olist, ()
        if case.goal.domain == "retail":
            database = tmp_path / f"{case.id}.duckdb"
            events = build_retail(case, database)
        prefixes = {fact.split(":")[0] for fact in case.findings}
        if case.goal.dimension in prefixes:
            prefixes.remove(case.goal.dimension)
            prefixes.add("breakdown")
        evidence = [observe(database, case.goal, action, f"E{i}", events)
                    for i, action in enumerate(sorted(prefixes | {"headline"}), 1)]
        assert all(not item.error for item in evidence), case.id
        findings = {fact for facts in event_findings(case.goal, evidence).values() for fact in facts}
        assert set(case.findings) <= findings, case.id
        assert len(evidence) <= 5, case.id


@pytest.mark.parametrize(("failure", "expected_requests", "expected_unknown"), [
    ("connect", 3, 0),
    ("read_timeout", 3, 2),
    ("rate_limited", 3, 2),
    ("unavailable", 3, 2),
    ("unauthorized", 1, 1),
])
def test_budgeted_low_retries_account_for_potential_provider_usage(
    monkeypatch, failure, expected_requests, expected_unknown,
):
    from types import SimpleNamespace as NS

    import httpx
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    from data_analysis_agent import model_budget

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    request_options = []

    def create(**options):
        request_options.append(options)
        if len(request_options) <= 2:
            if failure == "connect":
                raise APIConnectionError(request=request) from httpx.ConnectError("connection refused")
            if failure == "read_timeout":
                raise APITimeoutError(request=request) from httpx.ReadTimeout("response timeout")
            status = {"rate_limited": 429, "unavailable": 503, "unauthorized": 401}[failure]
            reply = httpx.Response(status, headers={"Retry-After": "2"}, request=request)
            raise APIStatusError("private provider detail", response=reply, body=None)
        return NS(choices=[NS(finish_reason="stop", message=NS(content='{"ok":true}'))],
                  usage=NS(total_tokens=50), reasoning_effort="low")

    monkeypatch.setenv("LLM_ROUTE_ORDER", "")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4.1-flash")
    monkeypatch.delenv("MODEL_SPEND_CAP_USD", raising=False)
    monkeypatch.setattr(model_budget, "OpenAI",
                        lambda **_: NS(chat=NS(completions=NS(create=create))))
    delays = []
    monkeypatch.setattr(model_budget.time, "sleep", delays.append)
    model = model_budget.BudgetedModel()
    if failure == "unauthorized":
        with pytest.raises(model_budget.PlannerResponseError, match="api_error"):
            model("probe")
        assert model.exhausted
    else:
        assert model("probe") == '{"ok":true}'
        assert model.planner_calls[-1]["reported_effort"] == "low"
    usage = model.usage()
    assert usage["requests"] == expected_requests == len(model.planner_calls)
    assert usage["unknown_usage_requests"] == expected_unknown
    assert usage["reserved_unknown_tokens"] == sum(row["retained_tokens"] for row in model.planner_calls)
    assert model.reserved == usage["tokens"] + usage["reserved_unknown_tokens"]
    assert len(delays) == expected_requests - 1
    assert all(options["reasoning_effort"] == "low" for options in request_options)
    assert "private provider detail" not in json.dumps(model.planner_calls)


def test_uncertain_timeout_reservation_can_prevent_retry(monkeypatch):
    from types import SimpleNamespace as NS

    import httpx
    from openai import APITimeoutError

    from data_analysis_agent import model_budget

    def timeout(**_):
        raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid")) from httpx.ReadTimeout("response timeout")

    monkeypatch.setenv("LLM_ROUTE_ORDER", "")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4.1-flash")
    monkeypatch.delenv("MODEL_SPEND_CAP_USD", raising=False)
    monkeypatch.setattr(model_budget, "OpenAI",
                        lambda **_: NS(chat=NS(completions=NS(create=timeout))))
    model = model_budget.BudgetedModel()
    model.max_tokens = 3000
    with pytest.raises(model_budget.PlannerResponseError, match="timeout"):
        model("probe", output_limit=1024)
    assert model.requests == 1
    assert model.usage()["tokens"] == 0
    assert model.usage()["reserved_unknown_tokens"] > 2000


@pytest.mark.parametrize(("failures", "expected_slots", "switches", "limit"), [
    ((), ("primary",), (None,), 10),
    ((401,), ("primary", "second"), (None, "authentication"), 10),
    ((402, 429), ("primary", "second", "flash-g"), (None, "quota", "rate_limited"), 10),
    (("dns",), ("primary", "official"), (None, "connection_error"), 10),
    ((503,), ("primary", "primary"), (None, None), 10),
    ((503, 503, 503), ("primary", "primary", "primary", "second"),
     (None, None, None, "provider_unavailable"), 10),
    (("low_mismatch",), ("primary", "second"), (None, "unexpected_reasoning_effort"), 10),
    ((401,), ("primary",), (None,), 1),
])
def test_budgeted_route_failover(monkeypatch, tmp_path, caplog, failures, expected_slots, switches, limit):
    from types import SimpleNamespace as NS

    import httpx
    from openai import APIConnectionError, APIStatusError

    from data_analysis_agent import model_budget
    from data_analysis_agent.deepseek_provider import DeepSeekNaturalLanguageModel

    env = tmp_path / ".env"
    env.write_text(
        "LLM_ROUTE_ORDER=vx_flash,vx_flash_g,official\n"
        "LLM_REASONING_EFFORT=low\n"
        "LLM_VX_FLASH_BASE_URL=https://vx.example.invalid/v1\n"
        "LLM_VX_FLASH_MODEL=deepseek-v4.1-flash\n"
        "LLM_VX_FLASH_API_KEY_1=private-primary\n"
        "LLM_VX_FLASH_API_KEY_2=private-second\n"
        "LLM_VX_FLASH_G_BASE_URL=https://VX.example.invalid/another/path\n"
        "LLM_VX_FLASH_G_MODEL=deepseek-v4.1-flash-g\n"
        "LLM_VX_FLASH_G_API_KEY=private-flash-g\n"
        "LLM_OFFICIAL_BASE_URL=https://official.example.invalid\n"
        "LLM_OFFICIAL_MODEL=deepseek-flash\n"
        "LLM_OFFICIAL_API_KEY=private-official\n"
    )
    for name in ("LLM_ROUTE_ORDER", "LLM_REASONING_EFFORT", "MODEL_SPEND_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_LLM_REQUESTS", str(limit))
    monkeypatch.setenv("MAX_LLM_TOTAL_TOKENS", "250000")
    monkeypatch.setattr(model_budget.time, "sleep", lambda _: None)
    caplog.set_level("INFO", logger=model_budget.__name__)
    attempted = []

    def openai(*, api_key, base_url, max_retries, **_):
        assert max_retries == 0

        def create(**options):
            assert options["reasoning_effort"] == "low"
            assert options["extra_body"] == {"thinking": {"type": "enabled"}}
            index = len(attempted)
            attempted.append((api_key.removeprefix("private-"), options))
            failure = failures[index] if index < len(failures) else None
            request = httpx.Request("POST", base_url)
            if failure == "dns":
                raise APIConnectionError(request=request) from httpx.ConnectError("DNS unavailable")
            if isinstance(failure, int):
                response = httpx.Response(failure, request=request)
                raise APIStatusError("private provider diagnostic", response=response, body=None)
            return NS(choices=[NS(finish_reason="stop", message=NS(content="{}"))],
                      usage=NS(total_tokens=9),
                      reasoning_effort="none" if failure == "low_mismatch" else "low")

        return NS(chat=NS(completions=NS(create=create)))

    monkeypatch.setattr(model_budget, "OpenAI", openai)
    if not failures:
        adapter = DeepSeekNaturalLanguageModel(env_path=env)
        model = adapter._routed
        assert adapter("probe") == "{}" and "response_format" not in attempted[0][1]
    else:
        model = model_budget.BudgetedModel(env_path=env)
        if limit == 1:
            with pytest.raises(RuntimeError, match="budget"):
                model("probe")
        else:
            assert model("probe") == "{}"
    assert tuple(key for key, _ in attempted) == expected_slots
    assert tuple(call["switch_reason"] for call in model.planner_calls) == switches
    assert model.requests == len(attempted) == len(model.usages)
    assert model.tokens == (18 if failures == ("low_mismatch",) else 0 if limit == 1 else 9)
    assert model.planner_calls[-1]["model"] == {
        "official": "deepseek-flash", "flash-g": "deepseek-v4.1-flash-g",
        "primary": "deepseek-v4.1-flash", "second": "deepseek-v4.1-flash",
    }[expected_slots[-1]]
    assert "private-" not in caplog.text + repr(model.routes) + json.dumps(model.planner_calls)


def test_business_action_correction_preserves_exact_tool_and_evidence(tmp_path):
    from data_analysis_agent.business_benchmark import score
    from data_analysis_agent.business_cases import DEV, build_retail

    case = next(item for item in DEV if item.id == "dev_wrong_timing")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    inputs = []

    def plan(prompt):
        data = json.loads(prompt.split("\n", 1)[1])
        inputs.append(data)
        if len(inputs) == 1:
            assert {"events", "timeline", "stop"} <= set(data["allowed_actions"])
            return {"action": {"tool": "events", "arguments": {"entity": "other"}},
                    "question": "Check a public incident."}
        if len(inputs) == 2:
            assert data["correction"]["error"] == "invalid_action_type"
            assert data["visible_evidence_ids"] == ["E1"]
            return {"action": {"tool": " EVENTS "}, "reason": "Check the public incident.", "evidence_ids": ["E1"]}
        if len(inputs) == 3:
            assert "events" not in data["allowed_actions"]
            return {"action": "timeline", "question": "Compare the first break with the incident date."}
        return {"action": "stop", "question": "The incident follows the initial drop.",
                "evidence_ids": ["E1", "E2", "E3"],
                "report": {"verdict": "insufficient", "claims": [
                    {"evidence": identifier, "fact": fact}
                    for identifier, fact in zip(("E1", "E3", "E2"), case.findings)
                ]}}

    state = investigate((database, database), case.goal, policy="llm_adaptive", budget=5,
                        planner=plan, events=events)
    assert len(inputs) == 4
    assert [item.action[0] for item in state.evidence] == ["headline", "events", "timeline"]
    rejected, accepted = state.trace[1:3]
    assert rejected["corrective_retry"] and rejected["error"] == "invalid_action_type"
    assert json.loads(rejected["raw_action"])["action"]["arguments"] == {"entity": "other"}
    assert json.loads(accepted["raw_action"])["action"] == {"tool": " EVENTS "}
    assert accepted["correction_of"] == "invalid_action_type"
    assert score(state, case)["correct"]


def test_business_action_correction_rejects_invented_evidence_and_tool_ids(tmp_path):
    from data_analysis_agent.business_cases import DEV, build_retail

    case = next(item for item in DEV if item.id == "dev_campaign")
    database = tmp_path / "observed.duckdb"
    events = build_retail(case, database)
    replies = iter((
        {"action": "events", "question": "Check the event.", "evidence_ids": ["E999"]},
        {"action": "search_events", "question": "Check the event.", "evidence_ids": ["E1"]},
    ))
    state = investigate((database, database), case.goal, policy="llm_adaptive", budget=5,
                        planner=lambda _: next(replies), events=events)
    assert state.model_error == "unavailable_action"
    assert [item.action[0] for item in state.evidence] == ["headline"]
    assert [(step["error"], step["corrective_retry"]) for step in state.trace
            if step["decision_source"] == "model_protocol"] == [
        ("invalid_evidence_id", True), ("unavailable_action", False),
    ]


@pytest.mark.parametrize(("replies", "expected_error", "expected_tools"), [
    (("length", "{", "[]", '{"actions":["timeline"],"reason":"Find the break."}',
      '{"verdict":"insufficient","claims":[]}'), None, 2),
    (("length", "length"), "truncated_output", 1),
    (("{", "{", "{"), "invalid_json", 1),
])
def test_business_recovery_bounds_and_does_not_spend_tool_slots(tmp_path, replies, expected_error, expected_tools):
    from data_analysis_agent.business_benchmark import _model_plan
    from data_analysis_agent.business_cases import DEV, build_retail
    from data_analysis_agent.model_budget import PlannerResponseError

    case = next(item for item in DEV if item.id == "dev_wrong_timing")
    database = tmp_path / "retail.duckdb"
    events = build_retail(case, database)
    attempts = []
    remaining = iter(replies)

    class Model:
        planner_calls = attempts

        @property
        def requests(self):
            return len(attempts)

        def __call__(self, prompt, *, output_limit, planning):
            assert (output_limit, planning) == (3072, True)
            reply = next(remaining)
            attempts.append({"total_tokens": 10})
            if reply == "length":
                attempts[-1]["response_status"] = "truncated_output"
                raise PlannerResponseError("truncated_output")
            return reply

    state = investigate((database, database), case.goal, policy="llm_frozen", budget=5,
                        planner=_model_plan(Model()), events=events)
    assert state.model_error == expected_error
    assert len(state.evidence) == expected_tools and state.remaining_budget == 5 - expected_tools
    assert len(attempts) == len(replies)
    assert "recovery_for" not in attempts[0] and "recovery_for" in attempts[1]
    if expected_error is None:
        assert [attempt.get("recovery_for") for attempt in attempts] == [
            None, "truncated_output", "invalid_json", "invalid_schema", None,
        ]


def test_business_budget_imports_distinct_history_once_and_survives_interruption(monkeypatch, tmp_path):
    from data_analysis_agent.business_benchmark import online_budget
    from data_analysis_agent.model_budget import BudgetedModel

    monkeypatch.setenv("LLM_ROUTE_ORDER", "")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4.1-flash")
    for key in ("MAX_LLM_REQUESTS", "MAX_LLM_TOTAL_TOKENS", "MODEL_SPEND_CAP_USD"):
        monkeypatch.delenv(key, raising=False)

    def model():
        return BudgetedModel(env_path=tmp_path / "absent", max_requests=80, max_tokens=260_000)

    def prior(name, requests, tokens):
        path = tmp_path / name
        path.write_text(json.dumps({"online_usage": {"requests": requests, "tokens": tokens,
                                                      "unknown_usage_requests": 0}}))
        return path

    dev, test = prior("dev.json", 19, 47_875), prior("test.json", 16, 52_392)
    journal = tmp_path / "usage.json"
    first = model()
    with online_budget(first, [dev, test, dev], journal_path=journal) as (_, _, mark, state):
        assert (first.max_requests, first.max_tokens) == (45, 159_733)
        mark("after_onset")
        first.requests += 1
        first.checkpoint_usage(8_000)
        journal_id = state["id"]
        # Simulate termination after the durable reservation, before a response.
    second = model()
    related = prior("continuation.json", 1, 0)
    saved = json.loads(related.read_text())
    saved["budget_journal_id"] = journal_id
    related.write_text(json.dumps(saved))
    with online_budget(second, [dev, test, related], journal_path=journal) as (_, seen, _, state):
        assert seen == {"after_onset"}
        assert (second.max_requests, second.max_tokens) == (44, 151_733)
        assert state["usage"]["unknown_usage_requests"] == 1
        assert state["usage"]["reserved_unknown_tokens"] == 8_000
