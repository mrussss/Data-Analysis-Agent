"""Run-scoped request and token limits for the optional OpenAI-compatible benchmark."""
from __future__ import annotations

import json
import logging
import re
import socket
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, DefaultHttpxClient, OpenAI

from .deepseek_provider import PROJECT_ENV_PATH, _model_settings, _ordered_routes, _usage_token, low_reasoning
from .observability import LLMCallObservation, current_observability_stage, record_llm_call


_LOGGER = logging.getLogger(__name__)


def _retry_interval(value: str | None, attempt: int) -> float | None:
    if value is None:
        return 0.5 * 2 ** attempt
    try:
        requested = float(value)
    except ValueError:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        try:
            requested = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.5 * 2 ** attempt
    return max(0.5 * 2 ** attempt, requested) if requested <= 10 else None


class PlannerResponseError(ValueError):
    def __init__(self, kind: str, *, fallback: str | None = None, skip_domain: bool = False):
        self.kind = kind
        self.fallback = fallback
        self.skip_domain = skip_domain
        super().__init__(kind)


def parse_model_json(content: str) -> dict:
    content = content.strip()
    if fenced := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.S | re.I):
        content = fenced[1]
    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        raise PlannerResponseError("invalid_json") from None
    if not isinstance(decision, dict):
        raise PlannerResponseError("invalid_schema")
    return decision


class BudgetedModel:
    def __init__(self, *, env_path=PROJECT_ENV_PATH, max_requests=60, max_tokens=250_000):
        config = _model_settings(env_path)
        self.routes = _ordered_routes(config)
        # The existing run metadata keeps the primary model. Each attempt
        # independently records the model that was used.
        self.name = self.routes[0].model
        if config.get("MODEL_SPEND_CAP_USD"):
            raise ValueError("A USD cap requires provider-reported spend or verified pricing")
        self.max_requests = int(config.get("MAX_LLM_REQUESTS") or max_requests)
        self.max_tokens = int(config.get("MAX_LLM_TOTAL_TOKENS") or max_tokens)
        self.requests = self.tokens = self.reserved = 0
        self.uncertain_tokens = 0
        self.unbounded_usage = False
        self.planner_calls: list[dict] = []
        self.usages: list[int | None] = []
        self.exhausted = False
        self._clients = {}
        self.checkpoint_usage = None

    def __call__(self, prompt: str, *, output_limit: int | None = None, planning: bool = False,
                 json_response: bool = True) -> str:
        blocked_domains = set()
        prior_failure = None
        first_attempt = len(self.planner_calls)
        for index, route in enumerate(self.routes):
            if route.domain in blocked_domains:
                continue
            remaining = tuple(item for item in self.routes[index + 1:] if item.domain not in blocked_domains)
            try:
                return self._call_route(route, prompt, output_limit=output_limit, planning=planning,
                                        json_response=json_response, prior_failure=prior_failure,
                                        has_fallback=bool(remaining),
                                        other_domain=any(item.domain != route.domain for item in remaining),
                                        first_attempt=first_attempt)
            except PlannerResponseError as exc:
                if not exc.fallback or self.exhausted:
                    raise
                prior_failure = exc.fallback
                if exc.skip_domain:
                    blocked_domains.add(route.domain)
                if not any(item.domain not in blocked_domains for item in remaining):
                    self.exhausted = True
                    raise
        raise RuntimeError("No model route available")

    def _call_route(self, route, prompt, *, output_limit, planning, json_response, prior_failure,
                    has_fallback, other_domain, first_attempt):
        mode = low_reasoning(route.model)
        if output_limit is None:
            output_limit = 3072 if mode else 768
        # Reserve message framing and the full output limit. An attempt with
        # potentially billable unknown usage keeps its reservation.
        # The business gateway has billed completion usage above max_tokens.
        # Keep extra headroom for an interrupted or unreported online attempt.
        reserved_output = output_limit * (2 if self.checkpoint_usage else 1)
        upper = len(prompt.encode("utf-8")) + 1024 + reserved_output
        cap = {"max_tokens": output_limit, **mode} if mode else {"max_completion_tokens": output_limit}
        if json_response:
            cap["response_format"] = {"type": "json_object"}
        if planning:
            cap["timeout"] = 85
        for attempt in range(3):
            if self.exhausted or self.requests >= self.max_requests or self.reserved + upper > self.max_tokens:
                self.exhausted = True
                raise RuntimeError("Model request/token budget reached")
            self.requests += 1
            self.reserved += upper
            started = time.perf_counter()
            tokens = None
            success = False
            delay = None
            entry = {"attempt": attempt + 1, "output_limit": output_limit, "token_reservation": upper,
                     "route": route.name, "model": route.model, "key_slot": route.key_slot,
                     "switch_reason": prior_failure if attempt == 0 else None,
                     "response_format": "json_object" if json_response else "text",
                     "reasoning_effort": mode.get("reasoning_effort"),
                     "thinking_enabled_requested": bool(mode), "timeout_seconds": cap.get("timeout", 55)}
            self.planner_calls.append(entry)
            if self.checkpoint_usage:
                self.checkpoint_usage(upper)
            try:
                client = self._clients.get(route)
                if client is None:
                    client = OpenAI(api_key=route.api_key, base_url=route.base_url, max_retries=0,
                                    timeout=55, http_client=DefaultHttpxClient(follow_redirects=False))
                    self._clients[route] = client
                response = client.chat.completions.create(
                    model=route.model, messages=[{"role": "user", "content": prompt}], **cap,
                )
                usage = response.usage
                tokens = _usage_token(usage, "total_tokens")
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice else None
                content = (message.content or "") if message else ""
                finish = getattr(choice, "finish_reason", None)
                reasoning = getattr(message, "reasoning_content", None) or ""
                reported_model = getattr(response, "model", None)
                if isinstance(reported_model, str) and re.fullmatch(
                    r"(?i)(?:[a-z0-9_.-]+/)?deepseek[a-z0-9_.-]{0,100}", reported_model
                ):
                    entry["reported_model"] = reported_model
                billed = getattr(usage, "cost_usd", None)
                if isinstance(billed, (int, float)) and not isinstance(billed, bool) and billed >= 0:
                    entry["provider_cost_usd"] = billed
                reported_effort = next((str(value).lower() for source in (response, message, usage)
                                        if (value := getattr(source, "reasoning_effort", None)) is not None), None)
                # Persist only recognized tracing values, not arbitrary gateway output.
                if reported_effort not in {None, "none", "low", "medium", "high", "max", "50"}:
                    reported_effort = "unrecognized"
                completion_tokens = _usage_token(usage, "completion_tokens")
                entry.update(finish_reason=finish, content_chars=len(content), blank=not content.strip(),
                             completion_over_requested_limit=(completion_tokens is not None
                                                              and completion_tokens > output_limit),
                             reasoning_chars=len(reasoning), reported_effort=reported_effort,
                             reasoning_tokens=_usage_token(getattr(usage, "completion_tokens_details", None),
                                                           "reasoning_tokens"),
                             content_equals_reasoning=bool(reasoning) and content == reasoning,
                             prompt_tokens=_usage_token(usage, "prompt_tokens"),
                             completion_tokens=completion_tokens)
                if tokens is None:
                    self.unbounded_usage = self.exhausted = True
                    raise PlannerResponseError("unreported_provider_usage")
                if mode and reported_effort not in {None, "low", "50", "unrecognized"}:
                    self.exhausted = not has_fallback
                    raise PlannerResponseError("unexpected_reasoning_effort",
                                               fallback="unexpected_reasoning_effort" if has_fallback else None)
                if finish == "length":
                    raise PlannerResponseError("truncated_output")
                if not content.strip():
                    raise PlannerResponseError("empty_output")
                entry["response_status"] = "content"
                success = True
                return content
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                # HTTPX ConnectError and ConnectTimeout fail before sending the
                # request body. A read timeout or other I/O error can be billed.
                cause = exc.__cause__
                unsent = isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError,
                                            socket.gaierror, ConnectionRefusedError))
                domain_error = unsent or (type(exc) is APIConnectionError and cause is None)
                if unsent:
                    tokens = 0
                temporary = (isinstance(exc, APIConnectionError) or status in {408, 429}
                             or (status is not None and 500 <= status < 600))
                code = getattr(exc, "code", None)
                quota = status == 402 or code in ("insufficient_quota", "quota_exceeded", "billing_hard_limit_reached")
                auth = status in {401, 403}
                mode_rejected = status in {400, 422} and getattr(exc, "param", None) in (
                    "thinking", "thinking.type", "reasoning_effort",
                )
                reason = ("connection_error" if domain_error else "quota" if quota else "authentication" if auth
                          else "rate_limited" if status == 429 else "low_rejected" if mode_rejected
                          else "model_unavailable" if status == 404 else
                          "timeout" if isinstance(exc, APITimeoutError) or status == 408 else "provider_unavailable")
                switch_now = (has_fallback and (auth or quota or status == 429)) or (domain_error and other_domain)
                retry_after = exc.response.headers.get("Retry-After") if isinstance(exc, APIStatusError) else None
                interval = _retry_interval(retry_after, attempt)
                retries = sum(item["attempt"] > 1 for item in self.planner_calls[first_attempt:])
                if temporary and not quota and not switch_now and retries < 2 and interval is not None:
                    retained = upper if tokens is None else tokens
                    if self.requests < self.max_requests and self.reserved + retained <= self.max_tokens:
                        delay = interval
                kind = ("timeout" if isinstance(exc, APITimeoutError) else "connection_error"
                        if isinstance(exc, APIConnectionError) else "api_error")
                entry.update(error_class=type(exc).__name__, http_status=status,
                             transport_cause=type(exc.__cause__).__name__ if isinstance(exc, APIConnectionError) else None,
                             response_status=kind, retry_delay_seconds=delay,
                             failure_reason=reason, request_body_unsent=unsent)
                if delay is None:
                    fallback = has_fallback and (temporary or auth or quota or mode_rejected or status == 404)
                    self.exhausted |= not fallback and (temporary or (status is not None and 400 <= status < 500))
                    raise PlannerResponseError(kind, fallback=reason if fallback else None, skip_domain=domain_error) from None
            except PlannerResponseError as exc:
                entry["response_status"] = exc.kind
                raise
            except Exception as exc:
                entry.update(error_class=type(exc).__name__, response_status="provider_error")
                self.exhausted = True
                self.unbounded_usage |= tokens is None
                raise PlannerResponseError("provider_error") from None
            finally:
                elapsed = (time.perf_counter() - started) * 1000
                entry.update(total_tokens=tokens, latency_ms=round(elapsed, 2),
                             retained_tokens=upper if tokens is None else 0)
                self.usages.append(tokens)
                if tokens is None:
                    self.uncertain_tokens += upper
                else:
                    self.tokens += tokens
                    self.reserved += tokens - upper
                self.exhausted |= self.reserved >= self.max_tokens
                if self.checkpoint_usage:
                    self.checkpoint_usage()
                _LOGGER.info("LLM attempt route=%s model=%s key_slot=%s switch=%s outcome=%s tokens=%s",
                             route.name, route.model, route.key_slot, entry["switch_reason"],
                             entry.get("failure_reason", entry.get("response_status")), tokens)
                record_llm_call(LLMCallObservation(
                    current_observability_stage() or "model_planning", route.model, elapsed,
                    entry.get("prompt_tokens"), entry.get("completion_tokens"), tokens,
                    "success" if success else "error",
                ))
            if delay is not None:
                time.sleep(delay)

    def usage(self) -> dict:
        return {"requests": self.requests, "tokens": self.tokens,
                "unknown_usage_requests": self.usages.count(None),
                "reserved_unknown_tokens": self.uncertain_tokens, "unbounded_usage": self.unbounded_usage}

    def plan(self, state: str) -> dict:
        first = len(self.planner_calls)
        try:
            content = self(
                'Choose one decision for a read-only metric investigation. Output JSON only.\n'
                'Continue example: {"action":{"dimensions":["weekday"],"cohort":{"state":"SP"}},'
                '"question":"Is the SP shift concentrated by weekday?","evidence_ids":["E1"]}.\n'
                'Stop example: {"stop":"sufficient_evidence","question":"Which evidence resolves the question?",'
                '"evidence_ids":["E1","E2"]}. Other stop reasons: insufficient_evidence, no_material_change.\n'
                'Dimensions are grouping keys (at most two). Cohort is an optional filter (at most one '
                'different key); a cohort with no dimensions verifies its total. Do not repeat tested actions. '
                'Use only visible evidence IDs. Explain the next question briefly; never assert causes. '
                'For a testable prediction, optionally add "expect":"increase", "decrease", or "stable" '
                'for the direction of the largest measurable within-group contribution. '
                'Do not choose a segment or calculate numbers. If useful, record a planned follow-up in '
                '"next_action" with the same action shape, before knowing the result. '
                'If you change that plan after a refutation, cite the new evidence and set "revises" to its hypothesis ID. '
                'Do not stop for no change without checking for offsets.\n' + state,
                output_limit=3072 if self.name.lower().startswith("deepseek") else 768, planning=True,
            ).strip()
            decision = parse_model_json(content)
        except (PlannerResponseError, RuntimeError) as exc:
            if len(self.planner_calls) > first:
                self.planner_calls[-1]["response_status"] = getattr(exc, "kind", "budget_limit")
            raise
        self.planner_calls[-1]["response_status"] = "json"
        return decision
