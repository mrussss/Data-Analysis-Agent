"""Run-scoped request and token limits for the optional OpenAI-compatible benchmark."""
from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from .deepseek_provider import PROJECT_ENV_PATH, OPENAI_DEFAULT_URL
from .observability import LLMCallObservation, current_observability_stage, record_llm_call


class PlannerResponseError(ValueError):
    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


class BudgetedModel:
    def __init__(self):
        load_dotenv(PROJECT_ENV_PATH)
        key, self.name = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")
        if not key or not self.name:
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required for --online")
        if os.getenv("MODEL_SPEND_CAP_USD"):
            raise ValueError("A USD cap requires provider-reported spend or verified pricing")
        self.max_requests = int(os.getenv("MAX_LLM_REQUESTS", "60"))
        self.max_tokens = int(os.getenv("MAX_LLM_TOTAL_TOKENS", "250000"))
        self.requests = self.tokens = self.reserved = 0
        self.planner_calls: list[dict] = []
        self.usages: list[int | None] = []
        self.exhausted = False
        self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL,
                             max_retries=0, timeout=55)

    def __call__(self, prompt: str, *, output_limit: int = 768, planning: bool = False) -> str:
        # The UTF-8 byte count conservatively bounds the supplied text for
        # byte-based tokenizers. Leave headroom for provider message framing.
        upper = len(prompt.encode("utf-8")) + 1024 + output_limit
        if self.exhausted or self.requests >= self.max_requests or self.reserved + upper > self.max_tokens:
            self.exhausted = True
            raise RuntimeError("Model request/token budget reached")
        self.requests += 1
        self.reserved += upper
        started = time.perf_counter()
        tokens = None
        success = False
        deepseek = planning and self.name.lower().startswith("deepseek")
        cap = {"max_tokens": output_limit, "reasoning_effort": "none",
               "extra_body": {"thinking": {"type": "disabled"}}} if deepseek else {
            "max_completion_tokens": output_limit,
        }
        if planning:
            cap["timeout"] = 85
        entry = {"output_limit": output_limit, "thinking_disabled_requested": deepseek,
                 "response_format": "json_object", "reasoning_effort": cap.get("reasoning_effort"),
                 "timeout_seconds": cap["timeout"]} if planning else None
        if entry is not None:
            self.planner_calls.append(entry)
        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.name, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, **cap,
                )
            except Exception as exc:
                # Provider messages may echo environment or request details.
                if entry is not None:
                    code = type(exc).__name__
                    entry.update(error_class=code, http_status=getattr(exc, "status_code", None))
                    raise PlannerResponseError("timeout" if code == "APITimeoutError" else "api_error") from None
                raise RuntimeError(f"Model call failed ({type(exc).__name__})") from None
            if response.usage is not None:
                tokens = response.usage.total_tokens
            choice = response.choices[0] if response.choices else None
            if entry is not None:
                message = choice.message if choice else None
                content = message.content or "" if message else ""
                finish = getattr(choice, "finish_reason", None)
                reasoning = getattr(message, "reasoning_content", None) or ""
                entry.update(finish_reason=finish, content_chars=len(content),
                             blank=not content.strip(), reasoning_chars=len(reasoning),
                             content_equals_reasoning=bool(reasoning) and content == reasoning,
                             prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                             completion_tokens=getattr(response.usage, "completion_tokens", None))
                if finish == "length":
                    raise PlannerResponseError("truncated_output")
                if not content.strip():
                    raise PlannerResponseError("empty_output")
            if choice is None or not choice.message.content:
                category = "length" if getattr(choice, "finish_reason", None) == "length" else "no content"
                raise RuntimeError(f"Model returned empty completion ({category})")
            success = True
            return choice.message.content
        finally:
            if entry is not None:
                entry.update(total_tokens=tokens, latency_ms=round((time.perf_counter() - started) * 1000, 2))
            self.usages.append(tokens)
            if tokens is None:
                self.exhausted = True
            else:
                self.tokens += tokens
                self.reserved += tokens - upper
                self.exhausted |= self.reserved >= self.max_tokens
            record_llm_call(LLMCallObservation(
                current_observability_stage() or "model_planning", self.name,
                (time.perf_counter() - started) * 1000, None, None, tokens,
                "success" if success else "error",
            ))

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
            if fenced := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.S | re.I):
                content = fenced[1]
            try:
                decision = json.loads(content)
            except json.JSONDecodeError:
                raise PlannerResponseError("invalid_json") from None
        except (PlannerResponseError, RuntimeError) as exc:
            if len(self.planner_calls) > first:
                self.planner_calls[-1]["response_status"] = getattr(exc, "kind", "budget_limit")
            raise
        self.planner_calls[-1]["response_status"] = "json"
        return decision
