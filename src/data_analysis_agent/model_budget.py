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
        self.usages: list[int | None] = []
        self.exhausted = False
        self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL,
                             max_retries=0, timeout=55)

    def __call__(self, prompt: str, *, output_limit: int = 768) -> str:
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
        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.name, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, max_completion_tokens=output_limit,
                )
            except Exception as exc:
                # Provider messages may echo environment or request details.
                raise RuntimeError(f"Model call failed ({type(exc).__name__})") from None
            if response.usage is not None:
                tokens = response.usage.total_tokens
            choice = response.choices[0] if response.choices else None
            if choice is None or not choice.message.content:
                category = "length" if getattr(choice, "finish_reason", None) == "length" else "no content"
                raise RuntimeError(f"Model returned empty completion ({category})")
            success = True
            return choice.message.content
        finally:
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
        content = self(
            "You choose the next action for a two-window business metric investigation. "
            "Overall comparison E1 was measured for everyone. Use only the supplied observations. "
            "An action may group by zero to two available dimensions and optionally filter one "
            "different dimension to one value. At least a group or a filter is required. "
            "A filter plus no groups is a targeted verification. Same filters apply to both windows. "
            "Mix and within effects are contributions on the original population scale. "
            "Direction means material increase, decrease or stable. Never state causal conclusions.\n"
            'Return one compact JSON object with "decision":"investigate", '
            '"action":{"dimensions":["state"],"cohort":{}}, '
            '"hypothesis":{"segment":["SP"],"effect":"within","direction":"increase"}, '
            '"question":"short question the action answers", "evidence_ids":["E1"], '
            '"next_action":null, "revises":null. Segment values follow the action dimension order '
            '(use [] for a filtered total; use null to test the strongest observed group). '
            'Allowed effects: within or mix. Allowed directions: increase, decrease or stable. '
            'Example dimension/value are illustrative, obey the goal. '
            'When at least two actions remain, commit one distinct provisional next_action '
            'you would choose absent new evidence (same schema). Use null if there is no sensible '
            'follow-up. When new evidence refutes a tested '
            'hypothesis, cite that result and give its H-ID in revises if your next action changes. '
            'Do not repeat actions; set next_action null if none is available. '
            'To stop return {"decision":"stop","reason":"sufficient_evidence|insufficient_evidence|'
            'no_material_change","question":"what is resolved or missing","evidence_ids":["E1"]}. '
            "No material change requires checking for offsetting groups. Never invent evidence IDs.\n"
            + state, output_limit=4096
        ).strip()
        # Some compatible gateways wrap JSON in a Markdown fence.
        if fenced := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.S | re.I):
            content = fenced[1]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Planner response was not a JSON object") from None
