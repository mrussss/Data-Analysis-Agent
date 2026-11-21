"""Run-scoped request and token limits for the optional OpenAI-compatible benchmark."""
from __future__ import annotations

import json
import os
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
                             max_retries=0, timeout=30)

    def __call__(self, prompt: str) -> str:
        # The UTF-8 byte count conservatively bounds the supplied text for
        # byte-based tokenizers. Leave headroom for provider message framing.
        output_limit = 768
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
            response = self.client.chat.completions.create(
                model=self.name, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, max_completion_tokens=output_limit,
            )
            if response.usage is not None:
                tokens = response.usage.total_tokens
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty model response")
            success = True
            return content
        except Exception as exc:
            # Provider messages may echo environment or request details.
            raise RuntimeError(f"Model call failed ({type(exc).__name__})") from None
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
        parsed = json.loads(self(
            "Select one available statistical action for this investigation. Prefer a conditional "
            "split that challenges the strongest marginal hypothesis. Use only evidence supplied. "
            'Return JSON with an allowed "action" array. Optionally propose a falsifiable '
            'hypothesis using "segment" (one value per dimension), "effect" (within or mix), '
            'and "direction" (-1 or 1).\n'
            + state
        ))
        return parsed
