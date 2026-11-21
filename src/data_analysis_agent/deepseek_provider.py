"""Minimal DeepSeek provider adapters for structured and plain-text calls."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI

from data_analysis_agent.observability import (
    LLMCallObservation,
    current_observability_stage,
    observability_is_active,
    record_llm_call,
)


OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
PROJECT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _configured_client(env_path):
    load_dotenv(dotenv_path=env_path, override=False)
    if key := os.getenv("OPENAI_API_KEY"):
        if not (model := os.getenv("OPENAI_MODEL")):
            raise ValueError("OPENAI_MODEL is required with OPENAI_API_KEY.")
        return OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL), model
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENAI_API_KEY or DEEPSEEK_API_KEY is required.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL), DEEPSEEK_MODEL


class DeepSeekTextToSQLModel:
    """Call DeepSeek once and return its response content unchanged."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        env_path: str | Path = PROJECT_ENV_PATH,
    ) -> None:
        self._client, self._model = ((client, DEEPSEEK_MODEL) if client is not None
                                     else _configured_client(env_path))

    def __call__(self, prompt: str) -> str:
        timing = _start_provider_timing()
        response = None
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content:
                raise ValueError("DeepSeek returned empty response content.")
        except Exception:
            _record_provider_call(timing, model=self._model, status="error", response=response)
            raise
        _record_provider_call(timing, model=self._model, status="success", response=response)
        return content


class DeepSeekNaturalLanguageModel:
    """Call DeepSeek once and return plain-text response content unchanged."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        env_path: str | Path = PROJECT_ENV_PATH,
    ) -> None:
        self._client, self._model = ((client, DEEPSEEK_MODEL) if client is not None
                                     else _configured_client(env_path))

    def __call__(self, prompt: str) -> str:
        timing = _start_provider_timing()
        response = None
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("DeepSeek returned empty response content.")
        except Exception:
            _record_provider_call(timing, model=self._model, status="error", response=response)
            raise
        _record_provider_call(timing, model=self._model, status="success", response=response)
        return content


def _start_provider_timing() -> tuple[str, float] | None:
    stage = current_observability_stage()
    if not observability_is_active() or stage is None:
        return None
    return stage, time.perf_counter()


def _record_provider_call(
    timing: tuple[str, float] | None,
    *,
    model: str,
    status: Literal["success", "error"],
    response: object | None = None,
) -> None:
    if timing is None:
        return
    stage, started_at = timing
    usage = getattr(response, "usage", None)
    record_llm_call(
        LLMCallObservation(
            stage=stage,
            model=model,
            latency_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
            prompt_tokens=_usage_token(usage, "prompt_tokens"),
            completion_tokens=_usage_token(usage, "completion_tokens"),
            total_tokens=_usage_token(usage, "total_tokens"),
            status=status,
        )
    )


def _usage_token(usage: object | None, field_name: str) -> int | None:
    value = getattr(usage, field_name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
