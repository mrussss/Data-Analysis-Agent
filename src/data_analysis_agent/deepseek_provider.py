"""Minimal DeepSeek provider adapters for structured and plain-text calls."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
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


def low_reasoning(model: str) -> dict:
    if model.rsplit("/", 1)[-1].lower().startswith("deepseek"):
        return {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}}
    return {}


def _adapter_options(model: str) -> dict:
    options = low_reasoning(model)
    return {"max_tokens": 3072, **options} if options else {}


@dataclass(frozen=True)
class ProviderRoute:
    name: str
    model: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    key_slot: str = "default"

    @property
    def domain(self) -> str:
        return (urlsplit(self.base_url).hostname or "").lower().rstrip(".")


def _model_settings(env_path: str | Path) -> dict[str, str | None]:
    return {**dotenv_values(env_path), **os.environ}


def _ordered_routes(config: dict[str, str | None]) -> tuple[ProviderRoute, ...]:
    order = (config.get("LLM_ROUTE_ORDER") or "").strip()
    if not order:
        model, key = config.get("OPENAI_MODEL"), config.get("OPENAI_API_KEY")
        if not model or not key:
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required for --online")
        return (ProviderRoute("openai_legacy", model, config.get("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL, key),)

    if (config.get("LLM_REASONING_EFFORT") or "low").strip().lower() != "low":
        raise ValueError("LLM_REASONING_EFFORT must be low for configured routes")
    routes = []
    names = [part.strip().lower() for part in order.split(",")]
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names) or len(set(names)) != len(names):
        raise ValueError("LLM_ROUTE_ORDER must contain unique route names")
    for name in names:
        prefix = f"LLM_{name.upper()}_"
        model, base_url = config.get(prefix + "MODEL"), config.get(prefix + "BASE_URL")
        if not model or not low_reasoning(model) or not base_url:
            raise ValueError(f"Route {name} requires a DeepSeek model and a base URL")
        try:
            url = urlsplit(base_url)
            valid_url = url.scheme in {"http", "https"} and url.hostname and not (url.username or url.password)
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(f"Invalid base URL for route {name}")
        stem = prefix + "API_KEY"
        numbered = sorted((int(var[len(stem) + 1:]), var) for var in config
                          if re.fullmatch(re.escape(stem) + r"_[1-9][0-9]*", var))
        keys = [(str(index), config[var]) for index, var in numbered]
        keys.append(("default", config.get(stem)))
        seen = set()
        for slot, key in keys:
            if key and key.strip() and key not in seen:
                routes.append(ProviderRoute(name, model, base_url, key, slot))
                seen.add(key)
        if not seen:
            raise ValueError(f"Route {name} requires an API key")
    return tuple(routes)


def _configured_client(config):
    if key := config.get("OPENAI_API_KEY"):
        if not (model := config.get("OPENAI_MODEL")):
            raise ValueError("OPENAI_MODEL is required with OPENAI_API_KEY.")
        return OpenAI(api_key=key, base_url=config.get("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL, max_retries=0), model
    key = config.get("DEEPSEEK_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENAI_API_KEY or DEEPSEEK_API_KEY is required.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, max_retries=0), DEEPSEEK_MODEL


def _adapter_client(client, env_path):
    if client is not None:
        return client, DEEPSEEK_MODEL, None
    config = _model_settings(env_path)
    if (config.get("LLM_ROUTE_ORDER") or "").strip():
        from .model_budget import BudgetedModel

        routed = BudgetedModel(env_path=env_path)
        return None, routed.name, routed
    client, model = _configured_client(config)
    return client, model, None


class DeepSeekTextToSQLModel:
    """Call DeepSeek once and return its response content unchanged."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        env_path: str | Path = PROJECT_ENV_PATH,
    ) -> None:
        self._client, self._model, self._routed = _adapter_client(client, env_path)

    def __call__(self, prompt: str) -> str:
        if self._routed:
            return self._routed(prompt)
        timing = _start_provider_timing()
        response = None
        try:
            response = self._client.chat.completions.create(
                model=self._model, **_adapter_options(self._model),
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
        self._client, self._model, self._routed = _adapter_client(client, env_path)

    def __call__(self, prompt: str) -> str:
        if self._routed:
            return self._routed(prompt, json_response=False)
        timing = _start_provider_timing()
        response = None
        try:
            response = self._client.chat.completions.create(
                model=self._model, **_adapter_options(self._model),
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
