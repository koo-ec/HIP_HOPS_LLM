"""Live provider clients for Mode B (``live_api``).

Design rules, enforced in code rather than in comments:

* **No moving aliases.**  ``configs/live_api.yaml`` must name immutable snapshot
  identifiers; a bare family name such as ``gpt-4o`` is rejected in strict mode.
* **No substitution.**  If a required snapshot is not offered by the provider the
  run stops with a precise error.  Another model is never used instead.
* **No implicit generation settings.**  System prompt, template, temperature,
  top-p, max output tokens, seed, ``n``, retry policy and timeout must all be
  present in the config; there are no defaults.
* **Errors stay errors.**  A failed call is recorded as an error, never silently
  converted into a task failure or success.  Task-conditional reliability and
  service-level reliability are reported separately.
* **Cleared context.**  Every i.i.d. benchmark task is issued as an independent
  request with no conversational history (paper Section 4.2).  The RQ7 memory
  experiment is the sole exception and uses an explicitly separate code path.
* **Only real responses are cached.**  The cache stores completed provider
  responses keyed by the full request; it can never contain a fabricated example.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "GenerationSettings",
    "ProviderError",
    "ModelUnavailableError",
    "MissingSettingError",
    "ResponseCache",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "build_client",
    "validate_snapshot_is_immutable",
    "RunLabel",
]

REQUIRED_GENERATION_KEYS = (
    "system_prompt",
    "prompt_template",
    "temperature",
    "top_p",
    "max_output_tokens",
    "n_generations",
    "seed",
    "timeout_seconds",
    "max_retries",
)


class ProviderError(RuntimeError):
    """A provider call failed.  Never swallowed into a benchmark outcome."""


class ModelUnavailableError(ProviderError):
    """The exact requested snapshot is not offered by the provider."""


class MissingSettingError(ValueError):
    """A generation setting the paper does not disclose was not supplied."""


class RunLabel(str):
    """Honest labelling of a live run.

    ``historical_exact`` may only be used when *every* generation setting has
    been recovered from an official source.  The HIP-LLM paper discloses none of
    them (no prompts, no temperature, no snapshot IDs, no seeds) and its
    repository contains no code, so any run performed today is
    ``contemporary_faithful_rerun``.
    """

    HISTORICAL_EXACT = "historical_exact"
    CONTEMPORARY = "contemporary_faithful_rerun"


def validate_snapshot_is_immutable(model_id: str, strict: bool = True) -> str:
    """Reject moving aliases such as ``gpt-4o`` or ``claude-sonnet-4-5``.

    An immutable snapshot carries a date suffix (``-YYYY-MM-DD`` for OpenAI,
    ``-YYYYMMDD`` for Anthropic).  In strict mode anything else raises.
    """
    import re

    if not model_id or not isinstance(model_id, str):
        raise ValueError("model identifier must be a non-empty string")
    immutable = bool(
        re.search(r"-\d{4}-\d{2}-\d{2}$", model_id) or re.search(r"-\d{8}$", model_id)
    )
    if not immutable and strict:
        raise MissingSettingError(
            f"model identifier {model_id!r} is a moving alias, not an immutable snapshot. "
            f"Strict mode requires a dated snapshot (e.g. 'gpt-4o-2024-08-06' or "
            f"'claude-sonnet-4-5-20250929'). Set live_api.strict_snapshots=false only if "
            f"you accept that the run cannot be pinned."
        )
    return model_id


@dataclass(frozen=True)
class GenerationSettings:
    """Every knob that affects a generation.  All are required."""

    system_prompt: str
    prompt_template: str
    temperature: float
    top_p: float
    max_output_tokens: int
    n_generations: int
    seed: int | None
    timeout_seconds: float
    max_retries: int
    stop_sequences: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GenerationSettings":
        missing = [k for k in REQUIRED_GENERATION_KEYS if k not in cfg or cfg[k] is None]
        if missing:
            raise MissingSettingError(
                f"generation settings missing required key(s): {missing}. The paper does not "
                f"disclose these; supply them explicitly in configs/live_api.yaml. A run using "
                f"guessed values must be labelled '{RunLabel.CONTEMPORARY}'."
            )
        if "{" not in str(cfg["prompt_template"]):
            raise MissingSettingError("prompt_template must contain at least one substitution field")
        return cls(
            system_prompt=str(cfg["system_prompt"]),
            prompt_template=str(cfg["prompt_template"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            max_output_tokens=int(cfg["max_output_tokens"]),
            n_generations=int(cfg["n_generations"]),
            seed=None if cfg["seed"] is None else int(cfg["seed"]),
            timeout_seconds=float(cfg["timeout_seconds"]),
            max_retries=int(cfg["max_retries"]),
            stop_sequences=tuple(cfg.get("stop_sequences", ()) or ()),
            extra=dict(cfg.get("extra", {}) or {}),
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


class ResponseCache:
    """On-disk cache of *completed provider responses*.

    Keyed by ``(provider, snapshot, generation fingerprint, rendered prompt,
    generation index)``.  Nothing else may be written here; a cache entry is
    always a record of a real HTTP round trip, together with its timestamp,
    resolved model id and token usage.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    @staticmethod
    def key(provider: str, snapshot: str, fingerprint: str, prompt: str, index: int) -> str:
        payload = json.dumps(
            [provider, snapshot, fingerprint, prompt, index], sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if p.is_file():
            self.hits += 1
            return json.loads(p.read_text(encoding="utf-8"))
        self.misses += 1
        return None

    def put(self, key: str, record: Mapping[str, Any]) -> None:
        if not record.get("raw_response"):
            raise ValueError("refusing to cache a record without a real provider response")
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(record), indent=2, default=str), encoding="utf-8")


@dataclass
class UsageLedger:
    """Running totals for the live-API diagnostics section of the report."""

    requests: int = 0
    failures: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "distinct_error_types": sorted({e["type"] for e in self.errors}),
        }


class LLMClient:
    """Base class holding the provider-independent policy."""

    provider = "abstract"

    def __init__(
        self,
        snapshot: str,
        settings: GenerationSettings,
        cache: ResponseCache | None = None,
        strict_snapshots: bool = True,
    ) -> None:
        self.snapshot = validate_snapshot_is_immutable(snapshot, strict=strict_snapshots)
        self.settings = settings
        self.cache = cache
        self.usage = UsageLedger()
        self._client = None

    # -- provider hooks ---------------------------------------------------- #
    def _connect(self) -> Any:
        raise NotImplementedError

    def _list_models(self) -> list[str]:
        raise NotImplementedError

    def _call(self, messages: Sequence[Mapping[str, str]], index: int) -> dict[str, Any]:
        raise NotImplementedError

    # -- public API -------------------------------------------------------- #
    def validate_availability(self) -> None:
        """Fail loudly if the exact snapshot is not offered.  Never substitutes."""
        available = self._list_models()
        if self.snapshot not in available:
            near = [m for m in available if m.split("-")[0] in self.snapshot][:8]
            raise ModelUnavailableError(
                f"{self.provider}: required snapshot {self.snapshot!r} is not available to this "
                f"account. Refusing to substitute another model. Closest available identifiers: "
                f"{near or 'none'}. Either obtain access to the exact snapshot or record this "
                f"experiment as blocked."
            )

    def generate(self, user_prompt: str, index: int = 0) -> dict[str, Any]:
        """One generation with **cleared context** (system + single user turn).

        Retries transient failures according to ``max_retries`` with exponential
        backoff and deterministic jitter; a permanently failing call raises
        :class:`ProviderError` rather than returning a fabricated outcome.
        """
        key = None
        if self.cache is not None:
            key = ResponseCache.key(
                self.provider, self.snapshot, self.settings.fingerprint(), user_prompt, index
            )
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        messages = [{"role": "user", "content": user_prompt}]
        last: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                self.usage.requests += 1
                record = self._call(messages, index)
                record.update(
                    {
                        "provider": self.provider,
                        "requested_snapshot": self.snapshot,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "generation_index": index,
                        "attempt": attempt,
                        "generation_fingerprint": self.settings.fingerprint(),
                        "context_cleared": True,
                    }
                )
                self.usage.input_tokens += int(record.get("input_tokens", 0) or 0)
                self.usage.output_tokens += int(record.get("output_tokens", 0) or 0)
                if self.cache is not None and key is not None:
                    self.cache.put(key, record)
                return record
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last = exc
                self.usage.retries += 1
                self.usage.errors.append(
                    {"type": type(exc).__name__, "message": str(exc)[:400], "attempt": attempt}
                )
                if attempt < self.settings.max_retries:
                    # deterministic backoff: no global RNG, no hidden state
                    time.sleep(min(2.0**attempt, 30.0) + 0.1 * ((index % 7) + 1))

        self.usage.failures += 1
        raise ProviderError(
            f"{self.provider}/{self.snapshot}: call failed after "
            f"{self.settings.max_retries + 1} attempt(s): {last}"
        ) from last

    def generate_in_session(
        self, history: list[dict[str, str]], user_prompt: str
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """RQ7 only: append to a **retained** conversation and return the new history.

        Deliberately separate from :meth:`generate` so that no i.i.d. experiment
        can accidentally inherit context.  Caching is disabled here because the
        response depends on the whole accumulated history.
        """
        messages = [*history, {"role": "user", "content": user_prompt}]
        self.usage.requests += 1
        record = self._call(messages, index=0)
        record.update(
            {
                "provider": self.provider,
                "requested_snapshot": self.snapshot,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "context_cleared": False,
                "history_turns": len(messages),
                "serialized_request_bytes": len(json.dumps(messages).encode("utf-8")),
            }
        )
        self.usage.input_tokens += int(record.get("input_tokens", 0) or 0)
        self.usage.output_tokens += int(record.get("output_tokens", 0) or 0)
        new_history = [*messages, {"role": "assistant", "content": record["text"]}]
        return record, new_history


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions client (GPT-4o family)."""

    provider = "openai"

    def _connect(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ProviderError(
                    "OPENAI_API_KEY is not set. live_api mode fails rather than skipping."
                )
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProviderError("the 'openai' package is not installed") from exc
            self._client = OpenAI(timeout=self.settings.timeout_seconds)
        return self._client

    def _list_models(self) -> list[str]:
        return sorted(m.id for m in self._connect().models.list().data)

    def _call(self, messages: Sequence[Mapping[str, str]], index: int) -> dict[str, Any]:
        client = self._connect()
        kwargs: dict[str, Any] = {
            "model": self.snapshot,
            "messages": [{"role": "system", "content": self.settings.system_prompt}, *messages],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_output_tokens,
        }
        if self.settings.seed is not None:
            kwargs["seed"] = self.settings.seed
        if self.settings.stop_sequences:
            kwargs["stop"] = list(self.settings.stop_sequences)
        resp = client.chat.completions.create(**kwargs)
        return {
            "text": resp.choices[0].message.content or "",
            "resolved_model": resp.model,
            "finish_reason": resp.choices[0].finish_reason,
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "raw_request": {k: v for k, v in kwargs.items() if k != "messages"},
            "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        }


class AnthropicClient(LLMClient):
    """Anthropic Messages API client (Claude Sonnet / Haiku families)."""

    provider = "anthropic"

    def _connect(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError(
                    "ANTHROPIC_API_KEY is not set. live_api mode fails rather than skipping."
                )
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProviderError("the 'anthropic' package is not installed") from exc
            self._client = anthropic.Anthropic(timeout=self.settings.timeout_seconds)
        return self._client

    def _list_models(self) -> list[str]:
        return sorted(m.id for m in self._connect().models.list(limit=1000).data)

    def _call(self, messages: Sequence[Mapping[str, str]], index: int) -> dict[str, Any]:
        client = self._connect()
        kwargs: dict[str, Any] = {
            "model": self.snapshot,
            "system": self.settings.system_prompt,
            "messages": list(messages),
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_output_tokens,
        }
        if self.settings.stop_sequences:
            kwargs["stop_sequences"] = list(self.settings.stop_sequences)
        resp = client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return {
            "text": text,
            "resolved_model": resp.model,
            "finish_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "system_fingerprint": None,
            "raw_request": {k: v for k, v in kwargs.items() if k != "messages"},
            "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        }


def build_client(
    provider: str,
    snapshot: str,
    settings: GenerationSettings,
    cache: ResponseCache | None = None,
    strict_snapshots: bool = True,
) -> LLMClient:
    """Factory.  Unknown providers raise -- there is no generic fallback client."""
    providers = {"openai": OpenAIClient, "anthropic": AnthropicClient}
    if provider not in providers:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(providers)}")
    return providers[provider](snapshot, settings, cache=cache, strict_snapshots=strict_snapshots)
