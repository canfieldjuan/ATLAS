"""Adapters bridging host LLM + Skill infrastructure into Content Ops.

The extracted Content Ops package declares two ports
(`extracted_content_pipeline.campaign_ports`):

- `LLMClient.complete(messages, *, max_tokens, temperature, metadata)
  -> LLMResponse` (async)
- `SkillStore.get_prompt(name) -> str | None` (sync)

The host already has equivalent infrastructure:

- `atlas_brain.services.LLMService.chat(messages, max_tokens, temperature)
  -> dict` (sync), resolved through the pipeline LLM router or the
  active registry fallback.
- `atlas_brain.skills.SkillRegistry.get(name) -> Optional[Skill]`,
  via `get_skill_registry()`.

This module provides thin adapters and factories so Content Ops
generators (`landing_page`, `campaign`, etc.) can use the host's LLM
and skill infrastructure without re-deriving the bridge each time.

This module is deliberately at `atlas_brain/_content_ops_infrastructure.py`
(not inside `atlas_brain/api/`) so the test harness can import
it without triggering the heavy router init chain (numpy /
torch / asyncpg).

See `plans/PR-Content-Ops-LLM-Skills-Infra-1.md` for the slice
contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from extracted_content_pipeline.campaign_ports import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    SkillStore,
)
from extracted_content_pipeline.campaign_llm_client import PipelineLLMClient


class _HostLLMClient:
    """Adapter implementing `LLMClient` over a host `LLMService`.

    Host `chat()` is synchronous and returns a `dict` carrying a
    `response` (or `content`) field plus optional `usage`. The
    extracted port is async and returns a structured
    `LLMResponse`. The bridge: run `chat()` in a worker thread
    via `asyncio.to_thread` and wrap the dict.

    `metadata` is dropped on the host side because the host's
    `chat()` doesn't accept arbitrary metadata today; the
    extracted package only uses the field for informational
    `asset_type` tags, so generation behavior is unaffected. If
    a future host LLM revision adds a metadata passthrough, this
    adapter is the single place to plumb it.
    """

    def __init__(self, host_llm: Any) -> None:
        self._host = host_llm

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        del metadata  # host chat() doesn't accept arbitrary metadata today.
        # The host's `chat()` reads `.role` / `.content`, and the cloud
        # backends (OpenRouter / Groq / Together / Ollama) additionally
        # read `.tool_calls` and `.tool_call_id` while building their
        # provider payloads. We pass a `SimpleNamespace` rather than
        # importing the host's `Message` dataclass because importing
        # `atlas_brain.services.protocols` triggers the full
        # `atlas_brain.services.__init__` chain (torch / ollama / etc.)
        # in dependency-light test envs. Including the optional
        # tool-call fields with `None` defaults keeps the duck-typed
        # shape compatible with all live backends without re-exposing
        # the heavy import.
        from types import SimpleNamespace

        host_messages = [
            SimpleNamespace(
                role=str(m.role or ""),
                content=str(m.content or ""),
                tool_calls=None,
                tool_call_id=None,
            )
            for m in messages
        ]
        result = await asyncio.to_thread(
            self._host.chat,
            host_messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        if not isinstance(result, Mapping):
            # Host returned a non-mapping (defensive); wrap as empty body.
            return LLMResponse(content=str(result or ""))

        content = str(result.get("response") or result.get("content") or "")
        model_name: str | None = None
        try:
            info = self._host.model_info
            if info is not None:
                model_name = str(getattr(info, "name", None) or info)
        except Exception:  # pragma: no cover - defensive
            model_name = None

        usage_value = result.get("usage")
        usage: dict[str, Any]
        if isinstance(usage_value, Mapping):
            usage = dict(usage_value)
        else:
            usage = {}

        return LLMResponse(
            content=content,
            model=model_name,
            usage=usage,
            raw=dict(result),
        )


class _HostSkillStore:
    """Adapter implementing `SkillStore` over a host-style `SkillRegistry`.

    Used as the fallback when the extracted-package registry isn't
    desired (e.g. tests that inject a stub registry with `.get(name)`
    returning a `Skill`-like object). Production callers get the
    extracted package's `LocalSkillRegistry` directly via
    ``build_content_ops_skill_store()``, which already implements
    ``get_prompt()`` and merges host overrides with the packaged
    skill defaults.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def get_prompt(self, name: str) -> str | None:
        skill = self._registry.get(name)
        if skill is None:
            return None
        content = getattr(skill, "content", None)
        if not isinstance(content, str):
            return None
        return content


def build_content_ops_llm_client(
    *,
    llm_registry: Any = None,
    pipeline_llm_resolver: Any = None,
) -> LLMClient | None:
    """Return a Content-Ops-shaped LLM client, or ``None`` if no
    host LLM is currently routable.

    Returning `None` (rather than raising) lets the
    services-bundle factory skip wiring LLM-needing generators
    when the host hasn't configured a model yet -- those slots
    stay `None` on the bundle and the executor surfaces
    `service_not_configured` per output, which the catalog
    endpoint exposes to the UI's Execute enable-state.

    The resolver / registry arguments are dependency-injection for tests:
    callers in dev environments without the host's full
    `atlas_brain.services` init chain (torch / ollama
    implementations) can pass stubs. Production callers omit
    both kwargs and the factory imports the canonical pipeline
    router and registry on demand.
    """

    if pipeline_llm_resolver is None and llm_registry is None:
        # Lazy import: production callers only. The pipeline router knows how to
        # resolve the configured OpenRouter Claude route even when no global
        # registry slot has been pre-activated.
        from atlas_brain.pipelines.llm import get_pipeline_llm

        pipeline_llm_resolver = get_pipeline_llm

    pipeline_kwargs = {
        "workload": "openrouter",
        "prefer_cloud": True,
        "try_openrouter": True,
        "auto_activate_ollama": False,
        "openrouter_model": None,
    }
    if pipeline_llm_resolver is not None:
        host_llm = pipeline_llm_resolver(**pipeline_kwargs)
        if host_llm is not None:
            return PipelineLLMClient(
                resolver=pipeline_llm_resolver,
                **pipeline_kwargs,
            )

    if llm_registry is None:
        # Lazy import: only production callers reach here. Tests
        # always inject a stub.
        from atlas_brain.services.registry import (
            llm_registry as default_registry,
        )

        llm_registry = default_registry

    host_llm = llm_registry.get_active()
    if host_llm is None:
        return None
    return PipelineLLMClient(
        resolver=lambda **_kwargs: host_llm,
        **pipeline_kwargs,
    )


_HOST_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def build_content_ops_skill_store(
    *,
    registry: Any = None,
) -> SkillStore:
    """Return a Content-Ops-shaped skill store.

    Production: returns the extracted package's
    `LocalSkillRegistry` via ``get_skill_registry`` with the
    host's `atlas_brain/skills/` tree as the override root.
    The extracted factory already implements host-first /
    packaged-fallback semantics, so:
    - Host-customized prompts (e.g. a host-specific
      `digest/blog_post_generation`) override the packaged
      default.
    - Packaged-only prompts (e.g. `digest/landing_page_generation`,
      `digest/report_generation`, `digest/sales_brief_generation`,
      `digest/b2b_campaign_reasoning_context`) resolve through
      the bundled defaults that ship inside the extracted
      package.

    Tests: pass a stub registry via the ``registry=`` kwarg.
    The stub must expose ``get(name)`` returning either a
    `Skill`-like object with `.content` or `None`. The factory
    wraps it in `_HostSkillStore` so tests can verify the
    adapter wiring without touching real markdown files.
    """

    if registry is not None:
        return _HostSkillStore(registry)

    from extracted_content_pipeline.skills.registry import (
        get_skill_registry as get_extracted_skill_registry,
    )

    return get_extracted_skill_registry(root=_HOST_SKILLS_DIR)


__all__ = [
    "build_content_ops_llm_client",
    "build_content_ops_skill_store",
]
