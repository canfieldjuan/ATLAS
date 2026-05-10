# PR: host-side LLMClient + SkillStore infrastructure for Content Ops

## Why this slice exists

PR #452 wired `signal_extraction` into the Execute route's
`execution_services_provider`. That generator has zero
dependencies. The remaining 5 generators (`campaign`,
`blog_post`, `report`, `landing_page`, `sales_brief`) all need
two shared infrastructure ports:

- `LLMClient` -- async `complete(messages, *, max_tokens,
  temperature, metadata=None) -> LLMResponse`
  (`extracted_content_pipeline/campaign_ports.py`).
- `SkillStore` -- sync `get_prompt(name) -> str | None`
  (same module).

The host already has equivalent infrastructure:

- `atlas_brain.services.LLMService` -- sync `chat(messages,
  max_tokens, temperature) -> dict` accessed through
  `llm_registry.get_active()`.
- `atlas_brain.skills.SkillRegistry` -- `get(name) ->
  Optional[Skill]` via `get_skill_registry()`.

Without an adapter, none of the 5 remaining generators can land.
With one, they each become a small follow-up slice that just
plugs the LLM + skills factory output into a service constructor.

This PR is the shared infrastructure. No service wired here.

## Scope (this PR)

One new module and a test file. The host's `__init__.py` is
NOT touched -- that's the next slice's job (when an actual
LLM-needing service ships).

### Files touched

1. `atlas_brain/_content_ops_infrastructure.py` (new):
   - `_HostLLMClient` adapter that wraps a host `LLMService`
     and implements the extracted package's `LLMClient`
     protocol (async `complete()`).
   - `_HostSkillStore` adapter that wraps `SkillRegistry` and
     implements `SkillStore.get_prompt()`.
   - `build_content_ops_llm_client()` -- returns an adapter
     wrapping `llm_registry.get_active()`, or `None` if no
     LLM is active. Callers (the next slice's services-bundle
     factory) decide whether to skip wiring LLM-needing
     generators.
   - `build_content_ops_skill_store()` -- returns an adapter
     wrapping `get_skill_registry()`. Always available (skills
     load lazily from disk on first access).

2. `tests/test_atlas_content_ops_infrastructure.py` (new):
   - `test_host_skill_store_returns_existing_skill_content`
     -- `get_prompt("digest/blog_post_generation")` returns
     the markdown body. Pins that the skill name space the
     extracted services expect resolves correctly through the
     host registry.
   - `test_host_skill_store_returns_none_for_missing_skill`
     -- canary for the "skill not found" branch.
   - `test_host_llm_client_translates_messages_and_response`
     -- a fake host LLMService receives the translated messages
     and returns a dict; the adapter wraps it into an
     LLMResponse with the right shape.
   - `test_host_llm_client_handles_content_field_alias` --
     adapter accepts `{"content": ...}` aliases for backends
     that return the field under that key instead of
     `"response"`.
   - `test_build_content_ops_llm_client_returns_none_when_no_active`
     -- factory short-circuits cleanly when `llm_registry`
     has no active service. Uses the dependency-injection
     kwarg to avoid importing the host's full services chain.
   - `test_build_content_ops_llm_client_wraps_active_service`
     -- factory returns a `_HostLLMClient` adapter when the
     registry has an active service.
   - `test_build_content_ops_skill_store_uses_injected_registry`
     -- factory accepts an injected registry stub for tests
     and delegates lookups through it.

   Total: 7 tests.

3. `plans/PR-Content-Ops-LLM-Skills-Infra-1.md` (this file).

### What's NOT in this slice

- Any service wired into `execution_services_provider`. The
  bundle factory in `atlas_brain/_content_ops_services.py`
  stays unchanged. Next slice (E2) wires `landing_page` (which
  has the smallest dependency footprint among LLM-needing
  services -- no `IntelligenceRepository`).
- Postgres-backed repositories
  (`PostgresLandingPageRepository`, etc.). Those are extracted
  package responsibilities; the bundle factory imports them
  from there when needed.
- A reasoning-context provider. PR #402 wired the route-level
  seam; the bundle's `with_reasoning_context()` derivation
  handles per-request rebinding. This slice doesn't touch
  reasoning.

## Mechanism

The LLM adapter bridges sync host -> async extracted via
`asyncio.to_thread`:

```python
class _HostLLMClient:
    def __init__(self, host_llm: LLMService) -> None:
        self._host = host_llm

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        del metadata  # host's chat() doesn't accept arbitrary metadata
        host_messages = [Message(role=m.role, content=m.content) for m in messages]
        result = await asyncio.to_thread(
            self._host.chat,
            host_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            content=str(result.get("response") or result.get("content") or ""),
            model=str(self._host.model_info.name) if self._host.model_info else None,
            usage=dict(result.get("usage") or {}),
            raw=result,
        )
```

The skill adapter is a thin delegator:

```python
class _HostSkillStore:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def get_prompt(self, name: str) -> str | None:
        skill = self._registry.get(name)
        return skill.content if skill is not None else None
```

Factories return adapter instances on demand (no module-level
singletons -- the host's `llm_registry.get_active()` can change
when the operator hot-swaps models, so the factory always reads
the current active instance).

Both factories accept an optional registry kwarg
(`llm_registry=` / `registry=`) for dependency injection. When
omitted, the factory imports the canonical singleton lazily on
first call. Tests pass stubs to verify the wiring without
triggering the host's full `atlas_brain.services.__init__`
chain (torch / ollama implementations); production callers
omit the kwarg and get the canonical singleton.

## Intentional

- **Adapters live in `atlas_brain/_content_ops_infrastructure.py`,
  not inside `api/`.** Same reason as
  `_content_ops_services.py`: imports under `atlas_brain.api.*`
  trigger the heavy router init chain (numpy / torch / asyncpg).
  Tests can pull these adapters in cleanly without that.
- **`metadata` kwarg on `complete()` is dropped.** The host's
  `chat()` doesn't accept arbitrary metadata. The extracted
  package's `metadata` is informational (asset_type tags etc.);
  losing it on the host adapter doesn't change generation
  behavior. Future host LLM versions may add a `metadata`
  passthrough; the adapter can plumb it then.
- **`build_content_ops_llm_client()` returns `None` when no
  LLM is active.** Cleaner than raising: the next slice's
  services-bundle factory can just skip wiring LLM-needing
  services and let them remain `service_not_configured`. The
  catalog endpoint already surfaces that to the UI.
- **No metaclass / abstract registration.** The adapter
  classes don't inherit from the extracted Protocols (they
  satisfy them structurally). Keeping them concrete classes
  with explicit method bodies makes the wire shape easy to
  audit.
- **`asyncio.to_thread` rather than spinning the host LLM in
  an async wrapper.** The host's `chat()` is sync today; if it
  ever goes async natively, the adapter changes one line.

## Deferred

- `landing_page` service wiring (next slice E2). Plugs LLM +
  Skill factories into the bundle.
- `campaign` / `blog_post` / `report` / `sales_brief` service
  wiring. Each adds a Postgres repo + plugs the same LLM/Skill
  adapters in.
- Streaming LLM responses. Extracted package's `LLMClient`
  Protocol is non-streaming today.
- `metadata` kwarg pass-through (host LLM doesn't accept it
  yet).
- Per-request LLM model selection. Today's adapter uses the
  globally active model.

## Verification

- `pytest tests/test_atlas_content_ops_infrastructure.py` --
  7 passed.
- AST-parse of the new module + the test file.
- ASCII-clean check on the new module + test file (the
  package-scoped `check_ascii_python.sh` script doesn't cover
  `atlas_brain/`; we run `read().encode('ascii')` separately).

## Estimated diff size

- `_content_ops_infrastructure.py`: ~110 LOC (LLM adapter +
  Skill adapter + 2 factories + module docstring).
- Test: ~150 LOC.
- Plan doc: ~190 LOC.

Total: ~450 LOC. Marginally over the 400 LOC soft cap; the
plan doc is most of it. Splitting LLM adapter from Skill
adapter would leave neither slice useful on its own (the
generators need both). Indivisible.
