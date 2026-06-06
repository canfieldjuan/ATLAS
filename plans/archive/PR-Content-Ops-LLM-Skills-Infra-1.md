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
   - `test_host_skill_store_falls_back_to_packaged_skills`
     -- pins the four packaged-only skill names
     (`digest/landing_page_generation`,
     `digest/report_generation`,
     `digest/sales_brief_generation`,
     `digest/b2b_campaign_reasoning_context`) resolve
     through the extracted-package fallback. Codex P2 canary.
   - `test_host_skill_store_returns_none_for_missing_skill`
     -- canary for the "skill not found" branch.
   - `test_host_llm_client_translates_messages_and_response`
     -- a fake host LLMService receives the translated messages
     and returns a dict; the adapter wraps it into an
     LLMResponse with the right shape.
   - `test_host_llm_client_messages_carry_tool_call_attributes`
     -- pins `.tool_calls` / `.tool_call_id` on each
     SimpleNamespace so cloud backends that read them during
     payload conversion don't AttributeError. Codex P1 canary.
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

   Total: 9 tests.

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
        del metadata  # host's chat() doesn't accept arbitrary metadata
        # SimpleNamespace duck-typing: importing
        # `atlas_brain.services.protocols` would trigger
        # `services/__init__.py` which eagerly loads ollama / torch.
        # The host's `chat()` reads only `.role` / `.content`
        # (and the cloud backends additionally read
        # `.tool_calls` / `.tool_call_id`); a SimpleNamespace
        # with those four attributes is structurally sufficient
        # without re-introducing the heavy import.
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
        # Response wrap: accept "response" or "content" aliases,
        # `model` from `host.model_info.name` (defensive), `usage`
        # only when Mapping-shaped, `raw` is the full dict.
        ...
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
- **`SimpleNamespace` duck-typing for the host `Message`
  shape.** Importing `atlas_brain.services.protocols` would
  trigger `services/__init__.py` which eagerly loads ollama /
  torch. The host's `chat()` reads only `.role` / `.content`
  from each message (and the cloud backends additionally read
  `.tool_calls` / `.tool_call_id`); a `SimpleNamespace` with
  those four attributes is structurally sufficient without
  re-introducing the heavy import. Codex P1 review on the
  initial commit confirmed the cloud-backend touchpoints
  (`atlas_brain/services/llm/openrouter.py:147`,
  `groq.py:105`, etc.); fields added with `None` defaults so
  the duck-typed shape stays compatible.
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
  9 passed.
- AST-parse of the new module + the test file.
- ASCII-clean check on the new module + test file (the
  package-scoped `check_ascii_python.sh` script doesn't cover
  `atlas_brain/`; we run `read().encode('ascii')` separately).

## Updates from review

After Codex review of the initial commit:

- **Codex P1 fix**: cloud backends (OpenRouter / Groq /
  Together / Ollama) read `msg.tool_calls` and
  `msg.tool_call_id` during payload conversion. The
  `SimpleNamespace` host-message shape now includes both
  fields with `None` defaults so payload conversion
  succeeds without re-introducing the heavy
  `services.protocols` import. New regression test
  `test_host_llm_client_messages_carry_tool_call_attributes`
  pins the contract.
- **Codex P2 fix**: skills the next-slice services depend on
  by default (`digest/landing_page_generation`,
  `digest/report_generation`,
  `digest/sales_brief_generation`,
  `digest/b2b_campaign_reasoning_context`) live only in the
  extracted package. The skill-store factory now uses the
  extracted package's `get_skill_registry(root=
  atlas_brain/skills/)`, which already does host-first /
  packaged-fallback. Host overrides win; missing host skills
  resolve through the packaged defaults. New regression test
  `test_host_skill_store_falls_back_to_packaged_skills` pins
  this for all four packaged-only skill names.

## Estimated diff size

Initial estimate undershot the implementation; defensive
typing + extracted-fallback wiring + the response-shape
unwrap added more than the rough mental model. Updated for
transparency.

- `_content_ops_infrastructure.py`: ~237 LOC actual (initial
  estimate ~110; the SimpleNamespace duck-type, the cloud
  backend `tool_calls` / `tool_call_id` fix-up, the
  defensive `Mapping` checks on the response, and the
  extracted-skill-fallback factory all came in heavier).
- Test: ~236 LOC actual (initial estimate ~150; 9 tests
  rather than 4).
- Plan doc: ~290 LOC actual (post-update, includes Updates
  section, SimpleNamespace bullet, and the expanded test
  inventory).

Total actual: **~755 LOC**. Over the 400 LOC soft cap. The
adapters and tests are structurally indivisible (the LLM and
Skill adapters share the docstring framing; splitting them
would leave one half unusable). Plan doc and tests dominate.
