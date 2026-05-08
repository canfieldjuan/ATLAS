# AI Content Ops post-merge audit (2026-05)

## Context

Post-merge audit of commits that landed direct-to-main between
PR-SalesBriefs-1b (#355) and PR #365, scoped to AI Content Ops.

Eleven commits (~2,400 LOC across 21 files) added a new control-surface
+ generation-plan + execution-seam layer plus a sixth blog-post asset
type. None of this work had per-PR review or a forward-looking scope
source declaration before merge. STATUS.md was updated to describe the
new layers, but only after the work merged — there was no `Plan:` line,
no `inflight.md` entry, and no scope source for a reviewer to verify
against.

This doc captures findings per file, organized by severity
(`BLOCKER` / `MAJOR` / `MINOR` / `NIT`). The recommended fix path
requires an architecture decision (see "Decision required" below)
before the follow-up PR(s) can be scoped.

The 11 commits in scope:

| Commit | Title |
|---|---|
| `aef4fdd4` | Tighten content asset shared contracts (`PR-ContentAssets-Consistency-1`) |
| `668c42b5` | Scaffold AI Content Ops control surfaces |
| `4b9b1887` | Add AI Content Ops execution seam |
| `d58fb978` | Add blog post generation service seam |
| `7616fc00` | Retry unparseable campaign draft JSON |
| `ec392da4` | Run Content Ops plan steps concurrently |
| `124b756e` | Retry unparseable blog post JSON |
| `3c4b00d1` | Add parse retry parity for generated assets |
| `83ba0eca` | Report Content Ops execution readiness |
| `809fce59` | Harden Content Ops execution service contract |
| `39aff435` | Harden Content Ops execution provider contract |

`aef4fdd4` was on-pattern (the consistency follow-up tracked in the
PR-#354 / PR-#355 reviews). The other ten commits introduced new product
surface that this audit walks.

---

## Audit progress

- [x] `extracted_content_pipeline/control_surfaces.py` (342 LOC, 6 tests)
- [x] `extracted_content_pipeline/generation_plan.py` (251 LOC, 6 tests)
- [x] `extracted_content_pipeline/content_ops_execution.py` (265 LOC, 6 tests)
- [x] `extracted_content_pipeline/api/control_surfaces.py` (195 LOC, 10 tests)
- [x] `extracted_content_pipeline/blog_generation.py` (396 LOC, 9 tests) + `blog_ports.py` (88 LOC)
- [ ] Parse-retry behavior change across the four existing generators (silent change to already-audited code) — pending

---

## Decision required: plan-as-execution-contract vs. host-pre-configures

The BLOCKER in `content_ops_execution.py` (below) forces a choice
between two architectures. Until this decision is made, the follow-up
PR cannot be scoped, because the right fixes for the upstream layers
(`control_surfaces.py`, `generation_plan.py`) depend on which one we
pick.

**Option A — plan-as-execution-contract**

The plan step config is load-bearing. The executor reads `step.config`
and threads each field to `service.generate(...)`. Requires:

- Every `*GenerationService.generate` accepts a `config` kwarg, OR
- Services become factories the executor instantiates per step, OR
- The executor builds a fresh `*GenerationConfig` per call and the
  service supports per-call config overrides.

Pro: the control surface is a real control surface. Per-call channel
selection, report type, brief type, parse-retry policy, etc. all do
what they say. Con: meaningfully larger change; touches every
generator's `generate` signature.

**Option B — host pre-configures everything**

Per-call config beyond `target_mode` / `limit` / `filters` /
`MarketingCampaign` is impossible. Host operators set channels,
report-type defaults, parse-retry policy, etc. at process-start time
via the injected `*GenerationConfig`. The control surface stops
displaying those fields. The plan stops listing them in
`step.config`.

Pro: minimum-change, matches what the executor already does today.
Con: the control surface becomes "which assets and how many" — no
true per-call tuning. UI affordances that look like per-call controls
are removed.

**Recommendation:** Option A is the right one if you want a control
surface. Option B is the right one if the per-call config story is
deliberately deferred. The current code is shape (A) pretending to be
(B). Pick.

---

## Top issues (across all files audited so far)

Ranked by impact:

1. **BLOCKER** — `content_ops_execution.py`: plan step config silently
   ignored by executor. Preview lies about what execution will do.
   Inherited by `api/control_surfaces.py` — the `POST /execute` route
   exposes the bug to HTTP callers verbatim.
2. **MAJOR** — `api/control_surfaces.py`: `errors[*].reason` from the
   executor (raw `str(exc)` from host service exceptions) is returned
   verbatim to HTTP clients. Information disclosure surface.
3. **MAJOR** — `api/control_surfaces.py`: `/preview`, `/plan`,
   `/execute` accept arbitrary unbounded JSON dicts. No Pydantic
   model, no size limit, no schema validation beyond soft coercions.
4. **MAJOR** — `content_ops_execution.py`: `MarketingCampaign.context`
   leaks unrelated request inputs (e.g., `report_type`, `opportunity_id`)
   into the landing-page LLM prompt.
5. **MAJOR** — `content_ops_execution.py`: `landing_page` is a
   hard-coded special case; every new asset with a non-standard
   `generate` signature requires another `if` branch.
6. **MAJOR** — `generation_plan.py`: landing-page step doesn't carry
   `MarketingCampaign` data — masked at runtime by the executor's
   `_marketing_campaign_from_inputs` reaching back into `request.inputs`.
7. **MAJOR** — `generation_plan.py`: step configs claim per-output
   config (channels, report_type, brief_type) that never reaches the
   service. Same root cause as #1.
8. **MAJOR** — `generation_plan.py`: `email_campaign` step emits
   `channels` (tuple) but `CampaignGenerationConfig` also has legacy
   `channel: str` field. Hidden config fields are a recurring drift
   symptom.
9. **MAJOR** — `control_surfaces.py`: silent preset-typo fallback to
   `email_only`. User types `preset="contmarket"`, gets email-only
   output, no diagnostic.
10. **MAJOR** — `control_surfaces.py`: `outputs` + `preset` both set:
    preset silently ignored. UI forms that send both think the preset
    is contributing.
11. **MAJOR** — `blog_generation.py`: blog generator consumes a
    different port (`BlogBlueprintRepository`) and ignores `topic`
    from `request.inputs`. Specific instance of #1 with a twist —
    the user input the catalog declares isn't passed AND the actual
    data source is host-injected blueprints.
12. **MAJOR** — `blog_generation.py`: concurrent writes to
    `blog_posts` from legacy autonomous task + new service. Slug
    uniqueness at DB level needs verification (migrations 084 + 264).

---

## Per-file findings

### `control_surfaces.py` (342 LOC, 6 tests)

**On-scope verdict for this file:** module is genuinely pure /
deterministic / no I/O — its docstring claim ("does not call an LLM,
read a database, or know about HTTP") holds. The pure-validator
pattern is internally coherent.

#### MAJOR — Silent preset-typo fallback

`resolve_outputs` (L223-232) falls back to `email_only` when
`request.preset` doesn't match a `PRESETS` entry, with no warning.

```python
if request.preset:
    preset = PRESETS.get(request.preset)
    if preset:
        return preset.outputs
return PRESETS["email_only"].outputs  # silent default
```

**Failure mode:** caller passing `preset="contmarket"` (typo for
`content_marketing`) gets email-only output and no diagnostic.
Persists through preview into the generation plan. Operator sees an
email and wonders why the blog post they thought they ordered didn't
run.

**Fix:** in `preview_control_surface`, when `request.preset` is set
but not in `PRESETS`, append `f"Unknown preset: {request.preset}"` to
warnings and set `can_run=False` (consistent with how
`unknown_outputs` is handled — both are user errors).

#### MAJOR — `outputs` + `preset` both set: preset silently ignored

`resolve_outputs` returns `request.outputs` first, ignoring any
preset. No warning.

**Failure mode:** UI forms or API clients that always send both
fields think their preset is contributing. Operator confusion when
explicit outputs override preset selection invisibly.

**Fix:** warn `"preset '{request.preset}' ignored because explicit
outputs were provided"` when both are set.

#### MINOR — `missing_required_inputs` empty-check asymmetric

L264:
```python
if value in (None, "", [], {}):
```

Catches `[]` and `{}` and `""` and `None` as missing. Does not catch
empty tuple `()` or empty `set()`.

**Failure mode:** caller passes `tags=()` and gets past validation;
caller passes `tags=[]` and gets a `missing_inputs` error. Same
semantic, different behavior.

**Fix:** change to `if not value:` (truthiness) or expand the tuple
to include `()` and `set()`. Truthiness is simpler unless `0` and
`False` should count as present (current behavior treats them as
present, which is correct for input semantics).

#### MINOR — `limit=0` silently coerced to 1

`request_from_mapping` (L209): `max(1, int(payload.get("limit") or 1))`.
The `or 1` short-circuits when `limit=0` because `0 or 1` is `1`,
before `max` ever runs.

**Failure mode:** caller asks for `limit=0` (e.g., "preview only,
generate nothing") and gets a plan for 1 item. UI that exposes a
0-item dry-run mode silently runs.

**Fix:** drop the `or 1` and clamp at `max`:
`max(1, int(payload.get("limit") or 0))`. Or surface the floor
explicitly with a warning when input was below 1.

#### MINOR — `max_cost_usd ≤ 0` accepted without validation

A negative budget produces a warning ("estimate exceeds budget") and
`can_run=False`, but the diagnostic is misleading — the real issue is
the budget value, not the estimate.

**Failure mode:** operator sees "Estimated cost 1.46 > -1.00" warning
and tries to debug their cost estimate. Real fix is in
`max_cost_usd`.

**Fix:** in `request_from_mapping`, when
`max_cost_usd is not None and float(value) <= 0`, warn explicitly
(`"max_cost_usd must be positive; got {value}"`) and either reject
or treat as no budget.

#### MINOR — `inputs` non-Mapping silently becomes `{}`

L215:
```python
payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
```

**Failure mode:** caller passes `inputs=["target_account", "Acme"]` (a
list by mistake — common API contract slip) and gets
`missing_inputs=["target_account", "offer", ...]`. The diagnostic
blames the wrong thing.

**Fix:** when `payload.get("inputs")` is non-None and not a Mapping,
raise `ValueError("inputs must be an object")` or surface a warning
so the caller knows what actually went wrong.

#### NIT — `OUTPUT_CATALOG` and `PRESETS` are mutable module-level dicts

Anyone can `OUTPUT_CATALOG["report"] = ...` from another module and
the change persists. Not a real concern in the current codebase, but
`MappingProxyType` would lock it cheaply.

#### Test coverage gaps

The 6 tests exist (CSV+dedupe, default→email_only, happy path, blocks
unimplemented, blocks over budget, allows future via flag) but the
following edges aren't pinned:

- Unknown preset id fallback (the MAJOR above)
- `outputs` + `preset` both set: precedence (the MAJOR above)
- `request_from_mapping` malformed input (`limit="abc"`,
  `inputs=[]`, `max_cost_usd=-1`)
- `missing_required_inputs` with `()` vs `[]`
- Catalog completeness (5 implemented + 1 unimplemented as documented)
- `as_dict` shape (the API contract for the FastAPI router that
  consumes it)

---

### `generation_plan.py` (251 LOC, 6 tests)

**On-scope verdict for this file:** bridge layer that consumes
`ContentOpsRequest` from `control_surfaces.py` and produces a per-asset
step list naming a runner string + a config snapshot. Inherits any
silent-fallback bugs from `control_surfaces.py` (preset typos,
outputs+preset precedence). No new I/O — pure mapping.

#### MAJOR — `landing_page` step claims `runnable` but doesn't carry the data the runner needs

`LandingPageGenerationService.generate` signature:
```python
async def generate(self, *, scope: TenantScope, campaign: MarketingCampaign) -> LandingPageGenerationResult
```

Takes a `MarketingCampaign` (per-marketing-campaign trigger). Does not
take a `limit`, an `opportunity_id`, an `offer`, or an `audience`
field.

The generation plan for `landing_page` (L160-173) emits a config dict
that is missing both `MarketingCampaign` and the upstream catalog's
declared `required_inputs=("offer", "audience")`. The step is marked
`status="runnable"` regardless.

**Failure mode at the plan layer (decorative):** `step.config` is a
lie about what the runner needs.

**Failure mode at runtime:** masked by `content_ops_execution.py`'s
`_marketing_campaign_from_inputs` reaching back into `request.inputs`
to build a `MarketingCampaign`. So the system doesn't actually fail
at call time — but only because the executor ignores the plan and
re-derives data from the request. Two-source-of-truth bug.

**Fix:** plan step config carries the data the runner needs. Either
include a serialized `MarketingCampaign` payload in `step.config`,
OR mark `landing_page` as a "needs adapter at execution time"
status that the executor explicitly handles. Don't claim runnable
when the config alone isn't enough to run.

#### MAJOR — Step configs forward `limit` but lose other inputs needed at call time

`SalesBriefGenerationService.generate` and
`ReportGenerationService.generate` take
`(scope, target_mode, limit, filters)` and consume opportunities via
`intelligence.read_campaign_opportunities`. The `inputs` dict from the
request (with `target_account`, `opportunity_id`, etc.) is not
threaded into the step config.

**Failure mode:** the executor either (a) ignores the user's filter
intent and runs the generator against the host's full opportunity
list, or (b) has to reach back into the request payload to recover
what the plan should have already serialized. Two-source-of-truth bug
again.

**Fix:** the step config should carry `request.inputs` (or a filtered
subset matching the output's `required_inputs`) so the executor has a
single payload to act on.

#### MAJOR — `email_campaign` config emits `channels` but `CampaignGenerationConfig` also has legacy `channel: str`

`CampaignGenerationConfig` declares both:
```python
channel: str = "email"               # singular, legacy
channels: tuple[str, ...] = ()       # plural, new
```

The plan only emits `channels` in its config snapshot (L131-141).

**Failure mode:** any code path inside `CampaignGenerationService`
that still consults `config.channel` (singular) gets `"email"`
regardless of what the user picked.

**Fix:** confirm `CampaignGenerationService` only reads `channels`
(plural) and remove the legacy `channel` field, OR include both
fields in the plan config and document the precedence.

#### MINOR — `_landing_page_config_for_request(request)` discards the request

L104-106:
```python
def _landing_page_config_for_request(request: ContentOpsRequest) -> LandingPageGenerationConfig:
    del request
    return LandingPageGenerationConfig()
```

Other `_*_config_for_request` helpers consume `request.inputs` and
`request.limit`. This one discards the request and returns defaults.
Combined with the MAJOR above (no `MarketingCampaign` plumbing), the
landing-page step is the most under-specified in the plan.

**Fix:** part of the same MarketingCampaign-plumbing fix above.

#### MINOR — `quality_gates_enabled` exists in some step configs and not others

Three different conventions for the same conceptual setting in five
generators:

- `email_campaign` step uses `quality_revalidation_enabled` (the
  actual `CampaignGenerationConfig` field name)
- `landing_page` and `sales_brief` step configs include a synthesized
  `quality_gates_enabled` that isn't an actual field on the
  `LandingPageGenerationConfig` or `SalesBriefGenerationConfig`
  dataclasses
- `report` and `blog_post` step configs don't include any
  quality-gate flag

**Failure mode:** consumer reading the step config has to know which
field name to look for on which output. Someone wires a UI checkbox
to "quality gates" and it works for landing-page/sales-brief but
silently does nothing for report/blog/email.

**Fix:** decide on one convention. Either every step config includes
`quality_gates_enabled: bool` mirroring `request.require_quality_gates`
(purely informational), or every step config includes the underlying
service-config field name (load-bearing). Mixing both is the worst
of both.

#### Test coverage gaps

The 6 tests assert step config shape and `status == "runnable"`. None
of them:

- Construct an actual `LandingPageGenerationConfig` /
  `ReportGenerationConfig` from the step config and verify it's a
  valid dataclass invocation
- Verify that `LandingPageGenerationService(...)` could actually be
  called with whatever the plan claims is needed
- Verify `_step_for_output` for an unknown output (L207-212 fallback)

Without those, the MAJOR above (landing-page runner + missing
MarketingCampaign) wasn't caught.

**Suggested:** one round-trip test per asset that verifies step config
keys are a subset of the dataclass fields, and that any kwargs the
executor would pass are present in the config.

---

### `content_ops_execution.py` (265 LOC, 6 tests)

This audit changes the read of the prior layers. Most of the
`generation_plan.py` MAJORs turn out to be symptoms of one
architectural BLOCKER that lives here.

#### BLOCKER — Plan's per-output config silently ignored at execution time

`_run_step` (L180-198) is the entire dispatch surface. For every
output except `landing_page`:

```python
return await service.generate(
    scope=scope,
    target_mode=request.target_mode,
    limit=request.limit,
    filters=filters,
)
```

That's the contract. Nothing from `step.config` is passed. The user
picks `channels=["email_cold"]` in the control surface, the plan
records `"channels": ["email_cold"]` in `step["config"]`, and the
executor calls `service.generate(scope, target_mode, limit, filters)`
— no channels, no `parse_retry_attempts`, no `temperature`, no
`report_type`, no `brief_type`, no `topic`. The host's pre-injected
`CampaignGenerationService` (constructed once at process start with
whatever `CampaignGenerationConfig` the host supplied) decides
everything.

**Failure mode:**
- The control surface preview shows the user
  `outputs: ["email_campaign"], channels: ["email_cold"]`.
- The plan step config shows the user
  `runner: "CampaignGenerationService.generate", channels:
  ["email_cold"]`.
- The execution calls `campaign_service.generate(scope=...,
  target_mode="vendor_retention", limit=2, filters=None)`.
- The campaign service uses whatever channels were injected at
  construction time (default: `("email_cold", "email_followup")`).
- The user sees both an email_cold draft AND an email_followup draft,
  despite asking for cold-only in the UI.

This breaks the trust contract between preview and execution. The
preview is decoration. Severity is BLOCKER because the system actively
misleads the operator about what their controls do — same
severity-bracket as "wrong data persisted."

**Fix:** depends on the architecture decision above. Either Option A
(plan-as-execution-contract) or Option B (host-pre-configures, strip
step config of fields the executor can't pass). Pick before fixing
upstream layers.

#### MAJOR — `landing_page` is special-cased; everything else falls through

L188-198:
```python
if step.output == "landing_page":
    return await service.generate(
        scope=scope,
        campaign=_marketing_campaign_from_inputs(request.inputs),
    )
return await service.generate(
    scope=scope,
    target_mode=request.target_mode,
    limit=request.limit,
    filters=filters,
)
```

`landing_page` gets a `MarketingCampaign` built from `request.inputs`
(good — that's what the service signature requires). Every other
output gets the same generic `(scope, target_mode, limit, filters)`
shape.

**Failure mode:** the next asset that has a non-standard signature
(signal_extraction is already in the catalog as `implemented=False`
— what shape will its `generate` take?) needs another hard-coded
`if step.output == "...":` branch. The dispatcher doesn't scale
beyond hard-coding.

**Fix:** route through a per-output adapter table. Something like:

```python
_DISPATCH = {
    "landing_page": lambda svc, *, request, scope, filters: svc.generate(
        scope=scope,
        campaign=_marketing_campaign_from_inputs(request.inputs),
    ),
    # default: all per-opportunity services
}
async def _run_step(step, *, request, service, scope, filters):
    handler = _DISPATCH.get(step.output, _per_opportunity_default)
    return await handler(service, request=request, scope=scope, filters=filters)
```

That isolates per-asset dispatch differences in one structure, makes
adding a new asset a one-line registry edit, and forces the question
"what shape does this service's `generate` actually take?" at the
registry-add site.

#### MAJOR — `MarketingCampaign.context` leaks unrelated request inputs

`_marketing_campaign_from_inputs` (L206-222):

```python
context={
    str(key): value
    for key, value in inputs.items()
    if key not in {"campaign_name", "offer", "audience", "vendors", "categories", "tags"}
},
```

Every `inputs` key not in the explicit allow-list lands in
`MarketingCampaign.context`. So a request like:

```python
inputs={
    "campaign_name": "Q2",
    "offer": "Churn audit",
    "audience": "B2B SaaS founders",
    "topic": "Renewal pressure",
    "report_type": "competitive_pressure",
    "opportunity_id": "opp_42",
    "filters": {"status": "ready"},
}
```

…produces `MarketingCampaign(context={"topic": "...", "report_type":
"...", "opportunity_id": "...", "filters": {...}})`. The landing-page
LLM prompt serializes that context (per the skill's
`{campaign_json}` substitution). The model sees `report_type` and
`opportunity_id` and `filters` as if they were marketing-campaign
metadata.

**Failure mode:** prompt pollution. The LLM may pick up irrelevant
fields and write copy referencing `opp_42` or `competitive_pressure`
because they're in the JSON it was told to read. Quality may degrade
silently.

**Fix:** invert the allow-list to a deny-list of marketing-only keys,
OR restrict context to keys explicitly tagged for marketing-campaign
use:

```python
context={
    str(key): value
    for key, value in inputs.items()
    if key.startswith("marketing_") or key in MARKETING_CONTEXT_KEYS
}
```

#### MINOR — All-steps-failed reports "partial" not "failed"

L122:
```python
status = "completed" if not errors else "partial"
```

If every step fails (e.g., none of the host services are configured),
`status="partial"` with zero successful steps. Should be "failed"
semantically. A consumer scripting on `status` might retry "partial"
but escalate "failed" — current logic merges both into a single
bucket.

**Fix:**
```python
if not errors:
    status = "completed"
elif not executed_successes:  # all steps failed
    status = "failed"
else:
    status = "partial"
```

#### MINOR — Host service concurrency assumption is undocumented

L109-118 calls `asyncio.gather(*step_executions)`. Host-injected
services run concurrently. If a host injects a stateful service
(e.g., one that maintains an in-memory queue or counter), the
executor races it silently.

**Failure mode:** a host that built their service against the old
single-step execution path adopts the control surface and gets
random races. Hard to diagnose because the service's tests pass
single-threaded.

**Fix:** docstring on `ContentOpsExecutionServices` (or
`execute_content_ops_request`) calling out: "Services must be safe
for concurrent calls; the executor invokes per-step `generate()` in
parallel via asyncio.gather."

#### MINOR — `_failed_step` and `error` dict have inconsistent shapes

L140-141:
```python
error = {"output": step.output, "reason": "service_not_configured"}
return _failed_step(step, "service_not_configured"), error
```

The `error` dict carries `(output, reason)`.
`ContentOpsStepExecution.error` carries the bare reason string. Same
information, two shapes. If they're both present, they can diverge
after a refactor.

**Fix:** drop the standalone `errors` list (the per-step errors are
already in `result["steps"][N]["error"]`) OR make the per-step error
a `Mapping[str, Any]` instead of `str`. One source of truth.

#### NIT — `for_output` if/elif chain

L26-37 is a six-branch if/elif chain. A dict literal would be
slightly cleaner. Worth folding the cleanup into the dispatcher-table
refactor (MAJOR above).

#### Test coverage gaps

The 6 tests verify concurrent dispatch, order preservation, landing
page MarketingCampaign construction, filters threading, and missing
service handling.

**Not tested (the BLOCKER above is invisible without these):**

- User's `channels=["email_cold"]` from control surface actually
  reaches the campaign service
- User's `report_type` selection takes effect
- User's `parse_retry_attempts` override actually overrides
- All-failed steps surface as "failed" not "partial"
- Mixed success/failure preserves both
- Service raising an exception surfaces as step error
- Non-Mapping `inputs.filters` falls back to `None` correctly
- `MarketingCampaign.context` doesn't leak unrelated inputs

The first three would catch the BLOCKER immediately and force the
architecture decision.

---

### `api/control_surfaces.py` (195 LOC, 10 tests)

**On-scope verdict for this file:** FastAPI router factory exposing
4 routes (`GET /control-surfaces`, `POST /preview`, `POST /plan`,
`POST /execute`). Soft FastAPI dependency (try/except ImportError) so
the lower layers stay importable without FastAPI installed — good.
Execute route correctly opt-in (503 if no provider configured) — good.
Hosts inject auth via `dependencies` kwarg — standard pattern.

The router itself is thin: it delegates to `preview_from_mapping`,
`build_generation_plan_from_mapping`, and
`execute_content_ops_from_mapping`. Most findings here are inherited
from the layers below + the additional concerns of "this is now an
HTTP surface."

#### BLOCKER (inherited) — `POST /execute` exposes the per-call-config-silently-ignored bug to HTTP callers

The execute route (L130-155) calls `execute_content_ops_from_mapping`
and returns its result. Whatever the BLOCKER in
`content_ops_execution.py` does at the dispatcher layer, an HTTP
caller hitting `POST /execute` gets exactly that. The route adds no
guards.

**Failure mode:** an external API consumer sends a JSON request with
`{"outputs": ["email_campaign"], "inputs": {"channels":
["email_cold"]}}`, sees a 200 response with the plan showing
`channels: ["email_cold"]`, and gets back drafts the host service
generated using its construction-time channel default (likely both
cold and follow-up). The HTTP layer doesn't reveal the lie any more
than the in-process layer did, but it makes it remotely exploitable
as misconfiguration.

**Fix:** depends on the architecture decision. Whichever way the
underlying BLOCKER is fixed, this route inherits it.

#### MAJOR — Execution error responses leak internal exception strings to HTTP callers

The execute route returns `result` from
`execute_content_ops_from_mapping` directly to the client (L148-155).
That payload includes `result["errors"]`, which contains entries
shaped:

```python
{"output": step.output, "reason": str(exc)}
```

`str(exc)` for an arbitrary host service exception leaks whatever
the exception message contains — DB connection strings if asyncpg
included one in a traceback, file paths, internal IDs, full tracebacks
if the service stringifies its own state, etc. The route returns this
verbatim.

**Failure mode:** information disclosure to API consumers. A request
that triggers an internal failure can surface infrastructure details
through the response body. Standard OWASP "verbose error message"
class.

**Fix:** at the API boundary, sanitize `errors[*].reason` to a stable
error code (`"execution_failed"`) and log the full reason
server-side. Or document that hosts must add a response sanitizer
middleware. Note: the executor itself is host-injected service code
so the host knows which exceptions are safe to expose — but the
default should be safe.

#### MAJOR — `/preview`, `/plan`, `/execute` accept arbitrary unbounded JSON dicts

L123, L127, L131:
```python
async def preview_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
async def plan_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
async def execute_generation(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
```

No Pydantic model, no size limit, no schema validation beyond the
soft coercions in `request_from_mapping` (which silently swallow
malformed types into defaults). FastAPI/Starlette will accept any
size payload by default unless the host adds middleware. A malicious
or buggy caller can:

- POST a 10MB JSON object and exhaust memory during parsing
- POST deeply nested objects that hit Python's recursion limit
- POST unexpected types that trigger AttributeError downstream
  (caught by the generic `except Exception` in the executor, but
  logged + returned as 500 anyway)

**Failure mode:** API DoS surface. Cheap to fix at the parsing layer.

**Fix:** define a Pydantic model that mirrors the `ContentOpsRequest`
shape and use it as the body type:
```python
class ContentOpsRequestModel(BaseModel):
    target_mode: str = "vendor_retention"
    preset: Optional[str] = None
    outputs: Sequence[str] = ()
    limit: int = Field(1, ge=1, le=1000)
    max_cost_usd: Optional[float] = Field(None, ge=0)
    inputs: Mapping[str, Any] = Field(default_factory=dict)
    ...

@router.post("/preview")
async def preview_generation(payload: ContentOpsRequestModel) -> dict[str, Any]:
    return preview_from_mapping(payload.dict())
```

This locks payload structure, bounds numeric fields, and lets
FastAPI's parser reject malformed input before it reaches the
control-surface code. Or document the host responsibility to add a
size-limiting middleware and accept the schema validation gap.

#### MINOR — `partial` / `failed` execution results return HTTP 200

L153-155:
```python
if result["status"] == "blocked":
    raise HTTPException(status_code=400, detail=result)
return result
```

Only `status == "blocked"` (plan not executable: budget violation,
missing inputs, blocked outputs) maps to 4xx. `status == "partial"`
or (after the `content_ops_execution.py` MINOR fix) `status ==
"failed"` returns 200.

**Failure mode:** consumers scripting on HTTP status code think the
call succeeded when zero outputs were produced or all services
failed. Standard pattern is to map server-side execution failure to
5xx (or 207 Multi-Status for partial success).

**Fix:** map status to HTTP code:
- `blocked` → 400 (current)
- `failed` (all steps failed) → 502 Bad Gateway (host service issue)
- `partial` (some succeeded) → 207 Multi-Status (or 200 with
  documentation that response payload is the source of truth)
- `completed` → 200 (current)

Or document that HTTP status only signals "did the request reach
us" and `result.status` is the source of truth. Either is defensible
if explicit.

#### MINOR — Provider exceptions propagate as bare 500s

`_resolve_provider` (L167-173) doesn't catch exceptions from the
provider callable. If the provider raises (e.g., DB connection error
retrieving services, or a misconfigured DI container), the exception
propagates from the route handler as an unhandled 500 with whatever
exception message the framework decides to render.

**Failure mode:** same information-disclosure concern as the
`errors[*].reason` issue above. A provider stack trace at startup
might include DB credentials in some deployments.

**Fix:** wrap the provider call:
```python
try:
    value = provider()
    if hasattr(value, "__await__"):
        value = await value
except Exception as exc:
    logger.exception("Content Ops provider failed")
    raise HTTPException(
        status_code=503,
        detail="Content Ops services are temporarily unavailable.",
    ) from exc
```

#### MINOR — `describe_control_surfaces` calls the execution-services provider on every request

L81-86:
```python
execution_services = await _resolve_execution_services(execution_services_provider)
configured_outputs = set(
    execution_services.configured_outputs()
    if execution_services is not None
    else ()
)
```

If the provider hits a DB or external service to instantiate the
execution services, every catalog request triggers that work. UI
admin dashboards that poll this route at 30s intervals would
re-instantiate every poll.

**Failure mode:** provider-side resource churn (DB connections, LLM
client objects) in proportion to UI poll frequency, not actual usage.

**Fix:** either (a) cache the result for the lifetime of the router
(at the cost of services becoming stale until restart), or (b)
document that providers should be cheap (idempotent factory return,
not a per-call construction).

#### MINOR — `_scope_from_value` doesn't strip empty strings from `allowed_vendors` / `roles`

L182-183:
```python
allowed_vendors=tuple(str(item) for item in value.get("allowed_vendors", ()) or ()),
roles=tuple(str(item) for item in value.get("roles", ()) or ()),
```

`account_id` and `user_id` go through `_clean()` (strip + None on
empty). `allowed_vendors` and `roles` don't. A scope payload with
`{"allowed_vendors": [""]}` produces `allowed_vendors=("",)`.

**Failure mode:** a downstream tenant-scoping check like
`vendor in scope.allowed_vendors` would match an empty vendor name
and bypass scoping. Edge case, but worth tightening since it's a
security boundary.

**Fix:**
```python
allowed_vendors=tuple(
    item for item in (str(v).strip() for v in value.get("allowed_vendors", ()) or ())
    if item
),
roles=tuple(
    item for item in (str(r).strip() for r in value.get("roles", ()) or ())
    if item
),
```

#### MINOR — Wrong-type provider return diagnoses as "not configured" (503)

`_resolve_execution_services` (L160-164) silently treats a wrong-type
provider return as no provider:

```python
async def _resolve_execution_services(provider):
    value = await _resolve_provider(provider)
    return value if isinstance(value, ContentOpsExecutionServices) else None
```

A host accidentally returning `dict(campaign=...)` instead of
`ContentOpsExecutionServices(campaign=...)` debugs as a 503
"services not configured" when the real bug is in the provider.

**Failure mode:** misleading diagnostic. Operator chases provider
configuration when the bug is provider return type.

**Fix:** raise (or 500) with a clearer message when the provider
returns a value that's not None and not a
`ContentOpsExecutionServices`:
```python
if value is None:
    return None
if not isinstance(value, ContentOpsExecutionServices):
    logger.error(
        "Content Ops execution provider returned %r; "
        "expected ContentOpsExecutionServices",
        type(value).__name__,
    )
    return None  # or raise
```

#### Test coverage gaps

The 10 tests are reasonably thorough for the API contract:
- Catalog/presets/ingestion-profiles in describe response ✓
- Execution-configured reporting (3 variants) ✓
- Preview/plan endpoint smoke ✓
- Execute happy path ✓
- Execute 503 when no provider ✓
- Execute 503 when provider returns wrong type ✓
- Config validates absolute prefix ✓

**Not tested (would catch the BLOCKER + MAJORs):**
- User's `channels=["email_cold"]` POST'd to `/execute` doesn't reach
  the campaign service. The happy-path test checks
  `service.calls[0]["target_mode"]`, `["limit"]`, `["filters"]`,
  `["scope"]` — not what was OR wasn't passed for channels. So the
  test confirms the bug rather than catching it.
- `/execute` with a service that raises inside `generate()` —
  what HTTP response does the consumer see? Verifies the leakage
  concern above.
- `/execute` returning `partial` status doesn't surface as HTTP 4xx/5xx
- `/execute` with a Pydantic validation failure (when added)
- Auth dependencies actually firing (the `dependencies` param is
  passed but no test exercises it)
- `/preview` and `/plan` with malformed payloads (non-dict body, list
  instead of dict, oversized JSON)
- Concurrent `/execute` calls racing the host's services

---

### `blog_generation.py` (396 LOC, 9 tests) + `blog_ports.py` (88 LOC)

**On-scope verdict for these files:** Standalone blog-post generator
service mirroring `ReportGenerationService` /
`SalesBriefGenerationService` shape. Reuses pre-existing storage
(`blog_posts` table, migration 084 + 264) and pre-existing quality
pack (`extracted_quality_gate.blog_pack`, PR-B4a era).

**Important nuance to my earlier "5 of 5 → 6 silently" framing:** this
is NOT a brand-new asset type smuggled in. Blog already had storage
and a quality pack; what was missing was a standalone
`BlogPostGenerationService` matching the new control-surface pattern.
The diff still expands the asset surface from 5 to 6 in the control
catalog, but the underlying blog product was already in the codebase
— this just wraps it in the new service shape.

**Lessons from earlier reviews applied:**

- ✅ `json.JSONDecoder.raw_decode` parser (PR-Reports-1b lesson)
- ✅ System prompt `{blueprint_json}` once; user message structural
  (with template-fallback if placeholder missing) (PR-Reports-1b
  lesson)
- ✅ Quality pack imported at module top (PR-Reports-1b lesson)
- ✅ `BlogPostRepository.update_status` Protocol returns `bool`
  (PR-LandingPage-1a lesson)
- ✅ Parse-retry with usage accumulation across attempts (new
  pattern from `3c4b00d1` / `124b756e`)
- ✅ `_slugify` collapses non-alphanumeric runs to single dashes
  (correct fix not present in `landing_page_generation`'s
  `_slug_default`)

#### MAJOR — Blog generator consumes a different port; user's `topic` input from control surface doesn't reach the service

`BlogPostGenerationService.generate(scope, target_mode, limit,
filters)` consumes `BlogBlueprintRepository.read_blog_blueprints` —
NOT `IntelligenceRepository.read_campaign_opportunities` like the
other generators. Blueprints are produced by some other host-side
process (legacy autonomous task, manual import, etc.) and the
generator just consumes them.

`control_surfaces.OUTPUT_CATALOG["blog_post"]` declares
`required_inputs=("topic",)` — but:
- The user's `topic` from `request.inputs` doesn't reach the service.
  `generate()` takes only `target_mode`, `limit`, `filters`.
- The actual blog source data is the host-injected
  `BlogBlueprintRepository`, which the operator can't influence
  per-call.

**Failure mode:** a user picking blog_post and entering
`topic="Renewal pressure"` in the control surface gets blog posts
generated from whatever their host's blueprint repository returns
(presumably whatever blueprints were prepared by the legacy
autonomous task or a separate ingestion step). The topic input is
decorative.

This is a specific manifestation of the BLOCKER from
`content_ops_execution.py`, with a twist: the input the catalog
declares ("topic") isn't passed to the service either way, AND the
actual data source is a separate port.

**Fix (depends on architecture decision):**
- **Option A (plan-as-execution-contract):** Add a `topic` filter
  the executor passes through to `generate(filters={"topic": ...})`.
  Requires `BlogBlueprintRepository.read_blog_blueprints` to support
  filtering by topic.
- **Option B (host-pre-configures):** Change
  `OUTPUT_CATALOG["blog_post"].required_inputs` from `("topic",)` to
  `()`. Document that blog post generation pulls from host-side
  blueprints, not per-call inputs. The control surface stops
  prompting for a topic.

#### MAJOR — Concurrent writes to `blog_posts` table from legacy autonomous task and new service

The legacy `b2b_blog_post_generation` autonomous task (in
`atlas_brain/autonomous/tasks/`) writes to the same `blog_posts`
table. With the new `BlogPostGenerationService` exposed via control
surfaces, two write paths exist for the same table:

1. The legacy autonomous task running on its scheduled cadence
2. The new service called via `POST /content-ops/execute`

If both run concurrently and produce blog posts with the same slug
(e.g., both processing the same blueprint), the writes collide.

**Verification needed:** does migration 084 (or 264) enforce
`UNIQUE(slug)` or `UNIQUE(account_id, slug)`? If yes, the second
write fails with an integrity error. If no, both rows persist with
the same slug and downstream consumers (URL routing, listing) get
ambiguous results.

I haven't audited the migrations but I'd recommend verifying:

```bash
grep -A2 "CREATE.*INDEX\|UNIQUE" extracted_content_pipeline/storage/migrations/084_blog_posts.sql
grep -A2 "CREATE.*INDEX\|UNIQUE" extracted_content_pipeline/storage/migrations/264_blog_post_rejection_count.sql
```

**Failure mode:** silent slug collision, ambiguous post resolution.

**Fix:** if uniqueness isn't enforced at the DB level, either add a
constraint migration OR have the new service check for existing
slugs before insert (and either skip, version, or fail-loudly).

#### MINOR — `parse_blog_post_response` rejects valid JSON missing title or content

L74:
```python
if title and content:
    return {**dict(decoded), "title": title, "content": content}
return None
```

If the LLM returns valid JSON with `{"title": "..."}` but no
`content` (or vice versa), parser returns `None` and the executor
reports `unparseable_response`. The quality pack would have given a
specific `no_content` blocker.

Same parser-strictness pattern from the `landing_page_generation.py`
audit, same fix: parser should accept any well-formed JSON object
with at least one identifying field; quality pack judges the rest.

**Fix:** change to `if title:` (relax content requirement at parser
layer; quality pack handles missing content).

#### MINOR — `_accumulate_usage` assumes per-call usage from the LLM client

L101-119: accumulates usage across retry attempts under the
assumption that each `_llm.complete` call returns *just that call's*
usage. If a host's LLM client returns *cumulative* usage across the
session (some clients do), retries double-count tokens.

**Failure mode:** inflated cost reporting in
`metadata.generation_usage.input_tokens`. Doesn't break anything
upstream because the field is just informational, but operators
debugging cost will be confused.

**Fix:** docstring on `_accumulate_usage` (or
`BlogPostGenerationConfig.parse_retry_attempts`) noting the
"per-call usage" assumption. Or detect cumulative usage by checking
if the new attempt's `input_tokens` is monotonically increasing.

#### MINOR — Slug has no length limit

L348-351:
```python
def _slugify(value: Any) -> str:
    text = str(value or "blog-post").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug or "blog-post"
```

A 2000-char title becomes a 2000-char slug. Most CMS / URL routing
expects slugs <100 chars. The `blog_posts` table column is presumably
TEXT (no DB-level limit) but URLs would be unwieldy.

**Fix:** add a length cap:
```python
return slug[:96].rstrip("-") or "blog-post"
```

#### MINOR — Retry user message includes 800-char excerpt of prior LLM response

`_blog_generation_user_prompt` (L84-91) embeds up to 800 chars of
the prior invalid response in the retry user message. If the LLM
echoed sensitive data in its first attempt (e.g., parts of the
blueprint, host-specific identifiers, internal IDs), the retry
message re-includes that excerpt.

**Failure mode:** information leak within the same LLM session.
Probably benign because the model already saw the data, but worth
noting if the LLM is being audited or if responses are logged with
different retention than blueprints.

**Fix:** redact common patterns from the excerpt OR note in the
config docstring that `parse_retry_response_excerpt_chars=0`
disables the echo entirely.

#### NIT — `grounded_vendors` is `frozenset` where blog_pack docstring says `set`

L334:
```python
"grounded_vendors": frozenset(_string_tuple(blueprint.get("grounded_vendors"))),
```

`blog_pack`'s docstring (L42) declares `grounded_vendors: set[str]`.
`isinstance(frozenset(...), set)` is False. If `blog_pack` does
duck-typing (`vendor in grounded_vendors`), this works. If it does
`isinstance(grounded_vendors, set)`, it doesn't.

**Verification needed:** quick grep of `blog_pack.py` for
`isinstance.*grounded_vendors` or set-only operations. If any exist,
swap `frozenset(...)` to `set(...)`. If duck-typing only, fine as-is.

#### Test coverage gaps

The 9 tests cover:
- Parser strips code fences ✓
- Parser returns None for missing fields ✓ (pins the parser-strictness MINOR)
- Generate happy path via ports ✓
- Blueprint in user message when template lacks placeholder ✓
- Blocks low-quality posts ✓
- Reports unparseable ✓
- Parse retry behavior (3 paths: default-on, disabled, usage accumulation) ✓ — strong pinning of the new pattern from `3c4b00d1` / `124b756e`

**Not tested:**
- The MAJOR: end-to-end `topic` from control surface doesn't reach
  `generate()` (the test calls `generate()` directly; the
  control-surface roundtrip isn't tested at all)
- Concurrent writes to `blog_posts` from legacy + new code paths
- Slug edge cases (empty, very long, only-special-chars)
- `parse_blog_post_response` accepts a valid JSON missing only
  `content` and lets quality pack fail it (would catch the
  parser-strictness MINOR)
- `_accumulate_usage` with cumulative-usage LLM client (would
  surface the assumption MINOR)

---

## Suggested follow-up PR scope

(Pending the architecture decision above.)

**`PR-ContentOps-PostMergeAudit-1a: fix execution-blocking gaps in
control-surface plan layer`**

Whichever architecture is chosen, this PR fixes:

- The BLOCKER: plan step config either reaches the service (Option A)
  or is stripped of decorative fields (Option B)
- The MarketingCampaign context leak (real prompt-pollution bug,
  same fix either way)
- The "partial" vs "failed" status semantics
- The dispatcher table refactor (folds the `landing_page` special
  case + future-proofs)
- The `control_surfaces.py` MAJORs (silent preset-typo fallback,
  outputs+preset precedence) which are still real but smaller relative
  to the architectural fix

Plus three regression tests pinning the BLOCKER:

1. `channels=["email_cold"]` in request reaches the campaign service
2. `report_type="competitive_pressure"` reaches the report service
3. `MarketingCampaign.context` excludes non-marketing keys

**`PR-ContentOps-PostMergeAudit-1b: harden API surface`**

Separate from -1a so the architectural decision can land independently
from the API hardening:

- Pydantic request models for `/preview`, `/plan`, `/execute` (bounds
  payload size, locks schema, validates numeric ranges)
- Sanitize `errors[*].reason` at the API boundary (replace bare
  `str(exc)` with stable error codes; full reason logged server-side)
- Map `partial` / `failed` execution status to appropriate HTTP codes
  (or document the explicit choice)
- Wrap provider calls in try/except so provider exceptions don't
  propagate as bare 500s
- `_scope_from_value` strips empty/whitespace from `allowed_vendors`
  / `roles` (security boundary)
- Test pinning auth dependencies actually fire when configured

The MINORs and NITs ride along as "incidental cleanup" or stay
deferred — author's call.

---

## What I did not review yet

- The parse-retry behavior change applied to the four already-audited
  generators (`campaign_generation.py`, `report_generation.py`,
  `landing_page_generation.py`, `sales_brief_generation.py`). Silent
  change to code that was previously audited as "no validation
  feedback retry, deliberate v0 simplicity." Need to verify the new
  default doesn't break cost ceilings or test pinning.

This doc will be appended as that audit completes.
