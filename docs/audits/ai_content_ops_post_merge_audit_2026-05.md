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
- [ ] `extracted_content_pipeline/api/control_surfaces.py` (195 LOC) — pending
- [ ] `extracted_content_pipeline/blog_generation.py` (396 LOC) — pending
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
2. **MAJOR** — `content_ops_execution.py`: `MarketingCampaign.context`
   leaks unrelated request inputs (e.g., `report_type`, `opportunity_id`)
   into the landing-page LLM prompt.
3. **MAJOR** — `content_ops_execution.py`: `landing_page` is a
   hard-coded special case; every new asset with a non-standard
   `generate` signature requires another `if` branch.
4. **MAJOR** — `generation_plan.py`: landing-page step doesn't carry
   `MarketingCampaign` data — masked at runtime by the executor's
   `_marketing_campaign_from_inputs` reaching back into `request.inputs`.
5. **MAJOR** — `generation_plan.py`: step configs claim per-output
   config (channels, report_type, brief_type) that never reaches the
   service. Same root cause as #1.
6. **MAJOR** — `generation_plan.py`: `email_campaign` step emits
   `channels` (tuple) but `CampaignGenerationConfig` also has legacy
   `channel: str` field. Hidden config fields are a recurring drift
   symptom.
7. **MAJOR** — `control_surfaces.py`: silent preset-typo fallback to
   `email_only`. User types `preset="contmarket"`, gets email-only
   output, no diagnostic.
8. **MAJOR** — `control_surfaces.py`: `outputs` + `preset` both set:
   preset silently ignored. UI forms that send both think the preset
   is contributing.

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

The MINORs and NITs ride along as "incidental cleanup" or stay
deferred — author's call.

---

## What I did not review yet

- `extracted_content_pipeline/api/control_surfaces.py` (195 LOC) —
  FastAPI router on top of all this. Probably surfaces the BLOCKER to
  HTTP callers; worth knowing how.
- `extracted_content_pipeline/blog_generation.py` (396 LOC) —
  undeclared 6th asset type. Need to know if it's a supported product
  asset (rev the count to "6 of 5") or an internal seam.
- The parse-retry behavior change applied to the four already-audited
  generators (`campaign_generation.py`, `report_generation.py`,
  `landing_page_generation.py`, `sales_brief_generation.py`). Silent
  change to code that was previously audited as "no validation
  feedback retry, deliberate v0 simplicity." Need to verify the new
  default doesn't break cost ceilings or test pinning.

This doc will be appended as those audits complete.
