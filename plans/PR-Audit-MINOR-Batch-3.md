# PR-Audit-MINOR-Batch-3: document the assumptions

## Why this slice exists

Three audit MINORs that share a "document-the-assumption" shape.
The fixes are docstring updates -- no behavior change. Scoping
keeps the diff focused on three closely-related findings rather
than a grab-bag.

1. **MINOR -- ``accumulate_usage`` assumes per-call usage from the
   LLM client.** Helper accumulates ``input_tokens`` across retry
   attempts under the assumption that each ``llm.complete()`` call
   returns *just that call's* usage. Some LLM clients return
   *cumulative* session usage; on those, retries double-count.
   Failure mode: inflated ``metadata.generation_usage.input_tokens``.
   Doesn't break anything upstream, but operators debugging cost
   reports will be confused.
2. **MINOR -- ``_landing_page_config_for_request`` discards the
   request.** Other ``_*_config_for_request`` helpers consume
   ``request.inputs`` and ``request.limit``; this one
   ``del request`` -s and returns defaults. Confusing to readers
   comparing the helpers; documenting the asymmetry closes the
   audit-trail gap. (Acting on the request would mean adding a
   ``limit`` field to ``LandingPageGenerationConfig`` -- but
   landing pages are per-campaign single-shot, so the field
   wouldn't have a sensible meaning.)
3. **MINOR -- Host service concurrency assumption is undocumented.**
   ``execute_content_ops_request`` calls
   ``asyncio.gather(*step_executions)`` -- host-injected services
   run concurrently. A host that built their service against the
   old single-step path may not realize this; if their service
   maintains in-memory state, they get silent races.

## Scope (this PR)

Pure documentation. No behavior change. No new tests (these are
docstring updates -- existing tests already lock the behavior).

### Fix 1: ``accumulate_usage`` docstring

Append a note to the existing helper's docstring about the per-call
usage assumption.

### Fix 2: ``_landing_page_config_for_request`` docstring

Replace ``del request`` with a short docstring explaining why the
request is intentionally discarded for this output.

### Fix 3: ``ContentOpsExecutionServices`` / ``execute_content_ops_request`` docstring

Add a docstring note to ``execute_content_ops_request`` (the public
entry point) about concurrent service invocation and the
re-entrancy expectation. Also touch the
``ContentOpsExecutionServices`` dataclass docstring so hosts see it
when constructing their service bundle.

## Intentional (looks wrong but is deliberate)

- **No new tests.** Three docstring updates; behavior is unchanged.
  Adding tests that assert "the docstring contains certain text"
  tests prose, not contract. The audit's request was to surface
  the assumptions, not to add behavior.
- **No cumulative-usage detection on ``accumulate_usage``.** The
  audit suggested "detect cumulative usage by checking if
  ``input_tokens`` is monotonically increasing" as an alternative
  to the docstring fix. Detection adds heuristic logic that could
  itself be wrong (a single retry with the same prompt may emit
  the same token count both times). The docstring path is
  cheaper, no chance of false positives, and surfaces the
  assumption to hosts -- which is what the audit actually flagged.
- **No new ``limit`` field on ``LandingPageGenerationConfig``.**
  Landing pages are per-campaign single-shot; ``limit`` doesn't
  apply. Documenting why is the right move.

## Deferred (still on purpose)

- ``for_output`` if/elif chain (audit NIT).
- ``describe_control_surfaces`` calls execution-services per
  request (audit MINOR) -- needs a separate caching design.
- ``topic`` for blog_post.
- ``PR-Campaign-Config-V2``.

## Verification

- ``python -c "from extracted_content_pipeline.services._parse_retry_helpers import accumulate_usage"`` -> imports
- ``python -c "from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices, execute_content_ops_request"`` -> imports
- ``python -c "from extracted_content_pipeline.generation_plan import build_generation_plan"`` -> imports
- Existing tests on the touched files still pass (no behavior change).
