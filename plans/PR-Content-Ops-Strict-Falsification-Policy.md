# Content Ops Strict Falsification Policy

## Why this slice exists

The strict reasoning preset is the right place for falsification, but the host
must own the rules. Running an empty/default falsification policy would add
extra LLM calls without a concrete product policy.

## Scope (this PR)

1. Mark `multi_pass_strict` as the catalog preset that supports
   falsification.
2. Add host-owned falsification rule config to the Content Ops control-surface
   API config.
3. Wire those rules into `FalsificationPolicy` only for strict packaged
   reasoning.
4. Surface the strict preset's falsification capability in generation-plan
   metadata.
5. Preserve no-op behavior when the host does not supply falsification rules.
6. Refresh status/backlog/coordination docs for the shipped strict
   falsification seam.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `plans/PR-Content-Ops-Strict-Falsification-Policy.md`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_reasoning_policy.py`

## Mechanism

`ContentOpsControlSurfaceApiConfig` now accepts a sequence of falsification
rule mappings plus conservative/drop behavior. `_structured_reasoning_context`
creates `FalsificationPolicy` only when the selected preset advertises
falsification and rules were supplied by the host.
`generation_plan` exposes `reasoning_falsification` as capability metadata;
runtime still requires host-supplied rules before a falsification policy runs.
Rules are documented as host-owned predicate mappings. Falsification runs per
generated claim, so enabling rules can add one LLM call per claim. Config
validation rejects `drop_falsified=True` without rules and caps rule count at
20 to keep every per-claim falsification prompt bounded.

## Intentional

No hidden falsification defaults. `multi_pass_structured` ignores the
falsification rule config, and `multi_pass_strict` leaves
`falsification_policy` unset when the rule list is empty.

## Deferred

Continued `extracted_reasoning_core` work remains separate if reasoning is
sold as a stronger standalone layer.

## Verification

pytest tests/test_extracted_content_reasoning_policy.py
tests/test_extracted_content_control_surface_api.py
tests/test_extracted_content_generation_plan.py -> 94 passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| **Total** | ~190 |
