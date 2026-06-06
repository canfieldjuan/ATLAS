# PR-Content-Ops-Brand-Voice-Profile

## Why this slice exists

Issue #1276 captured a missing content-ops control: LLM-backed copy can be
generated from evidence, but the run has no brand voice/tone profile and no
safe way to preserve a customer's established voice with exemplars. The
social-post lifecycle dependency named in the issue is now merged, and the
card visual-export lane is complete, so this slice can add the first executable
brand-voice primitive without colliding with the open output-variations plan
#1268.

This PR intentionally keeps brand voice as a request-time profile contract:
normalize and scope-check the profile, inject its descriptors/exemplars into
the original grounded system prompt, and surface deterministic adherence
metadata. That gives operators a usable path while leaving profile-management
CRUD/UI for a follow-up.

The slice is over the 400 LOC soft cap because the first executable contract is
cross-layer by nature: request/API parsing, plan metadata, dispatcher routing,
four generator prompt hooks, and the failure-branch tests must land together.
Splitting before generator coverage would create an inert request field; splitting
before the fail-closed tests would leave the tenant-scope safety claim unproven.

## Scope (this PR)

Ownership lane: content-ops/brand-voice-profile
Slice phase: Vertical slice

1. Add a pure extracted-package `BrandVoiceProfile` primitive with bounded
   descriptor, exemplar, banned-term, reading-level, and POV fields.
2. Thread optional `brand_voice_profile_id` plus inline `brand_voice` profile
   payload through `ContentOpsRequest`, the API model, plan metadata, and the
   execution dispatcher.
3. Inject the normalized brand voice block into the system prompt for the LLM
   copy generators named by #1276: `email_campaign`, `blog_post`,
   `landing_page`, and `sales_brief`.
4. Validate tenant scope fail-closed when an inline profile carries an
   `account_id` that does not match `TenantScope.account_id`.
5. Add deterministic adherence metadata for explicit rules only: banned terms,
   first/second/third-person POV hints, and rough reading-level band.
6. Preserve brand voice on the reachable landing-page draft repair path by
   accepting a re-supplied full profile and applying it to repair generation.

### Files touched

- `plans/PR-Content-Ops-Brand-Voice-Profile.md`
- `extracted_content_pipeline/brand_voice.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/campaign_generation.py`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_brand_voice.py`
- `tests/test_extracted_campaign_generation.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_landing_page_generation.py`

## Mechanism

`brand_voice.py` owns the profile contract and prompt/audit helpers. The
request accepts:

```json
{
  "brand_voice_profile_id": "acme-main",
  "inputs": {
    "brand_voice": {
      "id": "acme-main",
      "account_id": "acct-1",
      "name": "Acme editorial",
      "descriptors": ["plainspoken", "operator-led"],
      "exemplars": ["Real customer copy sample..."],
      "banned_terms": ["synergy"],
      "preferred_pov": "second_person",
      "reading_level": "plain"
    }
  }
}
```

The executor resolves the inline profile once per request and passes it only to
LLM copy outputs. Each generator applies the profile to the existing system
prompt before the user prompt is created; there is no generate-then-restyle
post-pass, so evidence grounding stays in the original generation discipline.
Parsed artifacts get `_brand_voice_profile` and `_brand_voice_audit` metadata
when a profile is active.

Landing-page draft repair accepts an optional repair payload with
`brand_voice_profile_id` and inline `brand_voice`. The full profile must be
re-supplied for repair because the saved draft metadata is intentionally lossy
and omits exemplars/banned terms until stored profile lookup lands.

## Intentional

- No profile CRUD/API/UI in this PR. The vertical slice proves the generation
  contract first; stored profile management can mirror the existing generated
  asset table pattern in the next slice.
- No social-post voice support. #1276 correctly notes current social posts are
  deterministic template assembly, not LLM generation.
- No automated "voice match" score. This PR checks only explicit rule-shaped
  constraints and leaves actual voice fidelity to the human review queue.
- No attempt to reconstruct a full repair profile from saved draft metadata.
  The repair caller must re-supply the full profile until stored profile lookup
  lands.
- No output-variations work. #1268 remains open and owned elsewhere.

## Deferred

- `PR-Content-Ops-Brand-Voice-Profile-Storage`: account-scoped stored
  `brand_voice_profiles` repository, API CRUD/list endpoints, and UI selector.
- `PR-Content-Ops-Brand-Voice-Presets`: shipped descriptor-only preset library.
- `PR-Content-Ops-Brand-Voice-Onboarding`: scrape/upload samples and pre-fill a
  custom profile for human edit.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `pytest tests/test_support_ticket_provider_landing_blog_execute.py -q` --
  13 passed.
- `pytest tests/test_extracted_brand_voice.py tests/test_extracted_landing_page_generation.py tests/test_extracted_content_asset_api.py -q`
  -- 140 passed.
- `pytest tests/test_extracted_brand_voice.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_generation_plan.py tests/test_extracted_campaign_generation.py -q`
  -- 165 passed.
- `python -m py_compile extracted_content_pipeline/brand_voice.py extracted_content_pipeline/landing_page_generation.py extracted_content_pipeline/api/generated_assets.py extracted_content_pipeline/content_ops_execution.py`
  -- passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  -- 146 matching tests enrolled.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 295 reasoning-core tests
  passed, then 3081 extracted-content tests passed / 10 skipped.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-brand-voice-profile.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 1 |
| `extracted_content_pipeline/api/generated_assets.py` | 10 |
| `extracted_content_pipeline/blog_generation.py` | 29 |
| `extracted_content_pipeline/brand_voice.py` | 305 |
| `extracted_content_pipeline/campaign_generation.py` | 20 |
| `extracted_content_pipeline/content_ops_execution.py` | 81 |
| `extracted_content_pipeline/control_surfaces.py` | 10 |
| `extracted_content_pipeline/generation_plan.py` | 12 |
| `extracted_content_pipeline/landing_page_generation.py` | 28 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/sales_brief_generation.py` | 20 |
| `plans/PR-Content-Ops-Brand-Voice-Profile.md` | 179 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_brand_voice.py` | 123 |
| `tests/test_extracted_campaign_generation.py` | 38 |
| `tests/test_extracted_content_asset_api.py` | 46 |
| `tests/test_extracted_content_generation_plan.py` | 29 |
| `tests/test_extracted_content_ops_execution.py` | 115 |
| `tests/test_extracted_landing_page_generation.py` | 38 |
| **Total** | **1088** |

Expected final: 19 files, +1065 / -23. This is over the 400 LOC soft cap
because the request contract, four prompt-injection hooks, dispatcher routing,
reviewed repair-path wiring, new helper, and failure-branch tests are the
minimum executable vertical slice.
