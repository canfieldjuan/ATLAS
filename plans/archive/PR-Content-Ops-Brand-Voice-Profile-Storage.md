# PR-Content-Ops-Brand-Voice-Profile-Storage

## Why this slice exists

PR-Content-Ops-Brand-Voice-Profile shipped the executable request-time brand
voice contract, but it intentionally requires callers to include the full
inline `inputs.brand_voice` payload whenever `brand_voice_profile_id` is set.
That proves prompt injection and tenant scope checks, but it leaves operators
without the product path they need: save a tenant profile once, select it in
the Content Ops UI, and have preview/plan/execute resolve it safely.

This slice closes the explicitly deferred
`PR-Content-Ops-Brand-Voice-Profile-Storage` item from that plan. It keeps the
storage and CRUD surface in the Atlas host because profiles are authenticated
tenant data, while adding only a dependency-injected lookup hook to the
extracted control-surface router so standalone extraction remains lightweight.

The PR is expected to exceed the 400 LOC soft cap because the usable path is
cross-layer: host migration/repository, host CRUD routes, extracted
preview/plan/execute lookup, frontend API/domain mapping, selector UI, and the
CI-enrolled frontend test have to land together. Splitting before the selector
would create unreachable storage; splitting before the lookup would create a
selector that sends an ID the backend still fails closed on.

## Scope (this PR)

Ownership lane: content-ops/brand-voice-profile-storage
Slice phase: Vertical slice

1. Add tenant-scoped host storage for saved brand voice profiles with soft
   delete, list, get, create/update, and display-safe DTO conversion.
2. Mount authenticated CRUD/list routes at
   `/api/v1/content-ops/brand-voice-profiles`, mirroring the existing
   Zendesk-credential route shape and admin write guard.
3. Add an extracted control-surface lookup provider that resolves
   `brand_voice_profile_id` into `inputs.brand_voice` for preview, plan, and
   execute before the existing executor gate runs.
4. Fail closed when a selected stored profile is missing, cross-tenant, or the
   host did not wire the lookup provider. Do not fall back to any shared/global
   profile.
5. Add frontend wire/domain types, API functions, and a selector on
   `ContentOpsNewRun` that sends only `brand_voice_profile_id`.
6. Add focused backend and frontend tests, including the exact intel-ui workflow
   enrollment step for the new frontend script.

### Files touched

- `plans/PR-Content-Ops-Brand-Voice-Profile-Storage.md`
- `atlas_brain/storage/migrations/333_content_ops_brand_voice_profiles.sql`
- `atlas_brain/_content_ops_brand_voice_profiles.py`
- `atlas_brain/api/content_ops_brand_voice_profiles.py`
- `atlas_brain/api/__init__.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_brand_voice_profiles.py`
- `tests/test_content_ops_brand_voice_profiles_api.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`

## Mechanism

Host storage owns rows keyed by `(account_id, id)`:

```sql
CREATE TABLE content_ops_brand_voice_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  descriptors JSONB NOT NULL DEFAULT '[]'::jsonb,
  exemplars JSONB NOT NULL DEFAULT '[]'::jsonb,
  banned_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
  preferred_pov TEXT,
  reading_level TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  archived_at TIMESTAMPTZ
);
```

The host repository normalizes profile payloads through the existing
`BrandVoiceProfile` helper, stores only bounded arrays, and returns display-safe
records. API writes require the same owner/admin role pattern used by Zendesk
credentials; reads use the authenticated account. Deletes are soft archives.

The extracted router gets a new optional provider:

```python
BrandVoiceProfileProvider = Callable[
    [TenantScope, str],
    BrandVoiceProfile | Mapping[str, Any] | None
    | Awaitable[BrandVoiceProfile | Mapping[str, Any] | None],
]
```

Before preview, plan, or execute, the router checks for
`brand_voice_profile_id` without inline `inputs.brand_voice`. If present, it
requires a tenant scope and provider, resolves exactly that profile for that
scope, injects the full normalized profile into `inputs.brand_voice`, and then
continues through the already-merged executor/generator path. A miss returns a
404/400-class failure rather than guessing or using defaults.

The UI loads profiles with `fetchContentOpsBrandVoiceProfiles()`, renders a
compact selector in the options area, and stores the selected ID on
`request.brandVoiceProfileId`. `toWireRequest()` maps it to
`brand_voice_profile_id`; the UI does not copy exemplars or banned terms into
the editable inputs JSON.

## Intentional

- No profile authoring UI in this slice. The API supports create/update/delete;
  the first UI surface is selection so the existing generation form can consume
  stored profiles without building a second editor workflow.
- Stored profiles are host-owned, not an extracted Postgres adapter. They are
  authenticated SaaS account data and reuse host `AuthUser`, `saas_accounts`,
  and admin write guards; the extracted router only gets a small lookup port.
- The executor still fails closed on bare `brand_voice_profile_id` when called
  directly without API lookup. That preserves the request-time contract for
  standalone callers and keeps stored lookup explicitly host-wired.
- No global preset/profile fallback. Missing tenant profiles are hard failures
  when selected.
- #1268 remains open and owned elsewhere; this PR does not touch output
  variations.

## Deferred

- `PR-Content-Ops-Brand-Voice-Profile-Editor`: full in-app create/edit form
  for descriptors, exemplars, banned terms, POV, and reading level.
- `PR-Content-Ops-Brand-Voice-Presets`: shipped descriptor-only preset library.
- `PR-Content-Ops-Brand-Voice-Onboarding`: scrape/upload samples and pre-fill a
  custom profile for human edit.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_brand_voice_profiles.py tests/test_content_ops_brand_voice_profiles_api.py -q` -- 17 passed.
- `pytest tests/test_extracted_content_control_surface_api.py -q` -- 131 passed, 1 skipped.
- `pytest tests/test_atlas_content_ops_generated_assets_api.py -q` -- 16 passed, 1 warning.
- `python -m pytest tests/test_atlas_content_ops_generated_assets_api.py tests/test_content_ops_brand_voice_profiles.py tests/test_content_ops_brand_voice_profiles_api.py -q` -- 33 passed, 1 warning.
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector` -- 6 assertions passed.
- `cd atlas-intel-ui && npm run build` -- passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 3105 passed, 10 skipped, 1 warning.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-brand-voice-profile-storage.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Brand-Voice-Profile-Storage.md` | 186 |
| `atlas_brain/storage/migrations/333_content_ops_brand_voice_profiles.sql` | 45 |
| `atlas_brain/_content_ops_brand_voice_profiles.py` | 328 |
| `atlas_brain/api/content_ops_brand_voice_profiles.py` | 212 |
| `atlas_brain/api/__init__.py` | 16 |
| `extracted_content_pipeline/api/control_surfaces.py` | 101 |
| `atlas-intel-ui/src/api/contentOps.ts` | 78 |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | 1 |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | 4 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 125 |
| `atlas-intel-ui/package.json` | 1 |
| `.github/workflows/atlas_intel_ui_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 14 |
| `scripts/run_extracted_pipeline_checks.sh` | 2 |
| `tests/test_content_ops_brand_voice_profiles.py` | 215 |
| `tests/test_content_ops_brand_voice_profiles_api.py` | 278 |
| `tests/test_extracted_content_control_surface_api.py` | 148 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 31 |
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 206 |
| **Total** | **1988** |

Expected final: 19 files, +1988 / -0. The overage is the
minimum useful vertical slice for stored profiles to be selectable and
generation-ready end to end.
