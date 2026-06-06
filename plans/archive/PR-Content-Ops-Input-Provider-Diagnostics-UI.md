# PR: Content Ops Input Provider Diagnostics UI

## Why this slice exists

PR #938 surfaces input-provider diagnostics from the Content Ops preview, plan,
and execute API routes. The frontend contract still ignored that field, so a
support-ticket CSV truncated to the synchronous package cap would be visible in
raw JSON but not in the operator UI.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Product polish

1. Add typed wire and domain models for `input_provider` diagnostics.
2. Map provider warnings and allowlisted metadata through the wire-to-domain
   adapter.
3. Show a compact source-package block in preview, plan, and execution panels.
4. Update frontend contract fixtures and the backend API contract doc.

### Files touched

- `plans/PR-Content-Ops-Input-Provider-Diagnostics-UI.md`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/preview-can-run.json`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/plan-runnable.json`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`

## Mechanism

The API adapter exposes the backend's optional `input_provider` object. The
domain mapper keeps the provider name, operational metadata, and normalized
warnings while preserving any extra warning fields under `details`.

`ContentOpsNewRun` renders the diagnostics as a source-package section. No-op
providers remain invisible because the backend omits their diagnostics.

## Intentional

- No generation, package, ingestion, or FAQ behavior changes.
- No new upload product policy. The UI surfaces backend warnings; it does not
  decide whether large files should be rejected or queued.
- No raw ticket rows or host-injected metadata are displayed. The backend only
  exposes allowlisted operational metadata.

## Deferred

- Future PR: hosted upload policy can decide whether files above the 1,000-row
  synchronous package cap should be blocked, warned, or sent to a background
  job.
- Parked hardening: none. `HARDENING.md` was scanned; the current entry is FAQ
  scale/backpressure work owned by the FAQ generation lane.

## Verification

- JSON fixture validation with `python -m json.tool` - passed.
- `npm --prefix atlas-intel-ui run build` - passed.
- `npm --prefix atlas-intel-ui run lint` - passed.
- `git diff --check` - passed.
- `scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| API/domain types and mapper | ~60 |
| UI rendering | ~70 |
| Fixtures and docs | ~65 |
| **Total** | **~265** |
