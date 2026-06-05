# PR: Content Ops Cache Diagnostics UI

## Why this slice exists

The Content Ops usage card now shows spend, savings, cache hits, and tokens, but
operators still cannot see why the cache applied or skipped from the UI. The
backend read model for cache diagnostics landed in PR #1031; this slice prepares
the Intel UI to consume that field without breaking against older deployed
responses that do not include the new key yet.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add optional cache diagnostic rows to the usage-summary wire contract.
2. Map cache mode, reason, lookup result, and store result into the Content Ops
   domain summary shape.
3. Default missing backend cache diagnostics to an empty list so the UI stays
   compatible with older deployed responses.
4. Render the top cache diagnostic rows in the existing Content Ops usage card.
5. Add focused UI contract tests for mapping and rendering strings.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Diagnostics-UI.md` | Plan doc for the cache diagnostics UI slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Extend the usage summary response type with optional cache diagnostic labels. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add domain fields for cache diagnostic breakdown rows. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map optional cache diagnostic rows and default missing data to an empty list. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Render cache diagnostic rows in the existing usage summary card. |
| `atlas-intel-ui/scripts/content-ops-usage-summary.test.mjs` | Pin mapping and source-level UI rendering contract. |

## Mechanism

`ContentOpsUsageSummaryResponse` accepts an optional `by_cache_status` list
whose rows reuse the existing breakdown payload and add optional cache labels:
mode, reason, lookup result, and store result. The domain mapper converts those
labels to camelCase and uses `wire.by_cache_status ?? []` so old backend
responses keep rendering.

The existing usage card remains the only UI surface. When cache diagnostics are
available, it shows the top three rows by backend sort order with a compact
label, call count, spend, and savings. If an older backend response omits the
field, the section is omitted.

## Intentional

- This is a frontend-only compatibility slice. It does not change backend
  usage-summary behavior.
- This does not add a new route or separate cache dashboard; the diagnostics
  live with the existing 7-day usage card.
- This renders top rows only so the card remains scannable.
- The wire field remains optional for deployment-order compatibility.

## Deferred

- Future PR: deeper cache diagnostics view if operators need more than the top
  three rows.
- Future PR: persisted tenant cache defaults if operators need always-on cache
  posture.
- Parked hardening: none. Root `HARDENING.md` was scanned; the current parked
  item belongs to the FAQ lane, not this cost-surfacing UI slice.

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-usage-summary — 4 passed.
- npm --prefix atlas-intel-ui run lint — passed.
- npm --prefix atlas-intel-ui run build — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <git-path>/codex-pr-bodies/cache-diagnostics-ui.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Wire/domain mapper | ~25 |
| Usage card UI | ~45 |
| Tests | ~25 |
| **Total** | **~180** |

This stays below the 400 LOC soft cap.
