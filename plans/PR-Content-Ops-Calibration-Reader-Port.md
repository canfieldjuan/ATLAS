# PR-Content-Ops-Calibration-Reader-Port

## Why this slice exists

Slice 7 (#1489) surfaces calibration anchors in the verdict, but they are
connector-supplied per request -- the marketer must resend the whole calibration
library on every `verify_draft` call. The agreed production path is a
server-side, tenant-scoped source of anchors, symmetric to the claim-registry
reader. This is **slice A** of that build: the read port, the merge semantics,
and graceful degradation, wired end to end with a fake/no-op reader and **no
database**. Persistence (the Postgres-backed repository + migration) and an admin
write surface are the following slices. Landing the seam and its behavior first
keeps the schema lift isolated and lets the merge/degradation contract be proven
in tests before any DB exists.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Host: add a `TenantCalibrationLibraryReader` Protocol; `run_content_ops_review`
   and `run_content_ops_review_for_bound_tenant` accept an optional
   `calibration_reader`; merge server-side anchors with request-supplied ones
   (union, server-first, dedup by `example_id`); any reader failure degrades to
   request-supplied anchors and never blocks.
2. MCP: `_get_calibration_reader()` + `_calibration_reader_override`, defaulting
   to a no-op (empty) reader until persistence lands; threaded into the verify
   tool and the ChatGPT adapter's review call.

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Calibration-Reader-Port.md`
- `plans/archive/PR-Content-Ops-Corroboration-Surfacing.md`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- A server-side reader's anchor surfaces in the verdict without being in the
  request.
- Merge is union, server-first: a server and request anchor sharing an
  `example_id` resolves to the server one; distinct ids keep both.
- A reader that raises degrades to request-supplied anchors; the verdict is
  unaffected and nothing propagates out of the review.
- `calibration_reader=None` reproduces slice-7 (request-supplied only) behavior.
- The default MCP reader returns nothing (no server anchors until persistence).

Affected surfaces: the review host service, the verify MCP tool, and the ChatGPT
adapter's review call. No schema, no new transport, no verdict-logic change. The
registry reader, OAuth, and tool args are untouched.

Risk areas: the failure-must-not-block divergence from the registry reader;
merge precedence; the blocked-result path staying request-only.

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

`TenantCalibrationLibraryReader` mirrors `TenantClaimRegistryReader` with one
documented divergence: anchors are evidence, not a gate, so the review treats any
reader exception as "no server-side anchors available" rather than a hard error.
`_merged_calibration_examples` reads the reader (when present), and on success
unions `server_examples + request.calibration_examples`, deduping by `example_id`
(first wins -> server wins). On any exception it logs a warning and returns the
request-supplied examples. `run_content_ops_review` selects anchors from the
merged set on the success path; the blocked-result path stays synchronous and
request-only (its scopes are either invalid or a hard failure, so consulting the
server is pointless -- anchors still surface from the request, matching slice 7).

The MCP server's `_get_calibration_reader()` returns a process-wide
`_EmptyCalibrationLibraryReader` (yields `()`), with a `_calibration_reader_override`
test seam, exactly like `_get_registry_reader()`. The verify tool and the ChatGPT
adapter both pass it into the review, so the seam is exercised on both transports
and persistence (slice B) is a drop-in swap of this one function.

## Intentional

- **Reader failure degrades, never blocks** -- the deliberate asymmetry from the
  registry reader (whose failure blocks because claims gate publishing).
  Hard-coding the degrade path here is the whole point of the slice.
- **Union, server-first, dedup by example_id** -- the tenant's curated set is
  canonical, and the connector can still supplement with new ids (slice 7's
  request path stays additive).
- **Blocked-result path stays request-only and synchronous** -- no scope to read
  by, or a hard failure already; surfacing request anchors there matches slice 7
  and avoids making `_blocked_result` async for no benefit.
- **Default reader is a no-op, not a DB call** -- slice A ships the seam; the
  empty default means verify behaves exactly as before until slice B swaps in
  the repository.

## Deferred

- **Slice B -- persistence:** migration `content_ops_calibration_library`
  (mirroring table 334) + `ContentOpsCalibrationLibraryRepository` implementing
  this port; `_get_calibration_reader()` swaps to it. Carries the label
  CHECK<->enum drift test and the teachable-only-at-DB decision.
- **Slice C -- admin write surface:** create/update/archive per tenant; deferred
  per the scoping decision (shared gap with the claim registry's unwired write).
- **Parked hardening:** none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest of the host-workflow, verify-MCP, and launcher-contract test
  files -- 103 passed (5 host + 2 MCP reader fixtures new).
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 168 enrolled.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_review_workflow.py` | 87 |
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 1 |
| `atlas_brain/mcp/content_ops_marketer_verify_server.py` | 24 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Calibration-Reader-Port.md` | 115 |
| `plans/archive/PR-Content-Ops-Corroboration-Surfacing.md` | 0 |
| `tests/test_atlas_content_ops_review_workflow.py` | 107 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 37 |
| **Total** | **374** |
