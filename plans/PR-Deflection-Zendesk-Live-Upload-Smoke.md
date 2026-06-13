# PR-Deflection-Zendesk-Live-Upload-Smoke

## Why this slice exists

PR #1523 shipped the Zendesk full-thread upload handoff: browser -> private
Blob -> portfolio submit proxy -> ATLAS multipart `json_file` -> the #1520
full-thread importer. The operator still needs a repeatable smoke command that
proves the same harness can exercise JSON/full-thread, not only CSV.

This turns the existing `portfolio-ui` submit smoke into Zendesk-path
functional validation while preserving CSV behavior. It adds full-thread mode,
a real Zendesk-shaped local fixture, route-handler `importer_mode="full_thread"`
for private JSON Blob pathnames, and the #1523 plan archive housekeeping.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Extend `smoke:deflection-submit-live` with `--importer-mode full_thread`
   / `ATLAS_DEFLECTION_IMPORTER_MODE=full_thread`.
2. Support a local Zendesk JSON fixture source using the committed full-thread
   fixture rather than a hand-built payload.
3. Make route-handler smoke payloads carry `importer_mode="full_thread"` only
   for private JSON Blob full-thread submissions.
4. Add CI-facing shell-test assertions for full-thread preflight, route-handler
   request shape, local fixture source mode, and invalid mode/source
   combinations.
5. Archive the merged #1523 plan doc and refresh `plans/INDEX.md`.

### Review Contract
- Acceptance criteria:
  - [ ] Existing CSV smoke behavior remains unchanged: local fixture default,
        route-handler CSV body, and HTTPS/account/token preflight checks.
  - [ ] Full-thread smoke mode uses the committed Zendesk full-thread JSON
        fixture by default and returns a distinct source mode for proof
        artifacts.
  - [ ] Route-handler full-thread mode sends `blob_pathname`,
        `support_platform="zendesk"`, support metadata, `limit`, and
        `importer_mode="full_thread"` without buyer headers or browser secrets.
  - [ ] Invalid combinations fail closed before calls: full-thread with a CSV
        pathname, CSV mode with a JSON pathname, unsupported importer modes,
        and route-handler local-file options.
  - [ ] The existing enrolled `test:deflection-upload-shell` script covers the
        new smoke branches; no new unenrolled npm script is introduced.
- Affected surfaces: portfolio live-submit smoke CLI, the enrolled portfolio
  upload-shell test, plan archive/index.
- Risk areas: false-green live smoke, mode/path confusion, accidental secret
  output, CI enrollment drift.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Deflection-Zendesk-Live-Upload-Smoke.md`
- `plans/archive/PR-Deflection-Zendesk-Upload-Handoff.md`
- `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The smoke script gains an `importerMode` option with allowed values `csv` and
`full_thread`. CSV remains the default. In local-fixture mode, CSV continues to
use the current sample CSV or `--csv-file`; full-thread mode reads
`tests/fixtures/zendesk_full_thread_seed_sample.json` by default or a
user-provided `--json-file`. The Blob reader returns the right content type and
the payload sent to `submitPrivateBlob` includes `importer_mode="full_thread"`
only for the full-thread path.

For route-handler smoke, the request body gains `importer_mode` only in
full-thread mode. Validation checks the blob pathname extension against the
selected importer mode so the command fails before touching Blob or ATLAS when
the mode/path combination is wrong.

The enrolled shell test imports the same smoke helpers and asserts both source
modes, route payload shape, and fail-closed validation messages.

## Intentional

- This does not fetch directly from the Zendesk API. It validates the uploaded
  JSON/private Blob handoff that #1523 shipped; credentialed API export remains
  a separate product slice.
- The smoke command still rejects local `http://localhost` ATLAS URLs because
  it is meant to prove deployed/hosted wiring, not local development plumbing.
- No new npm script is added. The existing enrolled
  `test:deflection-upload-shell` protects the new smoke helper branches.

## Deferred

- Direct Zendesk API export/import using tenant-scoped credentials.
- Browser-automated upload flow against the operator's private Blob/Zendesk
  trial data once a stable live fixture pathname is available.
- A dedicated JSON preflight diagnostics route if preview counts are needed
  before private upload.

Parked hardening: none.

## Verification

- `cd portfolio-ui && npm run test:deflection-upload-shell`
  - 29 passed.
- `cd portfolio-ui && npm run build`
  - Passed after `npm ci` installed this worktree's local dependencies. Vite
    emitted the existing large-chunk and sitemap-env warnings.
- `python scripts/archive_plans.py index`
  - Refreshed `plans/INDEX.md`; diff shows #1523 plan moved to archive.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Zendesk-Live-Upload-Smoke.md` | 117 |
| `plans/archive/PR-Deflection-Zendesk-Upload-Handoff.md` | 0 |
| `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs` | 83 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 196 |
| **Total** | **399** |
