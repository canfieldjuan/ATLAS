# PR-Content-Ops-UI-File-Ingestion

## Why this slice exists

PR #861 added server-side file ingestion routes so large customer CSV/JSON/JSONL
exports no longer need to be expanded into inline browser JSON posts. The New
Run UI still parses loaded files into the `rows` textarea and submits them to
the now-deprecated inline ingestion endpoints, so customer ticket exports over
the inline cap can still fail from the UI.

This slice moves the file-loaded path to the new multipart file endpoints while
leaving pasted/manual rows on the deprecated inline path for compatibility.

## Scope (this PR)

Ownership lane: content-ops/ui-file-ingestion

1. Add frontend API wrappers for:
   - `POST /content-ops/ingestion/files/inspect`
   - `POST /content-ops/ingestion/files/import`
2. Keep the old inline wrappers available but document them as deprecated.
3. Update `ContentOpsNewRun` so loaded files are retained as `File` objects and
   submitted through the file endpoints.
4. Keep pasted JSON rows working through the inline compatibility path.

### Files touched

- `plans/PR-Content-Ops-UI-File-Ingestion.md`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`

## Mechanism

The API adapter builds `FormData` with the selected `File`, `source_rows`,
`source`, `target_mode`, `max_source_text_chars`, `sample_limit`,
`default_fields`, and import-only `replace_existing` / `dry_run`. It does not
set `Content-Type`, letting the browser provide the multipart boundary.

`ContentOpsNewRun` stores the loaded file separately from the manual rows
textarea. Inspect/import choose the file route when a file is selected and the
inline route otherwise. Editing the rows textarea clears the selected file and
returns the operator to the manual compatibility path.

## Intentional

- Manual pasted rows still use the deprecated inline endpoints because the UI
  still needs a low-friction row editor for small/debug payloads.
- No backend files are touched. PR #864 is open against backend control-surface
  files, so this slice avoids that ownership lane.
- No durable job/upload polling yet. This slice only wires the existing
  server-side file routes into the UI.

## Deferred

- Future PR: remove the inline UI path after a compatibility window.
- Future PR: add durable upload/job status once the backend grows a job model.
- Future PR: show backend upload limits from catalog/config instead of hardcoded
  route behavior.
- Parked hardening: none.

## Verification

- `npm run build` from `atlas-intel-ui/`
  - TypeScript build passed.
  - Vite production build passed.
- `npm run lint` from `atlas-intel-ui/`
  - Passed.
- `npm run test:content-ops-input-display` from `atlas-intel-ui/`
  - `2` tests passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| API wrappers | ~160 |
| New Run UI wiring | ~270 changed, mostly deleting client-side file parsing |
| **Total** | ~520 changed |

Over the 400 LOC target because the UI migration removes the old client-side
CSV/JSONL parser, adds multipart wrappers, and keeps manual inline rows as a
compatibility path in one coherent screen/API contract update.
