# PR: Content Ops Ingestion CSV File Load UI

## Why this slice exists

The New Run ingestion panel can load JSON, JSONL, and NDJSON files, but customer
exports commonly arrive as CSV. The backend and host CLI already support CSV
opportunity/source-row ingestion, so the browser UI should not force operators
to convert CSV exports before inspection or import.

## Scope (this PR)

Add browser-side CSV parsing to the existing ingestion file loader. CSV rows are
converted into the same array-of-object JSON shape already pasted into the
textarea, then the existing inspect/import API calls continue unchanged.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-CSV-File-Load-UI.md`

## Mechanism

- Extend the file picker accept list and visible label from JSON/JSONL to
  JSON/CSV.
- Route `.csv` files through a local CSV parser in `ContentOpsNewRun.tsx`.
- Treat the first CSV row as headers, reject empty or duplicate headers, skip
  blank body rows, and emit row objects using the header labels as keys.
- Preserve the existing request-id race guard, loading state, source label
  behavior, textarea handoff, and inspect/import API path.

## Intentional

- The parser handles commas, quoted fields, escaped double quotes, CRLF, LF, and
  quoted multiline values.
- Parsed CSV cell values remain strings because host exports are text-first and
  the backend normalizer already owns semantic coercion.
- The UI still shows the loaded rows in the textarea so operators can inspect or
  edit before running the API calls.

## Deferred

- No server-side file upload endpoint; this remains browser-side conversion into
  inline rows.
- No XLSX parser.
- No automatic source-row toggle based on CSV headers; operators still choose
  whether rows are source exports.

## Verification

- npm --prefix atlas-intel-ui ci
- npm --prefix atlas-intel-ui run build
- git diff --check
- bash scripts/local_pr_review.sh

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| New Run CSV file loader parsing and labels | ~85 |
| Docs/coordination/plan | ~60 |
| **Total** | ~145 |
