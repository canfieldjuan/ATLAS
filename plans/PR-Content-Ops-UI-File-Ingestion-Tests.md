# PR-Content-Ops-UI-File-Ingestion-Tests

## Why this slice exists

PR #865 routed selected ingestion files through the new server-side upload
endpoints, while keeping pasted/manual rows on the deprecated inline fallback.
The reviewer called out that the new UI path did not have a direct frontend
test proving the split. This slice adds that regression coverage without
changing production behavior.

## Scope (this PR)

Ownership lane: content-ops/ui-file-ingestion-tests

1. Add a focused adapter-level Node test for Content Ops ingestion routing.
2. Prove uploaded files use multipart `/ingestion/files/inspect` and
   `/ingestion/files/import`.
3. Prove inline rows still use JSON `/ingestion/inspect` and
   `/ingestion/import`.
4. Add an npm script so the test can run independently.

### Files touched

- `plans/PR-Content-Ops-UI-File-Ingestion-Tests.md`
- `atlas-intel-ui/scripts/content-ops-ingestion-routing.test.mjs`
- `atlas-intel-ui/package.json`

## Mechanism

The test transpiles `src/api/contentOps.ts` in-process, stubs the API base,
local storage, and `fetch`, then invokes the public adapter functions. Captured
fetch calls assert the URL, method, body type, headers, and important request
fields.

This avoids a brittle DOM-level form test while still proving the production
adapter will send `File` payloads over multipart requests and inline rows over
JSON.

## Intentional

- No React component test in this slice. The route split lives in the API
  adapter, and PR #865 already made the screen call those adapter functions.
- No source-code changes. This is a regression test for shipped behavior.
- The test stubs imports from auth/config because it is verifying request
  construction, not auth refresh behavior.

## Deferred

- Parked hardening considered: `FAQSTRESS-1` and `FAQSTRESS-2` in
  `HARDENING.md`. They remain parked because they are hosted-runtime scale and
  DB pressure work, not required for this UI adapter test slice. The merged
  execute concurrency gate partially mitigates `FAQSTRESS-2`, but a background
  job boundary or deploy-aware admission control remains a separate hardening
  slice.

## Verification

- Passed: focused UI adapter routing test:
  `npm run test:content-ops-ingestion-routing` (`4 passed`).
- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| UI adapter test | ~250 |
| Package script | ~1 |
| **Total** | **~321** |
