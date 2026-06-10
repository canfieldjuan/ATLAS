# PR-Content-Ops-Adapter-Contract-Example

## Why this slice exists

#1420 captured a live ChatGPT adapter usability gap: the adapter works
end-to-end, but its contract document lists only top-level field names. The
first real smoke submission therefore had to discover nested request shapes by
getting blocked on malformed coverage, missing quality-report flags, and missing
brand-voice flags.

This slice closes the narrow adapter contract/documentation gap before changing
the richer Claude verifier schema surface.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Product polish

1. Keep the ChatGPT adapter tool surface exactly `search` and `fetch`.
2. Keep current dispatch behavior: empty or non-JSON-string `query` returns the
   contract document; a JSON-encoded string whose decoded value is an object
   submits one review request.
3. Add a concrete valid review-request example and dispatch metadata to the
   contract document returned by `fetch`.
4. Prove the example can be submitted through `search` without the schema-shape
   blockers from #1420.

### Files touched

- `plans/PR-Content-Ops-Adapter-Contract-Example.md`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

- Acceptance criteria:
  - [ ] The adapter contract response keeps `accepted_fields` and adds a valid
        example payload.
  - [ ] The contract response describes the implicit dispatch rule without
        changing dispatch behavior.
  - [ ] The example payload uses backend-aligned nested field names for coverage,
        quality reports, brand voice, claims, comments, and rule packet fields.
  - [ ] Submitting the example through `search` avoids the #1420
        schema-shape blockers.
  - [ ] The ChatGPT adapter still exposes only `search` and `fetch`.
- Affected surfaces: Content Ops marketer ChatGPT adapter contract metadata.
- Risk areas: MCP contract compatibility, operator usability, backcompat.
- Reviewer rules triggered: R1, R2, R5

## Mechanism

Add package-local adapter contract helpers that return the accepted top-level
fields, dispatch description, and a minimal valid review-request example. The
example should include passing quality and brand-voice evidence so the contract
teaches the fields that live testing showed were missing.

The `search` dispatch remains intentionally unchanged for this slice.
JSON-encoded string payloads whose decoded value is an object keep submitting
review requests; empty, non-string, or non-JSON payloads keep returning the
contract search result.

## Intentional

- This does not add an explicit `mode` discriminator yet. That would be a
  behavior change for live ChatGPT adapter payloads and belongs in a later
  compatibility slice if needed.
- This does not change the rich Claude `verify_draft` parameter schema. The
  next planned slice can enrich that surface separately after this adapter
  contract is stable.
- This does not change tenant binding, OAuth, DCR, verdict caching, or tool
  names.
- AI reconciliation: fixed the reviewer MAJOR / Codex P2 by documenting that
  `query` is a JSON-encoded string, adding `submit_example_query`, and pinning
  the raw-object fallback case in tests.

## Deferred

- `PR-Content-Ops-Verify-Draft-Rich-Schema`: add richer nested schema hints to
  the Claude rich verifier tool while preserving decoded-input tolerance.
- Explicit adapter mode discriminator remains deferred unless live usage proves
  implicit dispatch is still confusing after the contract example lands.
- Parked hardening: none.

## Verification

- Passed: focused MCP adapter pytest for `tests/test_mcp_content_ops_marketer_verify.py` (34 passed after review fix).
- Passed: dedicated Content Ops MCP workflow pytest covering `tests/test_atlas_content_ops_review_workflow.py`, `tests/test_check_content_ops_marketer_verify_oauth_e2e.py`, `tests/test_content_ops_marketer_verify_launcher_contract.py`, `tests/test_start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py`, and `tests/test_mcp_content_ops_marketer_verify.py` (92 passed after review fix).
- Passed: extracted pipeline wrapper `scripts/run_extracted_pipeline_checks.sh` (3553 passed, 10 skipped after review fix).
- Passed: py_compile for `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`.
- Passed: `git diff --check`.
- Passed: local PR review with body file `tmp/content_ops_adapter_contract_example_pr_body.md`.

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Plan doc | 100 |
| Adapter contract helpers | 154 |
| Tests | 59 |
| Total | 313 |
