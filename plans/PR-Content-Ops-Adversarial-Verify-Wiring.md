# PR-Content-Ops-Adversarial-Verify-Wiring

## Why this slice exists

Operating-model slice 5 (#1487, merged) landed the adversarial-pass data model
and the calibration library as pure types, but both are dark -- nothing in a
live path consumes them. The marketer verify surface (#1353) currently accepts
brief evidence, claims, coverage, and free-form comments, and returns a
deterministic verdict; there is no structured way for the marketer's own
second-pass review (the doc's "annoying second reviewer") to enter that verdict.
This slice wires slice 5b's adversarial pass into the verify flow end to end:
the connector's model produces the findings (verify, not generate -- no
server-side LLM), and the server folds them deterministically into the
Content-PR as categorized, never-blocking evidence the editor sees. It is the
thinnest increment that makes slice 5 visible to a marketer.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Host: add `adversarial_passes` to `ContentOpsReviewRequest` and fold each
   pass's substantiated findings into the Content-PR comments via slice 5b's
   `comment_from_finding` before the verdict is computed.
2. MCP: accept an `adversarial_passes` tool arg on `verify_draft`, parse decoded
   rows into `AdversarialPass` / `AdversarialFinding`, and thread it through --
   including through the ChatGPT search/fetch adapter (it is exposed in that
   contract as an optional field).
3. Harden `comment_from_finding` so a blank/decoded-None category renders an
   `[adversarial:uncategorized]` prefix instead of a malformed `[adversarial:]`.
4. Park the StrEnum-shim harmonization follow-up (from #1487 review) in
   `HARDENING.md`.
5. Archive the merged #1487 plan doc.

### Files touched

- `HARDENING.md`
- `atlas_brain/_content_ops_review_workflow.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `extracted_content_pipeline/adversarial_pass.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Adversarial-Verify-Wiring.md`
- `plans/archive/PR-Content-Ops-Calibration-Library.md`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_extracted_content_adversarial_pass.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- A submitted adversarial pass's substantiated findings appear in the verify
  result as categorized comments; unsubstantiated findings (no message or no
  evidence) are dropped, not surfaced.
- Folded adversarial comments are never blocking, so they never change a
  BLOCKED/REVISION_REQUIRED/APPROVED verdict on their own (evidence, not judge).
- Explicit `comments` and adversarial-derived comments coexist; explicit
  comments keep their order, adversarial comments follow.
- The MCP parser tolerates decoded input: missing/blank fields, non-list
  payloads, and unknown category strings do not raise.
- No server-side LLM, DB, or new network call is added.

Affected surfaces: the marketer verify host service and MCP tool only. No
change to the registry reader, OAuth, transport, or the extracted package.

Risk areas: comment ordering; the never-blocking invariant; decoded-input
tolerance in the new parser.

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

`ContentOpsReviewRequest` gains `adversarial_passes: tuple[AdversarialPass, ...]`.
A host helper converts those passes to comments: for each pass it takes
`pass.substantiated()` (slice 5b's filter for findings carrying both an
objection and evidence) and maps each through `comment_from_finding`, which
already hard-codes `blocking=False` and routes voice slips to the brand-rule
lane and everything else to editorial judgment. `run_content_ops_review` and
the blocked-result path build the Content-PR comments as
`request.comments + adversarial_comments`, so the existing verdict logic is
unchanged: because the folded comments are non-blocking, `blocking_comments`
never includes them and the verdict moves only on the marketer's own blocking
comments, failing coverage, or blocking claims. The findings surface in
`as_dict` through the existing `content_pr.comments` serialization.

The MCP `verify_draft` tool gains an `adversarial_passes` argument. A parser
builds `AdversarialPass`/`AdversarialFinding` from decoded dict rows, coercing a
known category string to `AdversarialFindingCategory` and leaving an unknown one
as the raw string (which `comment_from_finding` already tolerates via its
category-to-lane `.get` fallback). The host stays the single place review logic
lives; the MCP layer only decodes transport.

## Intentional

- **Findings never block.** The doc is explicit the adversarial pass is evidence
  for the accountable editor, not a judge; reusing slice 5b's never-blocking
  seam keeps that invariant unbypassable here.
- **Only substantiated findings are folded.** An empty/decoration finding would
  add a bare `[adversarial:x]` comment with no objection; `substantiated()`
  keeps the verify result signal-dense.
- **No corroboration surfacing or calibration anchors in this slice.** Both are
  real follow-ups (see Deferred); this slice is the thin wiring that makes the
  second pass enter the verdict at all.
- **Category coercion is lenient.** An unknown category string is preserved
  rather than rejected, matching the package's decoded-input tolerance; the
  comment still lands in the editorial-judgment lane.

## Deferred

- **Corroboration surfacing (6b):** report categories raised by two or more
  independent passes (the strongest signal) in the verify result. Needs a
  package helper that generalizes slice 5b's pairwise `corroborated_categories`
  to N passes; left out to keep this slice thin.
- **Calibration-anchor attachment (6b):** for each fired finding category,
  attach the matching calibration-library anchors so the marketer sees a worked
  example of that failure mode. Needs a tenant calibration-library reader port,
  symmetric to the claim registry reader.
- **Parked hardening:** StrEnum-shim harmonization across `review_contract.py`
  and `calibration_library.py` to the hardened `__str__ = str.__str__` form
  (from #1487 review; inert today) -- logged in `HARDENING.md`.

## AI reconciliation

- [chatgpt-codex-connector] atlas_brain/mcp/content_ops_marketer_verify_server.py:361
  "Thread adversarial passes through ChatGPT adapter" - FIXED: the adapter now
  passes `request_payload.get("adversarial_passes")` into the review request and
  exposes it in the contract (accepted_fields + schema properties + example) as
  an optional field; a test asserts a JSON submission's findings reach the
  cached verdict.
- [reviewer NIT] atlas_brain/mcp/content_ops_marketer_verify_server.py:703
  "blank/missing category renders a malformed [adversarial:] prefix" - FIXED in
  `comment_from_finding` (covers all callers, not just the parser): a blank
  category renders `[adversarial:uncategorized]`. Negative fixture added.

All findings fixed or waived: yes.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest of the adversarial, verify, host-workflow, and launcher test
  files -- 98 passed (adapter-threading + contract-optional + blank-category
  sentinel fixtures added this round; no regression).
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 167 enrolled.
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 11 |
| `atlas_brain/_content_ops_review_workflow.py` | 40 |
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 54 |
| `atlas_brain/mcp/content_ops_marketer_verify_server.py` | 91 |
| `extracted_content_pipeline/adversarial_pass.py` | 3 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Adversarial-Verify-Wiring.md` | 156 |
| `plans/archive/PR-Content-Ops-Calibration-Library.md` | 0 |
| `tests/test_atlas_content_ops_review_workflow.py` | 123 |
| `tests/test_extracted_content_adversarial_pass.py` | 11 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 128 |
| **Total** | **620** |
