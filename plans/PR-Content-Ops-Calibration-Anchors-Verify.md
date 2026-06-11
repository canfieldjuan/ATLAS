# PR-Content-Ops-Calibration-Anchors-Verify

## Why this slice exists

Slice 6 (#1488) wired the adversarial pass (5b) into the live verify verdict,
but the calibration library (5a, #1487) is still completely dark -- nothing in a
live path reads it. The calibration library is the operating-model doc's
"missing anti-drift piece": curated worked examples that anchor a reviewer on a
failure mode. This slice makes it do real work: when an adversarial finding
fires (e.g. `overclaim`), the verify result surfaces the marketer's curated
calibration anchor for that failure mode, so the editor sees a worked example
of what tripped the draft -- not just a bare objection. The connector supplies
its curated anchors (verify, not generate -- no server-side LLM); the server
deterministically selects which apply.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Package: add `calibration_anchors.py` -- a partial finding-category to
   calibration-label map plus `anchors_for_finding_categories`, which selects
   the teachable anchors illustrating the fired finding categories.
2. Host: `ContentOpsReviewRequest` gains `calibration_examples`; the result
   gains `calibration_anchors`, computed from the fired finding categories, and
   surfaced in `as_dict`.
3. MCP: `verify_draft` accepts a `calibration_library` arg, parsed into
   `CalibrationExample` rows and threaded through -- including through the
   ChatGPT search/fetch adapter (exposed in that contract as an optional field).

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `extracted_content_pipeline/calibration_anchors.py`
- `extracted_content_pipeline/manifest.json`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Calibration-Anchors-Verify.md`
- `plans/archive/PR-Content-Ops-Adversarial-Verify-Wiring.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_extracted_content_calibration_anchors.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- An anchor surfaces only when a *mapped* finding category fired and the anchor
  is teachable; an anchor for an unmapped finding category, or a non-teachable
  anchor, never surfaces.
- A good-voice (or any non-fired-mode) anchor never surfaces just because it was
  supplied -- selection is driven by the fired finding categories.
- Anchors are de-duplicated by example_id and ordered by first fired category.
- Anchors surface on a BLOCKED verdict too (evidence for the editor regardless).
- The finding-to-label map is partial and conservative: only unambiguous
  correspondences (overclaim, voice_slip) map.
- The MCP parser and the ChatGPT adapter both tolerate decoded input and thread
  the field through; no server-side LLM/DB/network is added.

Affected surfaces: the marketer verify host service, the verify MCP tool, and
the ChatGPT adapter. No change to the registry reader, OAuth, or transport.

Risk areas: the finding-to-label mapping correctness; teachable/dedup filtering;
adapter threading (the slice-6 silent-drop class).

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

`calibration_anchors.py` holds `_FINDING_LABEL_MAP` (overclaim -> OVERCLAIM,
voice_slip -> VOICE_DRIFT), `label_for_finding_category` (value-based, so a
decoded string resolves), and `anchors_for_finding_categories(library,
categories)`. The selector maps each fired category to a label, collects the
library's teachable examples for that label, and de-duplicates by example_id,
preserving first-fired order. Unmapped categories and non-teachable anchors
contribute nothing.

In the host, `_calibration_anchors_for_request` builds a `CalibrationLibrary`
from the connector-supplied `calibration_examples`, gathers the fired categories
from the substantiated findings across all adversarial passes, and runs the
selector. Both the verdict path and the blocked path attach the result, and
`as_dict` serializes each anchor (id, label, excerpt, reasoning, source). When
no anchors are supplied or none map, the result is simply empty.

The MCP `verify_draft` tool gains a `calibration_library` arg with a
`_calibration_examples` parser (label coerced to the enum when known, kept as
text otherwise). The ChatGPT adapter threads `calibration_library` into the
review request and lists it in its contract (accepted_fields + schema +
example) as optional, so a JSON submission's anchors are not silently dropped.

## Intentional

- **Connector-supplied anchors, not a tenant reader port.** This keeps the slice
  thin and consistent with verify-not-generate (the marketer brings their
  curated set). A server-side tenant calibration-library reader (symmetric to
  the claim registry reader) is the production-hardening follow-up.
- **The finding-to-label map is deliberately partial.** Only overclaim and
  voice_slip have an unambiguous calibration label; forcing the others
  (ambiguity, missing_proof, ...) onto a label would surface misleading anchors.
- **Selection is driven by fired categories, never by what was supplied.** A
  supplied good-voice anchor stays dormant until a finding maps to it, so the
  result never pads itself with irrelevant examples.
- **Anchors surface on BLOCKED too.** They are editor evidence, not a gate
  input, so withholding them when the verdict blocks would hide the most useful
  teaching moment.

## Deferred

- **Tenant calibration-library reader port:** a server-side, tenant-scoped
  source of curated anchors (symmetric to `TenantClaimRegistryReader`), so the
  connector need not resend the library on every call. Needs persistence.
- **Extending the finding-to-label map:** e.g. generic_stretch ->
  weak_persuasion, once the finding and calibration vocabularies are reconciled.
- **Corroboration surfacing (from #1488):** categories raised by >=2 passes.
- **Parked hardening:** none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest of the calibration-anchors, host-workflow, verify-MCP, and
  launcher test files -- 95 passed (new package, host, MCP, and adapter fixtures).
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 168 enrolled.
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: python3 extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_review_workflow.py` | 45 |
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 41 |
| `atlas_brain/mcp/content_ops_marketer_verify_server.py` | 70 |
| `extracted_content_pipeline/calibration_anchors.py` | 73 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Calibration-Anchors-Verify.md` | 128 |
| `plans/archive/PR-Content-Ops-Adversarial-Verify-Wiring.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_atlas_content_ops_review_workflow.py` | 53 |
| `tests/test_extracted_content_calibration_anchors.py` | 107 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 84 |
| **Total** | **608** |
