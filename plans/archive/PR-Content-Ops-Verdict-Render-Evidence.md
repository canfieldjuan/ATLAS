# PR-Content-Ops-Verdict-Render-Evidence

## Why this slice exists

Slices 6 (#1488) and 7 (#1489) surfaced the adversarial objections and the
calibration anchors in the *structured* verify payload, but the ChatGPT
adapter's human-readable verdict document -- the text a marketer actually reads
in the connector -- renders only `Decision` and `Reasons`. So everything those
two slices built is invisible on the ChatGPT path: the editor sees "Decision:
revision_required" with no sight of the objection that caused it or the worked
example that illustrates it. This slice renders that evidence into the
marketer-facing verdict text, and fixes `_verdict_title`, which reads a
non-existent top-level `asset_id` (the id is nested under `content_pr`) and so
always prints "draft".

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Product polish

1. `_verdict_text`: after Decision/Reasons, render an Objections section (the
   Content-PR comments) and a Calibration anchors section (label + excerpt +
   reasoning), each only when non-empty.
2. `_verdict_title`: read the asset id from `content_pr.asset_id`.

### Files touched

- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Verdict-Render-Evidence.md`
- `plans/archive/PR-Content-Ops-Calibration-Anchors-Verify.md`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- When the verdict carries comments, the text includes an Objections section
  listing each comment's category + message (+ evidence, + a BLOCKING marker).
- When the verdict carries calibration anchors, the text includes a Calibration
  anchors section listing each anchor's label + excerpt + reasoning.
- Both sections are omitted entirely when empty; the existing Decision/Reasons
  lines are unchanged (the `"Decision: approved"` assertion still holds).
- The title prints the real asset id, not "draft".
- The render tolerates malformed payload shapes (non-list comments/anchors,
  non-dict rows, missing fields) without raising.

Affected surfaces: the ChatGPT adapter's `_verdict_text` / `_verdict_title`
render only. No change to the verdict computation, the structured payload, the
verify MCP tool, or transport.

Risk areas: defensive rendering of decoded payload shapes.

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

`_verdict_text` keeps the `Decision` + `Reasons` block, then appends two
optional sections built by small guarded helpers. `_comment_lines` reads
`payload["content_pr"]["comments"]` and renders each comment with a message as
`[<category>] <message>` (a `[BLOCKING]` marker when blocking, and a trailing
`(evidence: ...)` when present); a comment with no message is skipped.
`_anchor_lines` reads `payload["calibration_anchors"]` and renders each as
`<label>: <excerpt> -- <reasoning>`; an anchor with no excerpt is skipped. Both
helpers fail closed on non-list / non-dict input, returning no lines.
`_verdict_title` reads `content_pr.asset_id` (falling back to "draft" only when
truly absent).

## Intentional

- **Product polish, not new behavior.** The structured payload already carries
  this evidence (slices 6/7); this slice only makes the ChatGPT-rendered
  document show it. No verdict logic changes.
- **Sections are omitted when empty** rather than printed as "Objections: none"
  -- a clean approval reads as a clean approval.
- **All comments render, blocking or not.** A blocking comment is the most
  important thing for the editor to see; it is marked, not hidden.

## Deferred

- Tenant calibration-library reader port (from #1489) -- still the production
  path for server-side anchors.
- Corroboration surfacing (from #1488).
- Parked hardening: none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest tests/test_mcp_content_ops_marketer_verify.py -- 51 passed
  (4 new render fixtures: objections+anchors, empty-sections omitted, malformed
  tolerance, blocking marker).
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 54 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Verdict-Render-Evidence.md` | 93 |
| `plans/archive/PR-Content-Ops-Calibration-Anchors-Verify.md` | 0 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 81 |
| **Total** | **231** |
