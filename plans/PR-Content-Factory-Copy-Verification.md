# PR-Content-Factory-Copy-Verification

## Why this slice exists

The #2116 Content Factory contract gave `EditorialAudit` a `copy_verification` field and
a rule that a draft may not be promoted unless `copy_verification.verdict == "pass"` --
but nothing PRODUCED that verdict, so the promote guard was a shape with no teeth. The
real deterministic gate (banned marketing claims + PII) existed only as the operator's
"Resolution Audit Draft Verifier" tool inside the Open WebUI database, un-versioned and
un-testable. This slice (plan Phase 4.1) ports that gate's blocker core into the repo as
a deterministic producer of the `CopyVerification` verdict, so a draft that overclaims or
leaks contact PII cannot be promoted.

### Problem-derived contract

A correct fix must:
- Produce a `CopyVerification` (verdict + hits) from draft text, deterministically.
- Fail the verdict when the copy contains a forbidden OUTCOME claim (guaranteed savings,
  fixed deflection %, ticket reductions), a forbidden AUTOMATION claim (auto-publishing /
  auto-answering), a REPLACING-AGENTS / avoided-hire claim, or raw contact PII (email /
  phone) -- the categories the source tool marks "Do not post yet".
- Be negation-aware ("no guaranteed savings" is not a hit), matching the source tool.
- Port the catalogue and PII patterns VERBATIM -- this is the operator's safety policy,
  not a place to author or "improve" rules.
- Make the produced verdict actually gate promotion through the existing #2116 contract.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

One new service module (`content_factory_copy_verification.py`) with the verbatim blocker
catalogue, PII patterns, negation-aware scanner, and `verify_copy(text) -> CopyVerification`,
plus tests. No pipeline wiring yet; the producer is the deterministic core the runner /
Phase 4.2 Filter will call.

### Review Contract

- Acceptance criteria:
  - [ ] Clean copy yields verdict "pass" with no hits.
  - [ ] Each forbidden category (outcome / automation / replacing-agents) and each PII
        shape (email / phone) yields verdict "fail" with the hit recorded.
  - [ ] A negated forbidden claim yields "pass" (source-tool parity).
  - [ ] A "fail" verdict makes `recommendation == "promote"` invalid via the #2116
        EditorialAudit guard; a "pass" verdict allows promotion.
- Reachability proof: consumed by `EditorialAudit`'s promote-requires-pass validator
  (merged #2116); proof is the test suite, including the promotion-gating tests.
- Affected surfaces: one new service module and its test file; nothing calls it yet.
- Risk areas: fidelity of the ported catalogue/PII patterns (copied verbatim); the
  negation logic; the verdict -> promote-guard wiring.
- Reviewer rules triggered: R14 (a content safety gate / classifier producing a
  promote-blocking verdict).

### Files touched

- `atlas_brain/services/content_factory_copy_verification.py`
- `plans/PR-Content-Factory-Copy-Verification.md`
- `tests/test_content_factory_copy_verification.py`

## Mechanism

`_RULES` (outcomes / automation / replacing_agents), `_EMAIL_RE`, and `_PHONE_RE` are the
verbatim catalogue and PII patterns from the source tool. `_is_negated` and `_pattern_hits`
are ported verbatim so a negated claim is skipped. `verify_copy(text)` collects every
non-negated banned-claim hit plus every PII match as `"code: evidence"` strings and returns
`CopyVerification(verdict="fail" if hits else "pass", hits=hits)`. Because #2116's
`EditorialAudit` rejects `recommendation == "promote"` unless the verdict is "pass", a
draft that overclaims or leaks PII cannot be promoted.

## Intentional

- Patterns are ported VERBATIM except ONE operator-authorized fix: the source
  `fixed-ticket-volume-reduction` rule ended `(?:%|percent)\b`, and `%\b` can never match a
  `%` before a space, so "30%" slipped through while "30 percent" was caught. Moving the
  boundary inside the alternation (`(?:%|percent\b)`) closes the gap -- a strict tightening
  of the gate, never a loosening. This repo module is now the canonical gate; the Open
  WebUI copy is superseded and should be re-synced from here.
- Only the source tool's BLOCKER categories are ported. Its softer "needs human review"
  layer (answer/ownership qualifiers, owner-routing coverage, CTA reminder) produces
  warnings, not promote-blocks, and is a later slice.
- The producer is unwired: it returns the verdict; the runner / Phase 4.2 Filter that
  calls it on the editor stage is a separate slice.

## Deferred

- The "needs human review" warning layer (owner-routing coverage, answer/ownership
  qualifiers, honest-CTA reminder) from the source tool -- a later slice.
- Wiring `verify_copy` into the stage runner and the Phase 4.2 Editor Filter (fail-closed
  when the gate is unavailable).
- Re-syncing the Open WebUI copy_verification tool from this now-canonical repo module
  (ops step) so the `%`-boundary fix is reflected there too.

## Verification

```
python -m pytest tests/test_content_factory_copy_verification.py -q
```
22 tests pass: clean copy passes; each forbidden claim category and each PII shape fails
with the hit recorded; negated claims pass; multiple hits are all recorded; a non-string
is rejected; and the produced verdict gates promotion through the #2116 EditorialAudit
contract (forbidden copy cannot be promoted, clean copy can, forbidden copy may still be
recommended for revise).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_copy_verification.py` | 118 |
| `plans/PR-Content-Factory-Copy-Verification.md` | 112 |
| `tests/test_content_factory_copy_verification.py` | 116 |
| **Total** | **346** |
