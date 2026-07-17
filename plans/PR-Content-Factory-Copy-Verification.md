# PR-Content-Factory-Copy-Verification

## Why this slice exists

The #2116 Content Factory contract gave `EditorialAudit` a `copy_verification` field and
a rule that a draft may not be promoted unless `copy_verification.verdict == "pass"` --
but nothing PRODUCED that verdict, so the promote guard was a shape with no teeth. The
deterministic claim/PII catalogue existed only as the operator's "Resolution Audit Draft
Verifier" tool inside the Open WebUI database, un-versioned and un-testable. This slice
(plan Phase 4.1) builds the producer in the repo: a deterministic BEST-EFFORT BACKSTOP
that fails the verdict when copy contains a forbidden marketing claim or raw contact PII,
so the obvious overclaims are hard-blocked and the human approval step (the real safety
guarantee) sees less noise.

### Problem-derived contract

A correct fix must:
- Produce a `CopyVerification` (verdict + hits) from draft text, deterministically.
- Fail the verdict on the operator's promote-blocking categories -- forbidden OUTCOME
  claims (guaranteed savings, fixed deflection %, ticket reductions), AUTOMATION claims
  (auto-publishing / auto-answering), REPLACING-AGENTS / avoided-hire claims, and raw
  contact PII (email / phone) -- matching the COMMON wordings of each, not one fixture
  phrase.
- Scope negation to the claim: a negation immediately governing a claim suppresses it,
  but an unrelated earlier negation must not, and an ambiguous-scope negation errs toward
  flagging (fail closed).
- Never persist the raw PII it blocks: PII hits record only a redacted marker.
- Not over-claim: a regex catalogue cannot enumerate every paraphrase of a natural-
  language claim, so the gate is a backstop, not a complete classifier; human approval
  remains the real gate.
- Make the produced verdict actually gate promotion through the existing #2116 contract.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

One new service module (`content_factory_copy_verification.py`) -- the claim/PII catalogue,
negation-aware scanner, and `verify_copy(text) -> CopyVerification` -- plus tests. No
pipeline wiring yet; the producer is the deterministic core the runner / Phase 4.2 Filter
will call.

### Review Contract

- Acceptance criteria:
  - [ ] Clean/legitimate copy yields "pass" (false-positive guard on non-claim uses of
        "guarantee", "auto-", "replace", "reduce tickets", "avoid").
  - [ ] Each promote-blocking category fails on its common wordings, including modifier
        and inflection variants (guaranteed cost/monthly savings; 40% or 40 percent
        deflection; reduce tickets/ticket volume by N%; N% fewer support tickets;
        auto-publish/publishes/published; replace your agents; avoid hiring another agent).
  - [ ] A directly-governing negation passes; an unrelated earlier negation still fails;
        an ambiguous-scope negation fails (conservative).
  - [ ] PII fails the verdict but the raw email/phone never appears in `hits` (only a
        redacted marker).
  - [ ] A "fail" verdict makes `recommendation == "promote"` invalid via the #2116
        EditorialAudit guard; a "pass" verdict allows promotion.
- Reachability proof: consumed by `EditorialAudit`'s promote-requires-pass validator
  (merged #2116); proof is the test suite, including the promotion-gating tests.
- Affected surfaces: one new service module and its test file; nothing calls it yet.
- Risk areas: pattern coverage vs false positives (both sides tested); negation scope;
  PII never persisted; the verdict -> promote-guard wiring.
- Reviewer rules triggered: R14 (a content safety gate producing a promote-blocking
  verdict). This gate is an incomplete-by-nature NL backstop, not a complete classifier.

### Files touched

- `atlas_brain/services/content_factory_copy_verification.py`
- `plans/PR-Content-Factory-Copy-Verification.md`
- `tests/test_content_factory_copy_verification.py`

## Mechanism

`_RULES` (outcomes / automation / replacing_agents) holds per-category regex that match
the common inflections/modifiers of each promote-blocking category. `_is_negated` checks
only the two words immediately before a match (segment-bounded by `.!?;,` / newline), so a
directly-governing negation suppresses the hit while an unrelated earlier negation does
not. `_claim_hits` records `"code: evidence"` for non-negated claim matches; `verify_copy`
adds a REDACTED PII marker (`"email: <redacted>"` / `"phone: <redacted>"`) when a raw
email/phone is present, then returns `CopyVerification(verdict="fail" if hits else "pass",
hits=hits)`. Because #2116's `EditorialAudit` rejects `recommendation == "promote"` unless
the verdict is "pass", a draft that overclaims or leaks PII cannot be promoted.

## Intentional

- The gate is a deterministic BEST-EFFORT BACKSTOP, not a complete NL classifier: a novel
  paraphrase can pass, so the module and Review Contract say so explicitly and point at
  human approval as the real guarantee. This is the honest answer to the open-category
  nature of banned-claim detection -- broaden common coverage, do not pretend completeness.
- Negation is scoped to the immediately-preceding words and errs toward FLAGGING when a
  negation's scope is ambiguous (fail closed): a false positive only routes to human
  review; a false negative would ship an overclaim.
- PII evidence is redacted out of `hits` because that verdict is persisted in the git-
  backed job folder -- the gate must not duplicate the PII it blocks.
- Derived from the operator's tool but not a byte-for-byte copy: this repo module is now
  the canonical gate (carrying the %-boundary fix, negation-scope fix, and same-category
  coverage broadening); the OWUI copy is superseded and should be re-synced from here.
- Only the promote-BLOCKING categories are implemented; the source tool's softer "needs
  human review" warning layer is a later slice.

## Deferred

- The "needs human review" warning layer (owner-routing coverage, answer/ownership
  qualifiers, honest-CTA reminder) from the source tool -- a later slice.
- Wiring `verify_copy` into the stage runner and the Phase 4.2 Editor Filter (fail-closed
  when the gate is unavailable).
- Re-syncing the Open WebUI copy_verification tool from this now-canonical repo module.

## Verification

```
python -m pytest tests/test_content_factory_copy_verification.py -q
```
43 tests pass: legitimate copy (incl. non-claim uses of the trigger words) passes; each
promote-blocking category fails on its common variants; a directly-governing negation
passes while an unrelated earlier negation and an ambiguous-scope negation fail; PII fails
but is redacted out of the hits (the raw value never appears); a non-string is rejected;
and the verdict gates promotion through the #2116 EditorialAudit contract.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_copy_verification.py` | 151 |
| `plans/PR-Content-Factory-Copy-Verification.md` | 132 |
| `tests/test_content_factory_copy_verification.py` | 180 |
| **Total** | **463** |
