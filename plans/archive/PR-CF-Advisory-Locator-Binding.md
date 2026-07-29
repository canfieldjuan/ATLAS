# PR-CF-Advisory-Locator-Binding

## Why this slice exists

Issue #2189 collects the five findings Codex filed on #2181 after merge. This
slice takes the two that are pure correctness defects in the persistence choke
point, and records the state of a third.

The P1 is the important one. The advisory contract claims that PII-shaped
producer values are **unrepresentable for every writer** -- that is one of the
two evidence theorems in the copy-verification module docstring. It was not
true. The locator grammar validated only the digit SHAPE, so a direct v2 writer
could persist `unqualified-answer-claim: sentence 2125551234` against a
one-sentence body: a raw phone number wearing a locator's clothes, landing in
the git-backed audit the theorem says it cannot reach.

Deliberately NOT in this slice: findings 2 and 3 from #2189 (product-term
negation scope, quantified routing subjects). Both are precision refinements to
the routing-coverage checker's linguistic scope -- a different mechanism from
the persistence gate, with no PII consequence -- and folding them in would put
two unrelated decision surfaces behind one review. #2189 stays open for them.

**Diff-budget overage (463 added lines vs the 400 soft cap) -- why this slice
is indivisible.** The mechanism is 121 lines; the rest is the required review
artifact and its proofs:

| | added |
|---|---:|
| production code (schemas 103 + engine 18) | 121 |
| the plan document itself | 169 |
| both-direction proofs and corrected fixtures | 173 |

The two remaining split lines both produce a worse intermediate state:

* **Move vs. bound.** The sentence definition moves into the contracts module
  ONLY so the locator bound can count sentences exactly as the producer does.
  Landing the move alone is a refactor with no consumer; landing the bound
  alone means a second sentence definition -- the precise defect #2201 round 13
  was filed for.
* **P1 vs. P2.** Splitting the casing fix out yields a ~3-line change plus its
  casing cross-product, in the same function family and the same test file, and
  leaves the P1 slice above the cap regardless.

Tests are not compressible here: the bound's real risk is FALSE REJECTION of
legitimate producer output, and the only thing that rules that out is the
generated lockstep corpus. Shipping the guard without it would land a
persistence gate whose failure mode is silently discarding valid audits.

Diff-budget override: the sentence definition move and the locator bound are one
decision, and separating either from its generated proof would land a
persistence gate without the evidence that it cannot false-reject real output.

### Problem-derived contract

- Root cause (P1): the locator grammar is a SHAPE check with no referent. It
  never binds the sentence number to the body the warning was computed from, so
  any ten-digit value is admissible on any body.
- Root cause (P2): the qualifier detector is case-insensitive but its
  complement-polarity check is not, so `NO proof` reads as a qualification
  while `no proof` does not. Polarity is not a property of capitalization.
- Correct fix must: bind the locator to the audited body's actual sentence
  range at the same choke point that already validates the grammar, for EVERY
  writer and every artifact that carries warnings; and match complement
  polarity case-insensitively.
- The bound must count sentences EXACTLY as the producer does. A second
  sentence definition would drift, and a bound that disagrees with the producer
  is either a false rejection of real output or an open door.
- Must not change: which forms count as PII, the blocking verdict semantics
  frozen in #2181, the warning grammar itself, or the routing-coverage
  linguistic scope (findings 2/3, deferred above).

## Scope (this PR)

Ownership lane: content-factory
Slice phase: production hardening
Max files: 4

1. `atlas_brain/schemas/content_factory.py`:
   - `sentence_spans` / `sentence_count` move here as the SINGLE sentence
     definition, because the contract now needs it to validate its own locator.
   - `_validate_advisory_warnings(warnings, body)` bounds each locator to
     `sentence_count(body)`.
2. `atlas_brain/services/content_factory_copy_verification.py`: imports the
   shared sentence machinery instead of defining it; `re.IGNORECASE` on the
   complement-polarity check.
3. Proof: both-direction probes, a casing cross-product, and a shared-definition
   assertion.

### Review Contract

- Acceptance criteria:
  1. `unqualified-answer-claim: sentence 2125551234` is REJECTED against a
     one-sentence body, for the audit and for a channel variant.
  2. Boundary, both sides: a locator naming the body's last sentence is
     admissible; one past it is not; `sentence 0` still fails the grammar.
  3. A blank body admits no locator at all, and an artifact with no single
     audited body (image-prompt set) admits none by default -- fail closed.
  4. Static checklist lines carry no locator and stay admissible on any body,
     including none.
  5. Producer/contract lockstep, strengthened: every warning
     `advisory_warnings(B)` emits validates on an artifact whose body is `B`,
     across the standing generated corpus. This is what proves the bound cannot
     false-reject real producer output.
  6. The negated-complement check behaves identically across casings, for every
     negation token, in both directions (a genuine qualifier still qualifies).
  7. The contract's `sentence_spans` and the engine's `_sentence_structure`
     return identical spans, so the two cannot drift.
- Reachability proof: the bound sits in the model validator every writer passes
  through -- `EditorialAuditV2`/`V3` and `ChannelVariant` -- so a direct writer
  is gated identically to the runner.
- Affected surfaces: the advisory persistence choke point, the sentence
  primitives' home module, and the qualifier complement check. No change to the
  blocking verdict, the PII patterns, or the routing-coverage scope.
- Risk areas: a bound that under-counts would false-reject legitimate producer
  output -- addressed by criterion 5 and by counting to the LAST content-bearing
  span rather than the number of content spans.
- Reviewer rules triggered: R2 (both-direction tests), R3 (gate cannot be
  self-reported), R10 (one shared definition), R13 (class findings), R14.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_copy_verification.py`
- `plans/PR-CF-Advisory-Locator-Binding.md`
- `tests/test_content_factory_copy_verification.py`

## Mechanism

`sentence_count` returns the 1-based index of the LAST span carrying content,
not the number of content spans. Both details are load-bearing:

- a terminator at end of text yields a trailing empty span, so `len(spans)`
  would admit a locator naming a sentence that does not exist;
- an empty span can also fall in the MIDDLE (a blank line), and the producer
  numbers locators by span POSITION -- so counting only non-empty spans would
  under-count and reject a legitimate locator.

The last content index can never be smaller than any locator the producer can
emit, which is exactly the property criterion 5 checks.

## Intentional

- **The sentence machinery moves to the contracts module rather than being
  duplicated.** The contract needs it to validate its own locator. This follows
  the `is_default_ignorable` precedent from #2201, where a partial second copy
  of a shared predicate was itself the defect.
- **The image-prompt set binds to an empty body**, so it admits no locator. It
  verifies each prompt independently and has no single audited body; the runner
  already forces its warnings to `[]`. The default is empty so a future caller
  that forgets to pass a body fails closed rather than open.
- **Test fixtures were corrected, not the guard.** `_audit()` computed
  `copy_verification` from a text but never set `edited_body_markdown`, pairing
  warnings from one text with an empty audited body -- the exact inconsistency
  this slice makes unrepresentable. Six existing tests carried that pairing.
- **One existing assertion was inverted deliberately.** The locator boundary
  probe asserted that `sentence 1000000` is admissible against no body. That IS
  the reported defect, so the probe now checks the real boundary instead.

## Deliberate exception to the v2/v3 freeze

This slice TIGHTENS validation on frozen contracts, which the freeze note
forbids in general. Recorded here as a decision with its evidence rather than
made silently.

**Why a new schema version does not solve it.** Putting the bound on a v4 while
leaving v2/v3 admissible leaves the P1 open for exactly the writer it was filed
against: a direct v2 writer could still persist a phone number as a locator. A
new version relocates the hole instead of closing it.

**Why the freeze's stated harm does not occur here.** The freeze exists because
adding a FIELD makes new artifacts unreadable to an old reader -- `extra=forbid`
rejects the unknown key, so a rollback loses data (that is the #2192 round-8
case). This change adds no field. Artifacts written after it remain readable
byte-for-byte by pre-change code, so the rollback direction is unaffected. Only
the reverse narrows: an artifact that was ALREADY semantically broken -- a
locator naming a sentence of a body that does not exist -- stops validating.

**What is actually invalidated.** A payload carrying a locator warning with a
blank or absent `edited_body_markdown`. That is not incidental breakage; it is
the P1 in its most extreme form, since a locator with no body is a raw producer
value with no referent at all. The runner cannot produce it: it clears
`advisory_warnings` whenever the edited body is blank.

**Evidence.** The local git-backed job store (`~/content-factory/jobs`, 3 jobs
/ 9 artifacts) contains zero editorial audits carrying a locator warning, so no
persisted artifact is invalidated in practice. This bounds the local blast
radius; it does not prove none exists anywhere.

**Migration path**, as the review asked for, and now stated in the validation
error itself so an operator hitting it gets the repair: supply the
`edited_body_markdown` the artifact was audited against, or drop the locator
warning -- without a body it names nothing.

## Deferred

- #2189 findings 2 and 3 (product-term negation scope, quantified routing
  subject nouns): both reproduce on current main, both are routing-coverage
  precision rather than persistence correctness. #2189 stays open.
- #2189 finding 1 (v1 metadata laundering) needs NO code: it was already fixed
  by the normalizer rewrite in #2192 round 8, which synthesizes `schema_version`
  only when the supplied value is absent or agrees with the declared tag.
  Verified on current main: `"bogus"` and `99` survive to be rejected by the
  version validator; only `None` is synthesized.
- Observed while probing, NOT a #2189 finding and not fixed here: qualifier
  opener/complement pairing is asymmetric on main -- `If the tickets contain
  proof` and `When evidence exists` qualify, while `When the tickets contain
  proof` and `If evidence exists` do not. Worth its own investigation; this
  slice asserts only the casing invariant.

Parked hardening: none.

## Verification

    python -m pytest tests/test_content_factory_runner.py \
        tests/test_content_factory_store.py \
        tests/test_content_factory_schemas.py \
        tests/test_content_factory_copy_verification.py -q
    # -> 1377 passed

Detection proven by injection, per AGENTS.md 3i. Neutralizing both fixes
(dropping `re.IGNORECASE`, and making the locator bound unreachable):

    mutated:   15 failed, 6 passed
    restored:  21 passed

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 122 |
| `atlas_brain/services/content_factory_copy_verification.py` | 61 |
| `plans/PR-CF-Advisory-Locator-Binding.md` | 234 |
| `tests/test_content_factory_copy_verification.py` | 239 |
| **Total** | **656** |
