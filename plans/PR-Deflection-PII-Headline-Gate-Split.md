# PR-Deflection-PII-Headline-Gate-Split

## Why this slice exists

#1742 now has the deterministic tiny-corpus scrubber loop clean: email, phone,
SSN, payment card, DOB, street address, order IDs, cue-prefixed names, and
must-survive tokens all score clean after #1757. The remaining
`free_high_severity_leak_count=1` is the explicitly deferred cue-less/open-set
`person_name-001` gap that #1742 says must be reported but not gate on until a
separate NER/open-set slice exists.

Root cause: the scorer headline currently exposes only one all-in
high-severity free-surface leak count. That count correctly keeps the open-set
name gap visible, but it also makes the deterministic gate-eligible state
ambiguous: the future advisory-to-gating decision cannot distinguish "the
deterministic scrubber is clean" from "all high-severity PII is clean." This
slice fixes the measurement root by adding an explicit gate-eligible headline
split while keeping the all-in/deferred counts visible.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 3

1. Add scorer headline fields for gate-eligible free high-severity leaks and
   deferred cue-less/open-set name leaks.
2. Keep the existing all-in `free_high_severity_leak_count` unchanged so the
   residual open-set gap stays visible.
3. Add focused tests proving cue-less names are excluded only from the
   gate-eligible count, while any other high-severity free-surface leak still
   blocks that gate-eligible count.

### Review Contract

Acceptance criteria:
- Existing `headline.free_high_severity_leak_count` remains the all-in
  diagnostic count and still reports `1` on the tiny corpus.
- New gate-eligible headline fields report `0` leaks / pass true on the current
  tiny corpus because the only residual is `person_name` with
  `name_subtype=cue_less`.
- A forced non-open-set high-severity free-surface leak increments both the
  all-in and gate-eligible counts.
- Leak samples remain surrogate-id only; no raw spans or tokens are added.
- This PR does not flip CI/advisory behavior to a hard gate and does not attempt
  cue-less/open-set NER.

Affected surfaces:
- `scripts/score_deflection_pii_recall.py`
- `tests/test_score_deflection_pii_recall.py`

Risk areas:
- Hiding the deferred cue-less name gap by filtering it out of the all-in
  headline or leak samples.
- Accidentally excluding non-name high-severity leaks from the gate-eligible
  count.

- Reviewer rules triggered: R1 Requirements match, R2 Test evidence, R10
  Maintainability, R14 Codebase verification.

### Files touched

- `plans/PR-Deflection-PII-Headline-Gate-Split.md`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The scorer already computes per-label leak scores with class, severity,
name_subtype, surface, surrogate_id, and leak status. This slice replaces the
single headline count helper with a headline summary helper that partitions
unique free-surface high-severity leaked surrogate IDs into:

- all high-severity leaked IDs;
- deferred cue-less/open-set `person_name` leaked IDs;
- gate-eligible leaked IDs, which are all leaked IDs minus the deferred cue-less
  name IDs.

The output remains additive. Existing consumers can keep reading
`free_high_severity_leak_count`; future advisory/gating code can use the
gate-eligible count without pretending the open-set name gap disappeared.

## Intentional

- No NER or cue-less name scrubber change; #1742 explicitly defers that to a
  separate slice.
- No CI gate flip. This only makes the scorer output unambiguous for the future
  operator decision.
- No raw leaked values in headline or samples.

## Deferred

- Operator-derived gold corpus, thresholds, and advisory-to-gating timing remain
  open #1742 decisions.
- Model-based NER for cue-less/open-set names remains separate.

Parked hardening: none.

## Verification

- `pytest tests/test_score_deflection_pii_recall.py -q` -- 17 passed.
- `python scripts/score_deflection_pii_recall.py --json` -- status ok;
  headline reports one all-in deferred open-set name leak and zero
  gate-eligible leaks.
- Python bytecode compile for the scorer script and scorer tests -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-PII-Headline-Gate-Split.md` | 114 |
| `scripts/score_deflection_pii_recall.py` | 51 |
| `tests/test_score_deflection_pii_recall.py` | 38 |
| **Total** | **203** |
