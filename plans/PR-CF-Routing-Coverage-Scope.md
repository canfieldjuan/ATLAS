# PR-CF-Routing-Coverage-Scope

## Why this slice exists

Closes the last two findings in #2189. Both are precision defects in the
routing-coverage checker, and both fail in the SAME direction: the checklist
stays silent on a draft that has not actually routed anything. A false negative
here is the expensive one -- the warning exists to tell a reviewing human that
owner routing is missing, so suppressing it hides the gap it was built to
surface.

The two remaining findings were separated from the persistence work in #2219
because they are a different mechanism (linguistic scope, no PII consequence).
This slice is that follow-up, and #2189 closes with it.

### Problem-derived contract

- Root cause (finding 2): product-term polarity is evaluated over
  `[term_start, CLAUSE END]`, so any negation later in the clause denies the
  report surface -- including one inside an unrelated trailing adjunct. `The
  Resolution Audit is provided without delay.` affirmatively provides the
  report with no routing, and emitted nothing. The deeper cause is that a
  character RANGE cannot express which negations govern the assertion; kind
  can.
- Root cause (finding 3): `_ANAPHORIC_SUBJECTS` lumps bare anaphors (`it`,
  `they`) together with quantifiers that CARRY A NOUN (`each`, `every`, `all`,
  `these`). `every` alone satisfied the subject test, so `Every invoice is
  assigned to Billing.` covered a report about issues.
- Correct fix must: bind polarity to the predicate governing the product term;
  and require a noun-bearing quantifier to carry a report item, while keeping
  genuinely bare anaphors binding.
- Must not change: the blocking verdict, the PII patterns, the warning grammar,
  or the locator binding from #2219.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: production hardening
Max files: 3

1. `atlas_brain/services/content_factory_copy_verification.py`:
   - `_assertion_negated` decides product-term polarity by negation KIND,
     with scopes and the verbal check cached per clause.
   - `_subject_binds_to_report(words)` replaces the duplicated inline subject
     test at both call sites.
2. Proof: generated cross-products on both sides of each decision.

### Review Contract

- Acceptance criteria:
  1. `The Resolution Audit is provided <adjunct>.` warns, across trailing
     adjuncts including `without delay`, `with no delay`, `before the review`.
  2. A negation that genuinely governs the product term still denies it: `is
     not provided`, `is never provided`, `cannot be provided` emit nothing.
  3. A quantified subject carrying a NON-report noun does not cover the report,
     across {each, every, all} x {invoice, invoices, customer, customers,
     vendor}.
  4. A quantified subject carrying a report item still covers it, across the
     same quantifiers x {issue, issues, ticket, tickets, finding, findings}.
  5. Genuinely bare anaphors and pro-forms still cover: `Each is assigned`,
     `Each one is routed`, `Each of them is routed`.
  6. Partitives do not smuggle a non-report subject: `Each of the invoices is
     routed` warns, `Each of the tickets is routed` does not.
  7. `both` is treated as noun-bearing: `Both invoices are assigned` warns,
     `Both tickets are assigned` does not.
  8. Modified heads are classified correctly: `Each of the open tickets is
     assigned` covers, `Each of the open invoices is assigned` does not.
  9. A noun-attached PP does not defeat a governing negation: `The Resolution
     Audit for this month is not provided.` emits nothing, while the same
     sentence without `not` warns.
  10. The pass stays linear: a 67 KB clause with 3,200 product terms runs in
      0.023s, against 0.022s on `main` (it was 4.6s mid-review).
  11. Emphatic `not only/just` is affirmative: `The Resolution Audit is not
      only provided but current.` warns, because the scope model already
      classifies it that way and the polarity check now asks IT rather than
      matching `not` independently.
  12. A PP-modified subject is classified by the noun the quantifier binds:
      `Each ticket in the report is assigned` covers, `Each invoice for a
      ticket is assigned` does not.
  13. The standing 18-round regression corpus stays green.
- Reachability proof: both fixes sit inside `advisory_warnings`, the same entry
  point the runner calls for every audit and channel variant.
- Affected surfaces: routing-coverage scope only. No change to the verdict,
  the PII patterns, the grammar, or the locator bound.
- Risk areas: over-warning (a fix that just makes everything warn) -- addressed
  by criteria 2, 4 and 5, which are the passing side of each decision.
- Reviewer rules triggered: R2, R10, R13, R14.

### Files touched

- `atlas_brain/services/content_factory_copy_verification.py`
- `plans/PR-CF-Routing-Coverage-Scope.md`
- `tests/test_content_factory_copy_verification.py`

## Mechanism

**Polarity by negation KIND, not by position.** The first attempt truncated the
range at the first preposition after the product term. Review found three
problems with that, all real: a PP can attach to the NOUN (`The Resolution
Audit for this month is not provided` truncated at `for` and ignored the
governing `not`); and re-scanning the clause suffix per term made the checker
quadratic (7.1s on a 54 KB clause).

The range is gone. Polarity is decided by the negation's KIND, which the scope
model already encodes: a scope covering the term itself denies it; a VERBAL
negator anywhere in the clause attaches to the predicate and denies wherever
the term sits; a bounded scope elsewhere is an adjunct's own complement and
denies nothing. Verbal negation is read from the scope model's OWN classification. Two earlier
attempts got this wrong in opposite directions: inferring "verbal" from scope
extent misread `without delay` at a clause end, and an independent regex over
every `not` disagreed with the model's emphatic exception, so `is not only
provided` read as a denial. The model already decides this; asking it is the
only version that cannot drift. Both the scopes and the verbal check are cached per clause, so the pass
is linear again (0.023s vs 0.022s on `main` for the same input).

**Subject binding by the actual head.** One helper, used by both the
same-sentence and later-sentence paths; they previously carried the same test
written twice, which is how one hole existed in both.

A noun-bearing quantifier is classified by its grammatical SUBJECT HEAD: the
noun it binds, before any post-modifier. Taking the last pre-predicate token
instead read `each ticket in the REPORT` as being about reports (regressing
valid routing) and `each invoice for a TICKET` as being about tickets
(preserving the original false negative). That handles determiners, modifiers and partitives
uniformly (`each of the open tickets` -> `tickets`) instead of special-casing
`of` and inspecting a fixed four-token window -- which review showed both
missed `both invoices` and rejected valid modified heads. `both` is noun-bearing
and moved out of the bare set. Bare use (a predicate follows directly) and
pro-forms still bind, since a pro-form inherits reference rather than renaming
the subject.

## Intentional

- **Pro-forms are not content nouns.** `each one is routed` was suppressed by
  the pre-change code and must stay suppressed: `one` inherits reference rather
  than renaming the subject. This surfaced as a regression-corpus failure on
  the first implementation, and is now an explicit criterion.
- **The subject test became one function.** Two copies of a predicate is how
  the reported hole survived in both paths; the #2201 round-13 lesson applied
  to this module.

## Deferred

- The qualifier opener/complement asymmetry recorded in #2219's plan (`If ...
  contain proof` and `When evidence exists` qualify; the crossed pairings do
  not) is untouched here. It is a different checker with no established correct
  behaviour yet.

Parked hardening: none.

## Verification

    python -m pytest tests/test_content_factory_runner.py \
        tests/test_content_factory_store.py \
        tests/test_content_factory_schemas.py \
        tests/test_content_factory_copy_verification.py -q
    # -> 1384 passed before the new tests; 1468 with them

Detection proven by injection, per AGENTS.md 3i. Reverting both fixes -- the
polarity range back to the clause end, and quantifiers binding unconditionally:

    mutated:   33 failed, 51 passed
    restored:  84 passed

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_copy_verification.py` | 154 |
| `plans/PR-CF-Routing-Coverage-Scope.md` | 172 |
| `tests/test_content_factory_copy_verification.py` | 152 |
| **Total** | **478** |
