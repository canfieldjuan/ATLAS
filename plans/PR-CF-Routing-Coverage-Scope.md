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
  report with no routing, and emitted nothing.
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
   - `_predicate_end` truncates the polarity range at the first adjunct
     preposition; `_report_shape_sentences` uses it.
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
  7. The standing 18-round regression corpus stays green.
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

**Predicate boundary.** A predicate ends where an adjunct PP begins, so the
polarity range stops at the first adjunct preposition after the product term
rather than at the clause end. Prepositions are a CLOSED class, which is what
makes this a grammar rule rather than a word list -- the contrast that matters
is `is not provided` (negation inside the predicate, still denies) versus `is
provided without delay` (negation inside an adjunct, does not).

**Subject binding.** One helper, used by both the same-sentence and
later-sentence paths. They previously carried the same test written twice,
which is how one hole existed in both. A quantifier binds when it is bare (a
predicate follows), when it carries a report item, or when a pro-form continues
the reference. Any other noun renames the subject.

The partitive branch looks THROUGH `of` and any determiner to the head noun.
Admitting `of` wholesale -- the first thing that made the regression corpus
pass -- would have reproduced the same hole one token further out, since `each
of the invoices` is about invoices exactly as `each invoice` is. That is the
string-closure trap AGENTS.md 3k.1 names, so it is closed as a class.

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
| `atlas_brain/services/content_factory_copy_verification.py` | 98 |
| `plans/PR-CF-Routing-Coverage-Scope.md` | 125 |
| `tests/test_content_factory_copy_verification.py` | 100 |
| **Total** | **323** |
