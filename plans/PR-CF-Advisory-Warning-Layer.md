# PR-CF-Advisory-Warning-Layer

## Why this slice exists

#2136 item 2: the pipeline records only the promote-blocking verdict; the
source verifier tool's softer "needs human review" checks (owner-routing
coverage, unqualified answer/ownership claims, honest-CTA reminder) exist
only in the opt-in OWUI tool, so a pipeline audit carries no advisory
signal for the approving human.

This slice exceeds the 400-LOC soft cap and is indivisible: the advisory
producer, the versioned contract that carries it (v2 + frozen v1), the
runner normalization, and the both-direction precision tests are one
reviewable behavior -- landing the producer without the versioned carrier
would re-open the rollback hazard round 2 flagged, and landing the carrier
without tests would ship unproven heuristics.

### Problem-derived contract

- Root cause: the deterministic gate covers blocking categories only;
  the advisory layer was never ported into the repo pipeline.
- Correct fix must: add a deterministic, PII-safe advisory producer next
  to the gate; persist its output on the audit artifact; inject it in the
  runner with the same self-report discipline as the verdict (worker
  claims discarded); change NO gating behavior.
- Correct fix must also (contract REVISED in round 3, per the review):
  version the audit contract (v2 carries the checklist; v1 frozen) and
  admit both versions at the store's audit stage -- persisting the new
  field is impossible without the version-admission touch, so the store's
  STAGE_SCHEMAS/admission check is IN scope for exactly that change.
- Must not change: `verify_copy` verdict semantics, the promote
  validator, any other store behavior, the OWUI tool (it is the source
  and already carries this layer, re-synced v0.2.0), the pre-#2181
  `EditorialAudit.model_validate(v1_payload)` API.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

1. `atlas_brain/services/content_factory_copy_verification.py`:
   `advisory_warnings(text)` ports the tool's soft checks (unqualified
   answer claims, unqualified ownership claims, report-shape without
   owner routing, unconditional honest-CTA reminder). FINAL DESIGN
   (review rounds 4-6): warnings are EVIDENCE-FREE -- claim code +
   1-based sentence locator + grammar-safe keyword only, no draft text --
   so nothing PII-shaped is representable; qualifier association is
   clause-scoped and counting-based (fail-closed, linear); sentence/
   clause spans are precomputed once per draft.
2. `atlas_brain/schemas/content_factory.py`: `EditorialAuditV2`
   (schema tag `editorial_audit.v2`) carries
   `advisory_warnings: list[str]` and VALIDATES each entry against the
   bounded deterministic grammar (persistence choke point for direct
   writers); v1 (`EditorialAudit`, pre-change API) is FROZEN and stays
   admissible. Warnings never gate the recommendation.
3. `atlas_brain/services/content_factory_runner.py`:
   `_enforce_copy_verification` overwrites `advisory_warnings` from the
   edited body alongside the verdict; empty body clears the list
   (a fabricated checklist must not blind the reviewer).
4. Proof: category tests both directions, PII probe, promote-with-warnings
   validity, runner overwrite + empty-body clearing, old artifacts without
   the field still validate.

### Review Contract

- Acceptance criteria:
  1. Each advisory category warns on its trigger and stays silent on the
     qualified/covered form (test-asserted both directions).
  2. A passing verdict may promote regardless of warnings; warnings are
     absent from every gating validator (schema review + test).
  3. Runner-persisted audits carry the deterministic checklist, never the
     worker's; empty edited body yields an empty checklist plus the
     existing fail verdict.
  4. No recorded warning can carry raw email/phone text (pre-redaction
     probe includes the sentence-split truncation case).
- Reachability proof: `run_stage` on any editorial audit (schema-gated,
  same path as the #2137 verdict enforcement); artifact lands in the
  git-backed job folder for the approving human.
- Affected surfaces: copy-verification module, editorial-audit contract,
  runner enforcement hook. Gating behavior unchanged.
- Risk areas: warning noise (CTA reminder is unconditional by design,
  mirroring the source tool -- consumers treat it as a checklist line,
  not a signal); regex drift vs the OWUI tool (the repo module is
  canonical; the tool is the synced copy).
- Reviewer rules triggered: R1 (#2136 item 2), R2 (both-direction tests),
  R5 (no gating change, old artifacts validate), R10 (advisory logic
  lives beside the gate it complements), R13 (open-text classifier:
  every review round's finding was closed as a CLASS -- evidence-free
  locators, bounded warning grammar at the schema, clause-scoped counting
  association, polarity with emphatic exclusion, owner-target routing --
  each with both-direction probes; the module contract remains
  best-effort-backstop with the human approval as the real gate), R14.

### Review round 15 (Codex)

Four findings, all fixed: sentence terminators protect abbreviations and
initials ("Dr. Billing", "J. Smith", "Acme Inc." stay one sentence --
locators match the human count); coordination is proposition-level only
("clearly and consistently ranks" keeps its report shape, "answers or
resolutions" stays inside the denial's scope, while clause coordination
still splits); bounded focus modifiers ("Only when evidence exists, ...")
are part of the fronted qualifier, not a proposition; and this plan's
Files-touched/diff table was regenerated by scripts/sync_pr_plan.py so
the size contract matches the reviewed patch exactly. The clause-span
and boundary-kind lists now come from ONE filtered pass, removing a
latent misalignment between them.

### Review round 14 (Codex)

Five findings, all fixed: the digit choke point is CATEGORY-COMPLETE --
the mask is the str.isdigit/isnumeric predicate itself, so circled/
superscript and every other numeral class a word-character can admit is
masked (no regex class to fall behind); same-sentence routing must also
be ABOUT the report (same clause, a verb-initial coordinated verb
phrase, or an anaphoric/report-item subject -- "while Billing probably
owns invoices" no longer covers); a clause-initial determiner negation
denies its whole proposition ("No support agent drafts answers") while
mid-clause determiners stay narrow; causal openers (because/since/so/
therefore/hence/thus) end negation scope, so "...because Billing owns
refunds" keeps its affirmative claim; and the locator grammar bound was
raised to ten digits, beyond any physically possible draft, with a
boundary probe at the old limit.

### Review round 13 (Codex)

Five findings, all fixed in the engine: the digit theorem masks Unicode
decimal digits in every script (the ASCII class was the hole); routing
and report-shape polarity use range intersection across the whole
relation ("Billing never owns them" and "The report does not rank
issues" are denials); report binding requires SUBJECT-position anaphora
or a determiner + report-item-noun subject -- this reverses the
round-12 clause-wide rule at the reviewer's direction (fail-closed:
"owns each fix" object anaphora no longer binds and the test flipped
accordingly); routing subject state is cached per clause (linear).

### Review round 12 (Codex) -- the hardening rewrite

Operator direction: no brittle heuristics in production. The advisory
layer was REWRITTEN as a small deterministic linguistic engine replacing
all positional regex windows: token stream -> sentence spans -> clause
spans with boundary KINDS (word openers vs punctuation); a negation
SCOPE model (determiner negation spans <=2 tokens, verbal negation spans
to clause end, emphatic not-only/just is affirmative, denial tested by
RANGE intersection so in-match negation counts); qualifier GOVERNMENT by
adjunct direction (own clause; postmodifier across word boundaries only;
fronted sentence-initial clause governs the next -- never both
directions); routing coverage BOUND to the report proposition (same
sentence, or a later sentence with an anaphoric token referring back);
label-style absence ("Owner lane: TBD") negates routing.

Two theorems replace pattern-completeness arguments: (1) gate hits mask
EVERY digit character after the readable markers -- no separator grammar
exists to enumerate; (2) advisory warnings carry a fixed code and
sentence number ONLY -- producer text (names, numbers) is
unrepresentable, and the v2 schema enforces the same grammar at
persistence for every writer.

All six round-12 findings close inside this structure (digit theorem;
denial-with-modifiers via scope; owner-lane labels; modifier-tolerant +
polarity-aware report-shape; report-bound routing; name-free locators).
The 12-round Codex counterexample corpus (60+ regressions) passes
unchanged, plus grammar-derived generative invariants: generated denial
templates never warn, generated bare-claim templates always warn, and
every producible output validates the schema grammar.

### Review round 11 (Codex)

Three findings, all fixed mechanically: the gate's digit-run mask treats
newlines as separators (claim patterns cross lines, so multiline
phone-shaped values in a hit are masked); a copular absence predicate
immediately after a routing relation negates it ("the owner lane is
unknown" warns) while unrelated trailing modifiers still do not; and
claim polarity binds to the three words before the assertion plus the
match span, so an earlier negative noun phrase ("With no delay we draft
answers") is not read as a denial while genuine denials stay recognized.

### Review round 10 (Codex) -- fixes + the declared precision boundary

Four findings fixed: the gate's digit-run mask allows multi-character
separators ("020--7946--0958" in a claim hit is masked); claim polarity
is evaluated across the subject-to-relation span (subject-first denials
like "Billing never owns refunds" register as denials); routing negation
is bound to the relation window, so unrelated absence language after the
target ("...assigned to billing with no due date") does not invalidate
coverage; and `EditorialAuditV2.schema_version` is `Literal[2]`.

Two findings WAIVED as the declared precision boundary of this layer
(recorded in the PR body's reconciliation): subordinate-modifier
qualifier binding ("Billing owns refunds that may be disputed" -- the
trailing "may" excusing the ownership claim) and Markdown-decorated
sentence starts ("Intro. **We draft answers.**" locator off-by-one)
require grammatical parsing, which a deterministic regex backstop
deliberately does not attempt. The module contract has stated since
Phase 4.1 that this layer is a best-effort backstop and the human
approval before publish is the real gate; residual precision cases are
operator-policy catalogue growth (#2136 item 4), not silent gaps -- the
boundary is documented here, in the module docstring, and in the waiver
ledger.

### Review round 9 (Codex)

Five findings, all fixed: the fronted-qualifier carry is bounded to the
SAME SENTENCE (a completed sentence's qualifier can never excuse the next
sentence's claim); emphatic "not only"/"not just" is affirmative in the
polarity check (mirroring the blocking gate); a single newline is never a
sentence boundary -- only a blank line is structural (capitalization is
not the signal, so wraps before proper nouns stay in place); every
routing alternative now requires an owner-like target ("routes each issue
by severity" is not coverage); and the Review Contract explicitly
declares R13 with the class-closure discipline this classifier is held
to.

### Review round 8 (Codex)

Four findings, all fixed (one was a real regression from round 7's
tightening): fronted qualifiers carry forward exactly one clause when
their own clause is claim-free ("When evidence exists, we draft answers."
is silent again; the carry cannot reach later clauses, so the round-6
attack still warns); `owned by` requires an owner-like target like the
other routing alternatives ("owned by severity" is not coverage); a bare
newline is a sentence boundary only at a blank line or a capital-starting
line (Markdown soft wraps stay in their sentence); and advisory matches
have a polarity check -- explicit denials ("We do not draft answers",
"Refunds are never owned by Billing") are not unqualified assertions.

### Review round 7 (Codex)

Four findings, all fixed: clause granularity tightened -- dashes, slashes,
and parentheses are clause boundaries, so a qualifier must share the
minimal punctuation-delimited proposition with the claim it excuses (the
dash-attack "We draft every answer regardless — when evidence exists ..."
now warns) while the per-clause count remains the fail-closed backstop;
assignment/routing targets must be owner-like ("assigned to a severity"
is not coverage, "assigned to the billing team" is); a sentence
terminator plus trailing newline is ONE boundary (normal Markdown
paragraphs no longer inflate locators); routing negation is evaluated
once per clause (cached -- removes the last rescan hot spot).

### Review round 6 (Codex)

Six findings, all fixed: qualifier association is now CLAUSE-scoped and
counting-based -- a qualifier only excuses claims inside its own clause
and at most as many as there are qualifiers there, which simultaneously
closes the cross-clause leak ("When evidence exists, support triages
tickets, but we draft an answer regardless" warns), stays fail-closed
under unknown separators, and is linear (no pairwise scan -- kills the
8k-claim pathological case); the producer normalizes matched keywords to
the schema's bounded alphabet (a wrapped "Billing\nreally owns" cannot
invalidate a valid audit -- round-6 BLOCKER); bare "routing" no longer
counts as owner coverage (a target relation is required); sentence
terminators require a following capital/quote or end-of-text (domains
and abbreviations do not inflate locators); and this plan's scope/
Intentional text was reconciled to the shipped evidence-free versioned
design.

### Review round 5 (Codex)

Five findings, all fixed — two structurally: (1) qualifier handling moved
from boundary enumeration to FAIL-CLOSED ASSOCIATION: each qualifier
occurrence excuses at most one claim (nearest in its sentence), so no
separator style — present or future — can hide a second claim behind a
qualified neighbor; (2) the schema is now the choke point: EditorialAuditV2
validates every warning against the bounded deterministic grammar
(static lines or code+sentence+alphabetic-keyword locators), so a DIRECT
writer cannot persist free text or PII either — the canonical strings
moved to the contract layer and the producer is lockstep-tested against
the grammar. Also: `owns` suppression bounded to owner-like subjects
("Caching owns the latency" no longer counts as routing); boundary-span
starts cached with the spans (true O(log n) lookups); sentence locators
count real sentences ("Version 2.1" does not split, "Really?!" is one
boundary).

### Review round 4 (Codex)

Four findings, all fixed — the big one structurally: advisory warnings no
longer persist ANY free draft text. Each warning records only the claim
code, the 1-based sentence number, and the matched keyword (word
characters by construction), so the no-raw-PII criterion holds by
construction instead of by redaction completeness ("020 - 7946 - 0958"
and every future separator style included). The reviewer locates the
sentence in the draft artifact beside the audit. Consequences: sentence/
clause boundaries are precomputed once per draft with O(log n) lookups
(kills the quadratic rescan finding); qualifiers bind per coordinated
clause (and/or are boundaries now, so "one answer when evidence exists
and another answer regardless" warns); owner-routing coverage requires a
negation-free COMPLETE clause ("assigned to nobody", "routing remains
unresolved" now warn). The gate's claim-hit evidence (which does record
matched phrases) keeps the digit-run redaction backstop.

### Review round 3 (Codex)

Four findings, all fixed: `EditorialAudit` keeps its pre-change v1 API
(the v2 contract is `EditorialAuditV2`; registry dispatches both);
evidence redaction gains a CLASS backstop -- any 5+ digit run joined by
single non-word separators is masked (`020/7946/0958` included), ending
the format-enumeration game on the evidence path; owner-routing coverage
requires a NON-NEGATED affirmative relation ("no one is assigned",
"not routed to Billing" now warn; bare "responsible for" no longer
suppresses); and this contract's "must not change the store" line was
revised to name the version-admission touch the v2 artifact requires
(the contradiction the review flagged).

### Review round 2 (Codex)

Nine findings, all fixed: sentence terminators added to clause boundaries;
owner-routing suppression requires AFFIRMATIVE assignment/review language
(bare "owner" no longer suppresses "the owner is unknown"); report-shape
matching is relational (report-noun + output-verb, or product terms) so
"The compliance audit passed" is silent; responsibility claims need an
owner-like subject; international AND local phone-shaped digit runs are
redacted in advisory evidence but the GATE expansion was REVERTED (the
slice contract freezes verdict semantics -- widening the PII block is a
separate operator decision); `editorial_audit.v2` carries
advisory_warnings while v1 is FROZEN and stays admissible for the audit
stage (rollback-safe: no v1-tagged artifact ever carries the new field;
the runner normalizes worker output to v2); a run_stage boundary test
proves warning persistence at the real entrypoint; the plan size table
was re-synced to the actual diff with this override rationale.

### Review round 1 (Codex)

Five precision/PII findings on the ported heuristics, all fixed: owner-routing
suppression requires relational routing language (bare topic nouns like
"billing" no longer suppress); product names (Resolution Audit/Snapshot) are
excluded from the answer-claim detector; qualifiers are evaluated per CLAUSE
so one qualified assertion cannot hide a separate unqualified one;
report-shape matching drops context-free nouns (draft/ranked/faqs);
international phone formats are redacted AND block the gate (+country-code
patterns; over-matching feeds redaction/blocking, both fail-closed).
The httpx workflow dependency this PR's CI surfaced was split into
PR #2183 (trusted-base execution means an in-PR workflow edit can never
fix its own CI) and is merged on main.

### Files touched

- `atlas_brain/schemas/content_factory.py`
- `atlas_brain/services/content_factory_copy_verification.py`
- `atlas_brain/services/content_factory_runner.py`
- `atlas_brain/services/content_factory_store.py`
- `plans/PR-CF-Advisory-Warning-Layer.md`
- `tests/test_content_factory_copy_verification.py`
- `tests/test_content_factory_runner.py`
- `tests/test_content_factory_schemas.py`

## Mechanism

One producer, one optional contract field, one runner injection point.
The runner treats the checklist exactly like the verdict (deterministic,
overwritten, never worker-supplied), so the reviewing human reads trusted
advisory output in audit.json without any new gate.

## Intentional

- **CTA reminder is unconditional** -- faithful to the source tool; every
  audit carries at least one checklist line, so an empty list is not a
  "clean" signal and is not treated as one anywhere.
- **Warnings never block** -- no validator references them; blocking
  advisory content would collapse the gate/checklist distinction #2136
  drew on purpose.
- **Evidence-free warnings over redaction** (supersedes the interim
  pre-redaction design): no draft text is persisted at all, so PII
  safety holds by construction, enforced again at the schema choke
  point -- not by any redaction pattern's completeness.

## Deferred

- #2136 item 4 (catalogue growth) remains standing operator policy.
- Rendering warnings in any UI/manifest surface (Phase 7 observability).

Parked hardening: none new.

## Verification

- Content-factory suites: 245 passed (280 with adjacent suites) (12 new advisory tests, 2 new runner
  tests). Adjacent `tests/test_leads_intake.py` green (187 combined).
- `python -m py_compile` on the three touched modules.
- NOT run: live OWUI worker pass (advisory output shape is fully covered
  by unit tests; next live pipeline run will carry the checklist).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/content_factory.py` | 88 |
| `atlas_brain/services/content_factory_copy_verification.py` | 689 |
| `atlas_brain/services/content_factory_runner.py` | 28 |
| `atlas_brain/services/content_factory_store.py` | 17 |
| `plans/PR-CF-Advisory-Warning-Layer.md` | 392 |
| `tests/test_content_factory_copy_verification.py` | 1001 |
| `tests/test_content_factory_runner.py` | 62 |
| `tests/test_content_factory_schemas.py` | 73 |
| **Total** | **2350** |
