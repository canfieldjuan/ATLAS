# PR-Deflection-Proven-Answer-Gate

## Why this slice exists

#1456 is a P0 launch-readiness issue: the paid deflection report currently
marks any non-empty `resolution_text` as `resolution_evidence`. That means
closure boilerplate such as "Customer did not respond, closing this out" or
internal notes such as "Escalated to T2 / refunded per policy 4.2" can become
paid, teaser-eligible, customer-facing answers.

This slice hardens the proven-answer gate before the #1440 real full-volume
paid run. It keeps the deterministic/no-LLM deflection lane intact and avoids
the open #1452 parser/full-volume submit PR.

Diff budget note: this PR is over the 400 LOC soft cap after the review-blocker
fixes because the gate, symmetric verb normalization, disposition-only note
filter, concrete-start action support, narrow synonym overlap support, and
failure-first fixtures need to land together. Splitting the past-tense
normalization, disposition-only rejection, valid "start the return"
restoration, or login/password-style synonym support from the gate would
knowingly leave either real support resolutions under-included or generic agent
status updates over-included.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Filter support-ticket resolution candidates before they can populate
   `resolution_text`, `evidence_group_key`, `resolution_evidence`, answer
   copy, or steps in `ticket_faq_markdown.py`.
2. Reject closure/disposition boilerplate, disposition-only agent update notes,
   and narrow internal-note patterns.
3. Require structural instruction-shape (not action-term/topic-overlap list
   membership) for a resolution to count as publishable answer evidence
   (operator decision #1466 Option 1).
4. Add failure-first fixtures proving boilerplate/internal notes and
   disposition-only agent updates stay `draft_needs_review`, plus positive
   fixtures proving real step-wise resolution evidence still gates, including
   past-tense agent language, concrete account-review steps, and common
   support synonym pairs such as login/password reset and receipt/invoice.

### Review Contract
- Acceptance criteria:
  - [ ] Closure boilerplate does not produce `resolution_evidence`, resolution
        source counts, customer-facing steps, or leaked answer copy.
  - [ ] Internal operational notes do not produce `resolution_evidence` or
        leak internal policy/escalation details into paid answers.
  - [ ] Disposition-only agent status updates such as reviewed/replied or
        checked/sent-update notes do not produce `resolution_evidence`,
        including phrasing where the update/reply appears before the recipient.
  - [ ] A genuine step-wise, on-topic resolution still gates as
        `resolution_evidence`, including "start the return" style portal
        instructions.
  - [ ] Realistic answers across varied verbs (incl. verbs absent from any
        list -- schedule/pin/narrow/forward/assign/rename/...) and symptom/fix
        synonym pairs (login/SSO, crash/cache, charged-twice/refund) PUBLISH;
        list membership is no longer required.
  - [ ] Question-topic overlap is demoted to an advisory `topic_aligned`
        signal (computed, surfaced for a future confidence layer) and no
        longer hard-rejects an off-topic-but-genuine instruction. (Operator
        Option 1: list-widening did not converge; the honesty floor is the
        reject filters, not topic membership.)
  - [ ] The filter is deterministic and lives in the extracted package; no
        LLM/Ollama/local model path is introduced.
- Affected surfaces: deflection FAQ markdown/report item construction and its
  extracted-checks test suite.
- Risk areas: over-filtering short legitimate resolutions, under-filtering
  generic closure text, grouping drift via bogus `evidence_group_key`.
- Reviewer rules triggered: R1, R2, R9, R10, R13.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Proven-Answer-Gate.md`
- `tests/test_build_deflection_messy_csv_fixtures.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The resolution-text normalizer is the earliest point where raw evidence/opportunity
fields are normalized into the row-level `resolution_text` used by grouping,
answer status, answer copy, and resolution source counts. This slice adds a
deterministic publishability predicate there.

**The gate is reject-known-bad + structural instruction-shape, not
require-membership (operator decision, #1466 Option 1).** Four earlier rounds
gated on membership in enumerated lists (action terms, disposition set,
topic-equivalence map); held-out probes showed those lists still rejected the
majority of realistic publishable answers and per-example list-widening did not
converge. The predicate now is:

1. compact the candidate text;
2. reject narrow closure/disposition boilerplate (`_RESOLUTION_CLOSURE_BOILERPLATE_RE`);
3. reject narrow internal-note patterns -- tier escalations, numbered internal
   policy references (`_RESOLUTION_INTERNAL_NOTE_RE`);
4. require minimum substance (`len(resolution_tokens) >= 3`);
5. reject disposition-only customer-update notes via the kept two-factor
   `_resolution_text_is_disposition_only` (weak status verbs + disposition
   regex), e.g. `sent an update to the customer`;
6. **positive gate -- the text must look like an instruction**
   (`_resolution_text_looks_instructional`): numbered/step structure, a UI
   navigation path ("Settings then Phone"), or a sentence that opens with an
   imperative verb. The verb may be a known action term (stemmed, so past-tense
   "enabled"/"configured"/"updated" register) OR an unknown verb in the
   "<verb> the/your/a <noun>" imperative shape (so "schedule", "pin", "narrow",
   "forward", "assign", "rename", "archive", "submit", "cancel", ... publish
   without ever being listed). Disposition sentences are disqualified
   per-sentence, and a first-person subject ("We received your request") must
   use a known action verb -- the generic object shape is imperative-position
   only -- so narration is not mistaken for an instruction.
7. **Two structural per-sentence disqualifiers** added in round 7 after running
   the gate over real support replies (Twitter brand replies + Ubuntu QA):
   - **Contact-channel redirection** (`_RESOLUTION_CONTACT_REDIRECT_RE`): a
     hand-off to a human over a private channel ("send us a DM", "message us",
     "shoot me a private message") is imperative-shaped but answers nothing --
     the disposition reject's imperative-phrased sibling. Narrow on purpose
     (matches "<verb> us/me" and the DM/PM/private-message redirect nouns, not
     legitimate steps that merely contain "send"/"message"), and checked against
     the sentence's lead clause only, so a real step with a trailing redirect
     fallback ("Reset the cache, then DM us") still publishes on the step.
   - **Answer-is-a-question**: the sentence splitter now retains terminators, so
     a sentence ending in "?" ("Did the lights change on the router?") -- a
     diagnostic prompt back to the requester -- is skipped, not parsed as a step.

The action-term-membership and question-topic-overlap checks are **demoted, not
deleted**, to advisory signals (`_resolution_advisory_signals`): the maps stay
live for a future confidence surface but never block a structurally-valid,
non-boilerplate instruction. If a candidate fails, the normalizer returns an
empty string, keeping bogus text out of `evidence_group_key`, collected
resolution texts, `resolution_source_count`, resolution-evidence scope, steps,
and paid answer summaries.

## Intentional

- No LLM judge in this slice. The launch blocker is that obvious non-answer
  text is currently trusted as proven evidence; a deterministic fail-closed
  filter closes that immediate risk without adding cost or model dependency.
- The internal-note matcher stays narrow. It rejects concrete operational
  patterns named in #1456 but avoids broad words such as "policy" or
  "escalation" by themselves, because those can appear in legitimate
  customer-facing instructions.
- The disposition-only guard is constrained to weak action-token sets plus
  customer-update/reply wording. It does not reject concrete step-wise account
  fixes such as opening invoices and updating a payment method, or concrete
  return-flow instructions such as starting a return in a portal.
- Topic alignment is deliberately NOT a gate (operator Option 1). The
  action-term and topic-equivalence maps are kept and computed as advisory
  signals (`_resolution_advisory_signals.topic_aligned` /
  `has_known_action_term`) for a future confidence surface, but an off-topic
  yet genuine instruction publishes. The tradeoff is conscious: four rounds of
  topic/verb list-widening still over-rejected real answers, so the honesty
  floor is the reject filters (boilerplate / internal-note / disposition-only /
  sub-3-token), not membership.
- This does not solve the separate #1460 fixed-bucket over-merge issue.
- The contact-redirect disqualifier is deliberately conservative: it fires on
  the lead clause, so a redirect leading a sentence rejects even if a real step
  trails it ("DM us, then reset"). Under-publishing a real answer is the safe
  direction for a proven-answer gate; the common shape (step first, redirect
  fallback) is preserved.

## Deferred

- #1460 remains the broader within-intent clustering/subcluster fix.
- A future robust-testing slice can add a larger resolution-quality corpus and
  calibrate thresholds against real help-desk exports.
- A separate over-accept class observed in the real-corpus run -- interjection /
  adjective sentence leads ("Guac on!", "Love the aesthetic", "Glad to know")
  parsing as "<verb> <object>" -- is out of scope here. It is Twitter-pleasantry
  noise that does not appear in real Zendesk `resolution_notes`; chasing it risks
  over-fitting. Left for the same future calibration slice.
- The structured-outcome factor (read `status` / `reopens` /
  `satisfaction_score`; gate `proven` on answer-AND-outcome) and a shape-aware
  importer (Zendesk full-export vs metrics-only CSV) -- per discussion #1507 --
  are the larger honesty win and a separate slice. This slice hardens only the
  answer-text factor.

Parked hardening: none.

## Verification

- Option-1 inversion (operator decision): held-out probe of
  `_resolution_text_is_publishable` -- 20/20 realistic answers publish
  (varied verbs incl. schedule/pin/narrow/forward/assign/rename/archive/submit/
  cancel + symptom/fix synonym pairs login-SSO/crash-cache/charged-refund +
  past-tense/first-person + numbered steps + "to fix this," preamble + round-7
  redirect/question guards: real "send"/"message" steps and step-then-fallback
  still publish), 19/19 honesty-floor non-answers rejected (incl. round-7
  contact-redirect + non-copula question), 0 misses either direction. The corpus
  is pinned into the test file (`_HELD_OUT_PUBLISHABLE` / `_HELD_OUT_REJECTED`).
- Round-7 real-corpus calibration: ran the gate over 400 real Twitter brand
  replies (reject-side) and 400 real Ubuntu QA responses (recall-side). The two
  new disqualifiers drop Twitter publish 15% -> 10% (the DM/clarifying-question
  false positives) while Ubuntu real-instruction publish holds 9% -> 8%.
- `drafted_answer_count` does not regress: the resolution-bearing live-proof,
  saas-demo, and macro-writeback fixtures keep their drafted counts;
  measured-repetition merged-state language-filter test stays green.
- Full extracted gauntlet (`scripts/run_extracted_pipeline_checks.sh`) -- 3952
  passed, 10 skipped, 0 failed (inversion + measured-repetition reconciliation
  + round-7 redirect/question disqualifiers).
- Focused pytest for `tests/test_extracted_ticket_faq_markdown.py`.
  - Passed, 219 tests.
- Downstream pytest targets in `tests/test_content_ops_deflection_resolution_live_proof.py`,
  `tests/test_extracted_ticket_faq_macro_writeback.py`,
  `tests/test_extracted_ticket_faq_output_ingestion.py`, and
  `tests/test_extracted_content_ops_live_execute_harness.py`
  - Passed, 4 tests.
- `./scripts/validate_extracted_content_pipeline.sh`
  - Passed.
- `./extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Passed.
- `./scripts/audit_extracted_standalone.py --fail-on-debt`
  - Passed.
- `./scripts/check_ascii_python.sh`
  - Passed.
- Python compile check for `extracted_content_pipeline/ticket_faq_markdown.py`
  and `tests/test_extracted_ticket_faq_markdown.py`
  - Passed.
- `./scripts/run_extracted_pipeline_checks.sh`
  - Passed, 3952 passed, 10 skipped; existing torch/pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 417 |
| `plans/PR-Deflection-Proven-Answer-Gate.md` | 237 |
| `tests/test_build_deflection_messy_csv_fixtures.py` | 11 |
| `tests/test_extracted_ticket_faq_markdown.py` | 475 |
| **Total** | **1140** |
