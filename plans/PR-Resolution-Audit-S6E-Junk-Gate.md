# PR-Resolution-Audit-S6E-Junk-Gate

Ownership lane: resolution-audit-csv

## Why this slice exists

Issue #2049 (S6E), parent #1993, launch blocker #5 (F2). The audit's live run
proved: 5 real "reset my password" tickets + 6 identical "Automatic reply:
Out of Office" rows produce a JUNK cluster ranked #1 with `ticket_count=6`,
owning the top priority-fix/drafted-resolution recommendation and inflating
the headline support tax by ~2x (F2, `docs/audits/resolution-audit-csv/FINDINGS.md`).
No junk gate exists anywhere before a ticket counts toward a billed cluster,
and S5's semantic clustering makes byte-identical junk cluster TIGHTER, so
the fix must be admission-side, upstream of clustering, at the same boundary
S6A guards. The original repro fixture lives in gitignored `_audit_scratch/`,
so this slice commits an equivalent F2 acceptance test.

### Problem-derived contract

- Root cause: no spam/auto-reply gate before a ticket counts toward a billed
  cluster; machine-generated rows (auto-replies, out-of-office, delivery
  bounces) are admitted as customer evidence and form billed clusters.
- Correct fix must touch/change:
  1. New owned module `extracted_content_pipeline/support_ticket_junk.py`
     (stdlib-only, mirrors `support_ticket_privacy.py`):
     `support_ticket_row_is_junk(subject, body_lines, had_source_text)`
     returning a bounded reason code (`auto_reply` | `bounce` |
     `no_new_content`) or None. Closed STRUCTURAL rules, not phrase
     enumeration: machine-generated mail betrays itself by position and
     shape -- (a) generator SUBJECT PREFIX forms with the trailing
     colon/bracket position as the discriminator ("Automatic reply:",
     "Out of Office:", "Undeliverable:", delivery-status families, optional
     leading [EXT]-style brackets); (b) FIRST-PERSON assertion shapes matched
     as whole admitted lines on the S6B line seam ("I am out of the office
     until...", "This is an automated response...", never ending in "?");
     (c) `no_new_content` for rows whose admitted text empties under line
     hygiene with no subject content.
  2. Gate wired in `_normalize_ticket_row` after text admission (beside the
     S6A privacy check), returning a `_junk_reason` marker the caller counts
     and drops before clustering.
  3. Diagnostics: `junk_excluded_count` + `junk_excluded_reasons` in package
     METADATA (the `source_id_fallback_count` surface) plus a
     `support_ticket_junk_excluded` warning -- counts, never content.
  4. Tests: the rebuilt F2 acceptance shape through the real
     `build_support_ticket_input_package` (junk forms no cluster, 5 real rows
     admitted, diagnostics = 6 auto_reply); BOTH error directions (customer
     tickets ABOUT auto-reply/out-of-office features -- "How do I set an out
     of office auto-reply?", "Out of office not working" -- provably pass);
     generator-subject and assertion-line classes parametrized; bounces;
     all-quote/no-content rows; row accounting still sums; CI enrollment in
     the same PR.
- Must not change:
  - Privacy admission (`support_ticket_privacy.py`, S6A).
  - The S6B extractor/scanner (`support_ticket_clustering.py`) -- this slice
    CONSUMES `support_ticket_plain_text_lines`, its first runtime consumer.
  - Clustering thresholds/semantics (S5).
  - Evidence-tier/status logic (S6D #2045 / M9 #2050).
  - Product shape: no new report fields; diagnostics are operator-facing
    metadata/warnings only.
  - Final-output scrubber grammar.
  - Zendesk `_AUTO_ACK_PATTERNS` (comment-level auto-ack filtering; separate
    concern, untouched).

Round-1 review refinements (all six verified failing first; correctness
gaps, not phrasing probes): comment-only rows join the gate's body input
(public comment text was admitted by `_ticket_text` but invisible to the
gate); the was-not-delivered bounce line matches (`(?:be\s+)?` was
mandatory); generator delimiters are colon-only so customer separators
("Out of office - not working") stay admitted; bounce subjects need their
generator delimiter ("Undeliverable emails are not reaching customers"
admitted); any interrogative line vetoes body-shape junk so mixed tickets
quoting a template stay admitted; generator prefixes are also checked on
the first body line for subject-in-text exports.

Round-2 review refinements (all four verified failing first): a question
ANYWHERE in a line vetoes body-shape junk ("Why is it not sending? Thanks");
comments reach the gate line-preserved per raw comment (`_raw_comment_texts`)
instead of pre-compacted, so multiline comment-only auto-replies classify;
the delivery-has-failed subject branch is anchored to the exact DSN sentence
so customer prose stays admitted; labeled subject lines
("Subject: Automatic reply: ...") in text exports classify. The
maturity-sweep ratchet's new-file finding is accepted via its documented
baseline flow: the module scores on regex density, and its behavior is
pinned by 34 both-direction tests.

Review-loop guard (the #2053 lesson): if review exceeds ~3 rounds of novel
junk-PHRASING probes, residuals are class-waived against this closed-rule
contract (structural position/shape rules; vocabulary is deliberately not
enumerable) rather than extended per spelling.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 8

1. `extracted_content_pipeline/support_ticket_junk.py` -- the gate module.
2. `extracted_content_pipeline/support_ticket_input_package.py` -- wiring +
   diagnostics (imports, `_normalize_ticket_row` check, caller counting,
   metadata + warning).
3. `extracted_content_pipeline/manifest.json` -- owned-file registration.
4. `tests/test_support_ticket_junk.py` -- F2 acceptance + class tests.
5. `scripts/run_extracted_pipeline_checks.sh` -- CI enrollment, same PR.
6. `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` --
   tracker discipline: S6B #2053 ticked merged; ledger row 5 marked in
   progress with this slice.
7. `tests/maturity_sweep/baseline_extracted_content_pipeline.json` --
   ratchet acceptance for the new module via the sweep's documented
   baseline flow.
8. This plan doc.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/support_ticket_junk.py`
- `plans/PR-Resolution-Audit-S6E-Junk-Gate.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
- `tests/test_support_ticket_junk.py`

### Review Contract

Acceptance criteria:
1. The F2 shape (5 real + 6 identical OOO auto-replies) yields NO junk
   cluster, 5 admitted rows, and `junk_excluded_reasons == {"auto_reply": 6}`
   through the real package builder.
2. Customer tickets about auto-reply/out-of-office features are admitted
   (both cited shapes pass; question/description forms never match the
   first-person assertion patterns).
3. Diagnostics expose counts and bounded reason codes, never row content.
4. Row accounting still sums (included + skipped == source rows).
5. The gate runs at row admission, upstream of clustering, beside S6A.

Reachability proof: wired into `_normalize_ticket_row` on the live
`build_support_ticket_input_package` path -- the F2 acceptance test exercises
the real builder end-to-end and asserts cluster output, not just the
predicate. This is also the first runtime consumer of the S6B line seam.

Affected surfaces: support-ticket row admission feeding clustering, evidence,
and report generation; package metadata/warnings diagnostics.

Risk areas: over-exclusion of real tickets mentioning auto-reply features
(guarded by assertion-shape rules + both-direction tests); junk vocabulary
gaps (bounded by the closed-rule contract; residual junk is quantified by
diagnostics).

Reviewer rules triggered: R1, R2, R10, R14 (extracted package change; test evidence; guard behavior; parser admission rule probed both directions).

## Mechanism

`support_ticket_row_is_junk` classifies by structure: subject generator
prefixes (colon/bracket position), first-person whole-line assertions on the
line-preserving body, and empty-after-hygiene rows. `_normalize_ticket_row`
returns a `{"_junk_reason": code}` marker; the package builder counts reasons
into `junk_excluded_counts`, drops the rows before clustering, and reports
`junk_excluded_count`/`junk_excluded_reasons` in metadata plus one warning.

## Intentional

- Admission-side gate (upstream of clustering), not a cluster-side filter:
  fixes the root (junk counts toward billed evidence) rather than hiding the
  symptom cluster.
- Structural rules over vocabulary: subject-prefix POSITION and
  first-person-assertion SHAPE are closed classes; junk phrasing is not.
- `no_new_content` fires only for empty-subject rows emptied by hygiene;
  subject-only rows stay admitted (S6D owns subject-only evidence semantics).
- Reasons are a bounded enum; diagnostics never carry content.

## Deferred

- S6C (#2044) scalar-history state machine; S6D (#2045) evidence tier; M9
  (#2050) status synonyms; S8a/S8b (#2051/#2052) runtime scorecard + money
  guard.
- Any junk-vocabulary extension beyond the structural classes (class-waived
  per the review-loop guard unless a structural gap is shown).

## Verification

- `python -m pytest tests/test_support_ticket_junk.py -q` (23 passed)
- Full CI mirror scripts/run_extracted_pipeline_checks.sh (5864 passed, 21
  skipped)
- scripts/validate_extracted_content_pipeline.sh (mapped files match)
- F2 acceptance verified live before tests: junk cluster absent, 6 excluded
  with reason auto_reply, feature questions admitted.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 4 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 61 |
| `extracted_content_pipeline/support_ticket_junk.py` | 154 |
| `plans/PR-Resolution-Audit-S6E-Junk-Gate.md` | 200 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/maturity_sweep/baseline_extracted_content_pipeline.json` | 28 |
| `tests/test_support_ticket_junk.py` | 269 |
| **Total** | **720** |
