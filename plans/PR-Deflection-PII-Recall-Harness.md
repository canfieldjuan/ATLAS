# PR-Deflection-PII-Recall-Harness

## Why this slice exists

#1742 says the next step after the surrogate corpus is the measurement harness:
run the deflection report path and score PII recall/precision per output
surface. PR #1744 landed the safe persisted corpus boundary, but it still does
not tell us whether the scrubbed report catches the labeled surrogate PII that
actually reaches a free or paid surface.

Root cause: the privacy arc now has a versionable eval input, but no
surface-attributed scorer. Without a scorer, future real-corpus runs can only be
inspected manually and the headline KPI from #1742 cannot be computed. This PR
fixes the root for harness availability by adding the scoring tool and a
CI-enrolled smoke on the committed tiny surrogate corpus. It does not claim real
outcome quality until an operator-supplied larger corpus is run.

This PR is over the 400 LOC soft cap because the scorer has to exercise four
real report surfaces, keep baseline-vs-scrubbed accounting separate, emit safe
operator summaries, and prove the failure branches with CI-enrolled tests in the
same slice. Splitting the script from the tests or CI enrollment would recreate
the false-green harness problem this lane is meant to close.

Review-fix root causes: the first pushed scorer treated the paid PDF renderer as
a required module-level dependency, so an extracted-checks environment without
`fpdf` failed at test collection before the harness could report anything. It
also validated only that each ticket had a non-empty labels list, so malformed
label or must-survive records could be counted in input totals but skipped from
the measured denominator. This update fixes those roots: paid PDF scoring is
optional when the renderer dependency is unavailable, and malformed measured
records fail closed before any surface scoring runs.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice

1. Add `scripts/score_deflection_pii_recall.py` to load a
   `deflection_pii_eval_corpus.v1` artifact, build a production deflection
   report artifact from its tickets, scrub it, project the free snapshot/teaser,
   render the paid PDF path, and score labeled surrogate spans by surface.
2. Score recall by PII class x surface for `free_snapshot`, `free_teaser`,
   `paid_artifact`, and `paid_pdf`, with a headline free-surface high-severity
   leak count.
3. Score the first precision guardrail by checking must-survive tokens per
   surface.
4. Emit a sanitized JSON summary with counts, cue-prefixed vs cue-less
   `person_name` split, and leak samples by surrogate id rather than raw text.
5. Add tests proving the harness math on the tiny committed surrogate corpus and
   on forced leak / must-survive violation fixtures.
6. Keep paid PDF scoring optional when `fpdf` is unavailable and report that
   skipped status instead of failing module import.
7. Reject malformed label and must-survive records with structured errors before
   computing recall or precision.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-PII-Recall-Harness.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_score_deflection_pii_recall.py`

### Review Contract

- Acceptance criteria:
  - [ ] The scorer consumes the #1744 surrogate artifact schema and rejects
        malformed/empty inputs with structured errors.
  - [ ] The scorer runs through the existing deflection report build, scrub, free
        snapshot/teaser projection, and paid PDF render path rather than scoring
        detector regexes in isolation.
  - [ ] Output reports recall per class x surface and computes the #1742
        headline free-surface high-severity leak count.
  - [ ] `person_name` recall is split by `cue_prefixed` vs `cue_less`.
  - [ ] Must-survive violations are reported per surface.
  - [ ] Leak samples are sanitized: class, severity, surface, and surrogate id
        are allowed; span text is not emitted by default.
  - [ ] Missing `fpdf` does not fail collection; the paid PDF surface reports a
        skipped status while the other surfaces still score.
  - [ ] Labels and must-survive records with missing/non-string spans fail closed
        with structured errors before scoring.
  - [ ] No raw ticket dump, auto-labeler, NER/model change, threshold gate, or
        report model change lands in this PR.
- Affected surfaces: one scorer script, one focused test file, CI enrollment,
  and this plan.
- Risk areas: accidentally scoring scrub regexes instead of the production path,
  treating non-reaching labels as leaks, leaking sample text from the scorer, or
  colliding with #1745's report-model lane.
- Reviewer rules triggered: R1, R2, R6, R9, R10, R12, R14.

## Mechanism

The scorer builds a small `TicketFAQMarkdownResult` from each surrogate eval
ticket so the existing report renderer sees answer, step, question,
customer-wording, source-id, and evidence fields. It then creates two surface
families:

- baseline surfaces from the pre-scrub artifact, snapshot/teaser, paid artifact,
  and PDF source text;
- scrubbed surfaces from the same artifact after `scrub_deflection_report_payload`
  and `build_deflection_snapshot` have run.

For each label, the scorer counts a surface only when the surrogate span reaches
that baseline surface. It counts a leak only when that same surrogate span is
still present in the scrubbed surface. Must-survive tokens use the same
surface-reaching rule in reverse: a token that reached a baseline surface must
still be present after scrubbing.

The paid PDF path renders bytes through the existing Atlas PDF renderer and uses
the renderer's curated PDF source text as the scoreable text surface. Actual PDF
text extraction is deferred; the render step still proves the paid PDF path is
exercised.

If the optional PDF renderer dependency is missing, the scorer leaves the
`paid_pdf` score surface empty and records a `missing_optional_dependency:fpdf`
skip reason in `surface_generation.paid_pdf`. That keeps the extracted-checks
lane importable without weakening local or full-env PDF proof when the renderer
is installed.

Corpus validation runs before any report build. Each ticket must carry at least
one label, and each label / must-survive record must be an object with a
non-empty string span. Invalid records return structured error codes and indices
without echoing the raw span or ticket body.

## Intentional

- No larger real gold corpus in this PR. #1742 keeps that operator-gated; this
  PR makes the harness ready for it.
- No threshold gate. The output computes the headline KPI, but CI remains a
  harness smoke until corpus size and thresholds are ratified.
- No detector, scrubber, report model, or PDF renderer changes. #1745 currently
  owns nearby report model work; this slice only consumes the existing surfaces.
- No extracted-checks install change for `fpdf`. The dependency is only needed
  for the optional paid-PDF surface; adding it to the whole lane would hide a
  harness import-boundary bug.
- No raw leak text in samples by default, even though the committed fixture uses
  synthetic surrogates. The same output shape must be safe for later
  operator-derived artifacts.
- Paid PDF scoring uses the renderer's curated PDF source text plus a successful
  PDF byte render, not extracted text from the generated PDF bytes.

## Deferred

- Operator-supplied larger gold corpus derived from real tickets.
- Threshold configuration and the later advisory-to-gating switch.
- Full over-redaction token-rate metric beyond must-survive violations.
- PDF byte text extraction if we need to validate text after PDF serialization,
  not only the renderer source text.

Parked hardening: none.

## Verification

- python -m py_compile scripts/score_deflection_pii_recall.py tests/test_score_deflection_pii_recall.py
- pytest tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py -- 25 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok, paid PDF rendered (5396 bytes), current tiny-corpus measurement reports free_high_severity_leak_count=2 and 3 must-survive precision violations.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Deflection-PII-Recall-Harness.md` | 169 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/score_deflection_pii_recall.py` | 554 |
| `tests/test_score_deflection_pii_recall.py` | 272 |
| **Total** | **1000** |
