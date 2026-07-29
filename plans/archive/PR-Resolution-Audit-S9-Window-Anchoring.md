# PR-Resolution-Audit-S9-Window-Anchoring

Ownership lane: resolution-audit-csv

## Why this slice exists

S9 of the #1993 remediation arc (closes #2056, filed from the S7 review).
A stale-but-valid upload generates a zero-ticket-source FAQ: the package
enables the downstream date window (`inputs["faq_window_days"]`) without
anchoring it, and the FAQ builder's `_date_window` falls back to
`date.today()` when no as-of date is provided, so every normalized source
row of e.g. a January export processed in July falls outside the window
and the generated paid FAQ has zero ticket sources. Reproduced before
coding: 5 rows dated 180 days ago, `window_days=30` -> window enabled,
`faq_as_of_date` absent -> builder returns 0 sources / 0 items;
data-anchored, the same rows return 5 sources / 1 item. Pre-existing
before S7 for any complete US-parseable stale upload; S7's tolerant
window widened the population.

### Problem-derived contract

- Root cause: the window's ANCHOR was never decided at the boundary that
  decides the window's VALIDITY. The receiving plumbing already exists
  end-to-end -- `generation_plan.py` reads and validates `faq_as_of_date`
  (YYYY-MM-DD, requires `faq_window_days`), threads it to
  `TicketFAQMarkdownService.generate` -> `build_ticket_faq_markdown` ->
  `_date_window` -- the package just never emits it.
- Correct fix must touch/change:
  1. `extracted_content_pipeline/support_ticket_input_package.py`
     (owned): `_source_date_diagnostics` captures `latest_source_date`
     (max parsed `created_at`) in its existing per-row loop; when
     `has_valid_date_window`, the package emits
     `inputs["faq_as_of_date"] = latest_source_date.isoformat()`
     alongside `faq_window_days`. The window then means "the last N days
     OF THE DATA" -- which is what `source_period` ("Last N days of
     support tickets") already claims about data it derives from.
  2. Tests: the pre-coding repro pinned both directions (stale upload's
     own emitted inputs keep its sources; fresh upload unchanged);
     `faq_as_of_date` present iff the window is valid; emitted format
     passes the existing `generation_plan` YYYY-MM-DD validation.
- Must not change:
  - `window_days` semantics (caller-declared length) and the 0.9
    coverage rule (S7).
  - `_date_window`'s `date.today()` fallback for callers that pass
    `window_days` without an anchor (other, non-package callers keep
    their behavior).
  - Caller precedence: the live deflection-submit route overlays only
    title/company/contact/platform on `package.inputs`
    (`control_surfaces.py` `execute_inputs`), so the package emission
    cannot clobber an operator-supplied value on any existing path.
  - Report content/shape, money, scrub, gate mechanics.

Review-loop guard: HARD cap of 3 Codex rounds counted by ROUNDS; at cap,
fix what is written, resolve/waive remaining threads, merge on
required-green.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 3

1. `extracted_content_pipeline/support_ticket_input_package.py` --
   latest-date capture + `faq_as_of_date` emission.
2. `tests/test_support_ticket_dates_window.py` -- both-direction pins.
3. This plan doc.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S9-Window-Anchoring.md`
- `tests/test_support_ticket_dates_window.py`

### Review Contract

Acceptance criteria:
1. A stale-but-valid upload's emitted inputs (`faq_window_days` +
   `faq_as_of_date`) drive the FAQ builder to keep its own sources; the
   zero-source path is closed.
2. `faq_as_of_date` is emitted iff the dated window is valid, equals the
   newest parsed source date, and passes the existing generation-plan
   YYYY-MM-DD validation.
3. Dateless/invalid-window uploads emit neither key (unchanged).
4. No downstream behavior change for callers that never receive a
   package-emitted anchor.

Reachability proof: the emission rides the same `package.inputs` dict the
live deflection-submit route feeds to `execute_generation`; the e2e test
drives the builder with the package's own emitted values.

Affected surfaces: package inputs, FAQ date-window basis for
package-driven runs.

Risk areas: anchor semantics ("Last N days" of data vs of wall clock) --
the data anchor matches what `source_period` already claims, and the
wall-clock anchor produced an empty paid report, which no reading of the
product intends.

Reviewer rules triggered: R1, R2, R10, R14 (extracted package change; test evidence; window-basis gate input change; both-direction probes).

## Mechanism

`_source_date_diagnostics` already parses every row's `created_at`; it now
also tracks the max parsed date and returns it as `latest_source_date`.
The package builder emits `faq_as_of_date` next to the existing
`faq_window_days` emission when the window is valid. Everything
downstream is existing, already-validated plumbing.

## Intentional

- Anchor decided at admission, once -- the same principle as S7's ISO
  canonicalization: the component that decides window VALIDITY decides
  the window ANCHOR, instead of every downstream consumer guessing.
- Data-max anchor (not upload timestamp): the report's claims are about
  the data's own recency; wall-clock anchoring is what produced the
  empty-report defect.

## Deferred

- Surfacing the anchored window dates in report copy (product wording;
  operator, #1993).
- Delta-path window semantics (separate lane).

## Verification

- `python -m pytest tests/test_support_ticket_dates_window.py -q`
- Adjacent: input-package + generation-plan + smoke suites.
- Full CI mirror: bash `scripts/run_extracted_pipeline_checks.sh`
- Pre-coding repro pinned as the e2e test (0 sources -> 5 sources).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 15 |
| `plans/PR-Resolution-Audit-S9-Window-Anchoring.md` | 138 |
| `tests/test_support_ticket_dates_window.py` | 61 |
| **Total** | **214** |
