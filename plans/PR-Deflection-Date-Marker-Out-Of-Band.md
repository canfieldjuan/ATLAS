# PR-Deflection-Date-Marker-Out-Of-Band

## Why this slice exists

Follow-up to the #1519 reviewer NIT. `build_support_ticket_input_package`
stamped an internal `_date_source_present` marker onto each normalized row dict
to record "this upload row carried a date column (even if the cell was blank)" --
a signal a parseable created_at alone cannot capture. That marker then rode
through clustering and every normalized_rows consumer, and was stripped only at
the single source_material egress (`_public_ticket_row`); only that egress was
guarded by a test. A future whole-row echo of a normalized row could leak the
marker. This slice carries the signal out-of-band so no internal marker rides on
the shared row dicts.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. `extracted_content_pipeline/support_ticket_input_package.py`: compute a
   source-date-signal count in the normalization loop of
   `build_support_ticket_input_package` (one increment per appended row where
   `_has_any_key` matches the date keys), instead of stamping the marker.
2. Remove the marker stamp from `_normalize_ticket_row`.
3. `_source_date_diagnostics` gains an out-of-band keyword count; when omitted it
   falls back to a created_at-only signal count (the only other caller,
   `_all_rows_have_dates`, reads only the missing count).
4. Pass the count at the single live call site.
5. `tests/test_extracted_support_ticket_input_package.py`: add a regression test
   asserting the blank-date-column warning still fires and no `_`-prefixed marker
   rides on the exported source rows.

### Review Contract

- Acceptance criteria:
  - [ ] No `_date_source_present` (or any `_`-prefixed marker) appears on the
        exported source rows.
  - [ ] A blank date-column upload still disables the dated window and emits
        `support_ticket_date_window_disabled` (behavior preserved).
  - [ ] A no-date-column upload still emits no date-window warning.
  - [ ] The out-of-band count equals the pre-change marker-based count
        (clustering is count/order-preserving, so the pre-cluster loop count ==
        the old post-cluster marker count).
  - [ ] Existing input-package + FAQ-markdown suites stay green.
- Affected surfaces: `build_support_ticket_input_package` normalization loop and
  `_source_date_diagnostics` internals, plus their test. No report
  snapshot/teaser/ladder/pricing shape change; no DB; no public API change.
- Risk areas: behavior preservation of the date-window warning trigger;
  determinism; the `_all_rows_have_dates` fallback path.
- Reviewer rules triggered: R1 (extracted_* change), R2 (failure-branch /
  fixtures), R10 (maintainability).

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Deflection-Date-Marker-Out-Of-Band.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The marker was exactly `_has_any_key` over the date keys on the raw row. The
created_at field is set only from a non-empty date-key value, so a parseable
created_at always implies the date key is present -- the marker supersets a
non-empty created_at, and the old `has_date_signal = marker OR created_at` equals
the marker. Computing the same key check per appended row in the normalization
loop reproduces the marker count exactly. `assign_support_ticket_clusters_with_diagnostics`
emits one copied row per input row in order (no filter, no reorder), so the count
is identical whether computed before or after clustering. The count is then handed
to `_source_date_diagnostics` out-of-band; the marker is gone from the row dicts
entirely.

## Intentional

- Out-of-band integer, not a parallel per-row structure: the signal is
  upload-level (a count), and clustering is count/order-preserving, so a single
  accumulator suffices and survives clustering without any row carrying state.
- `_source_date_diagnostics` keeps an optional keyword count with a created_at-only
  fallback so it stays callable standalone; the `_all_rows_have_dates` path is
  unchanged (it reads only the missing count).
- No touch to `_all_rows_have_dates` (pre-existing, uncalled) -- out of scope.

## Deferred

- Removing the dead `_all_rows_have_dates` helper -- separate cleanup slice.

Parked hardening: none.

## Verification

Ran locally -- 285 passed (incl. the new
test_support_ticket_date_signal_is_carried_out_of_band_not_on_rows); py_compile,
ASCII, and the extracted CI mirror all clean:

```
pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py
python -m py_compile extracted_content_pipeline/support_ticket_input_package.py tests/test_extracted_support_ticket_input_package.py
bash scripts/check_ascii_python.sh
bash scripts/run_extracted_pipeline_checks.sh
```

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | ~30 |
| `tests/test_extracted_support_ticket_input_package.py` | ~25 |
| `plans/PR-Deflection-Date-Marker-Out-Of-Band.md` | ~90 |
| **Total** | **~145** |
