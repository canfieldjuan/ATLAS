# PR-Hardening-Drain

## Why this slice exists

#1319 item 4. `HARDENING.md` is an append-only register: sessions park non-blocking
hardening discoveries under dated `## YYYY-MM-DD` headings, and the file itself asks
to "Periodically drain stale entries or promote them into the debt register under
`docs/technical-debt/`." Nothing does that today, so the register grows without bound
and every session that scans it for same-lane parked items pays a growing orientation
tax -- the same unbounded-governance-file problem the plans-archive arc (#1319 slices
1-3) just solved for `plans/`. This slice ships the mechanical drain tool, mirroring
`scripts/archive_plans.py`.

It runs over the 400 LOC soft cap (see Estimated diff size): the drain tool needs
section-splitting + date partitioning + archive append (more than the plain file-move
in `archive_plans.py`), plus an 8-case fixture suite covering the fenced-example trap,
footer/preamble preservation, and no-op byte-identity. The overage is mechanism and
coverage, not extra scope.

## Scope (this PR)

Ownership lane: governance/hardening-drain
Slice phase: Workflow/process

1. Add `scripts/drain_hardening.py` with two modes: `drain` (move parked entries
   older than `--max-age-days` into the drain archive under `docs/technical-debt/`)
   and `check` (non-blocking warning over a line/age threshold; always exits 0).
2. Ship `tests/test_drain_hardening.py` and enroll it in CI beside the other
   governance-tooling fixtures.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Hardening-Drain.md`
- `scripts/drain_hardening.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_drain_hardening.py`

## Mechanism

Pure functions (`split_sections`, `parse_entries`, `partition_by_age`,
`render_hardening`, `append_to_archive`, `check_warnings`) make the behavior testable
on text + an injected `today`. Parsing is anchored on the literal `## Parked Items`
marker, so the example `## YYYY-MM-DD` heading inside the fenced "Entry Format" block
in the preamble is never mistaken for a real entry. `drain` partitions dated entries
by age, appends stale ones to the archive, and rewrites `HARDENING.md` from
preamble + kept entries + the standing footer note -- preserving everything that is
not a stale dated block. When nothing is stale it does not rewrite the file at all
(byte-identical). `check` is the forcing function: it warns over threshold and
returns 0, so it never fails a PR.

## Intentional

- **Tooling only; no content change.** The current register has one 8-day-old entry,
  so at the default 90-day age threshold `drain` is a no-op and `HARDENING.md` is
  untouched in this PR -- exactly like slice 1 shipped the archive tool before the
  sweep. The drain runs meaningfully once entries actually age out.
- **Age-based drain, preserving the footer + preamble.** Entries are date-stamped, so
  age is the natural staleness signal; the standing `> ATLAS-HARDENING.md` footer note
  and the preamble are structurally preserved, never archived.
- **Unparseable dates are kept**, never silently dropped.
- **Fixture enrolled in `scripts/run_extracted_pipeline_checks.sh`** beside
  `tests/test_archive_plans.py`, with the script added to the workflow path filter --
  shipping a fixture CI never runs would repeat the #1318 enrollment gap.

## Deferred

- **Wiring `check` into `local_pr_review.sh`** as a second non-blocking advisory
  (parallel to the plans-archive advisory from slice 3) -- a small follow-up.
- **#1319 item 5:** segregating one-shot scripts into `scripts/oneshot/`.
- Parked hardening: none.

## Verification

```bash
python -m pytest tests/test_drain_hardening.py -q   # 8 passed
python scripts/drain_hardening.py check             # OK (real file within thresholds)
python scripts/drain_hardening.py drain             # "nothing to drain" (no entry > 90d)
```

Verified locally: 8/8 fixtures pass; `check` reports OK and `drain` is a no-op on the
real `HARDENING.md` (file untouched); parser ignores the fenced example heading.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Hardening-Drain.md` | 93 |
| `scripts/drain_hardening.py` | 220 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_drain_hardening.py` | 171 |
| **Total** | **489** |
