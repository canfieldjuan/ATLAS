# SUMMARY -- Resolution Audit CSV pipeline audit

One page. The arc (#1956) audited the full path from CSV upload to the delivered
Snapshot and Full Report, empirically, against `main`. Every claim below is backed
by a reproduction in FINDINGS.md; the criticals were independently re-run by the reviewer.

## Should this block sending reports to real customers? YES -- for the paid deflection number, until two fixes land.

On a clean, well-formed Zendesk export the pipeline is fine. On a **realistic messy
inbox** the headline deflection number -- the thing the customer pays for -- is wrong
in **both directions at once**, and the two errors do not cancel:

- It is **undercounted ~44-58%** because same-intent tickets phrased differently are
  scattered and dropped (F1/F3).
- It is **inflated >2x** because repeated auto-reply junk becomes its own #1 billed
  "repeat question" (F2).
- And on the paid artifacts the **dollar figures do not reconcile** -- three rounding
  helpers (half-up vs half-even) plus the $13.50 benchmark rendered as $14 in the PDF
  show the same item as **$40 and $41 in one email line**, and a ranked cost column
  ($190) that overshoots its own headline ($189) (P5-2, PRESENTATION.md).

A support leader who spot-checks their own data will catch all three, and the
customer-facing question text is not even stable across a re-upload (F8). **Do not
ship paid reports until the clustering (F1/F2) and the money-formatting (P5-2) are
fixed.** For the free snapshot / lead-gen, the risk is lower but the same numbers are shown.

## The three findings that matter most

1. **F1 -- Fragmentation undercounts the #1 issue by ~58%.** 12 tickets that are all
   "can't log in" -> reported as 1 cluster of 5 + 7 discarded as "non-repeat"
   ($810 vs ~$1,944 on the dateless x12 run-rate; the ~58% undercount is basis-independent).
   The token-set topic partition produces FINAL buckets and embeddings cannot re-merge
   across them (F6). *Critical, verified.*
2. **F2 -- Junk becomes the #1 recommended fix and doubles the tax.** 6 identical
   "Automatic reply: Out of Office" -> their own #1 billed cluster; annualized tax
   $1,782 vs $810 legit-only (x12 run-rate; the >2x ratio is basis-independent). No spam
   gate. *Critical, verified.*
3. **C1 -- A missing/stripped header row silently drops 20-100% of tickets**, reported
   as a healthy-looking smaller number with no error (the `>=2-cell` fallback-header
   rule consumes row 1). Non-US dates are also silently transposed to the wrong day
   (C3). *Critical, verified.*

## The single highest-leverage change

**Replace the token-set-partition clustering with one unified semantic pass over
compressed representatives** (REFACTORS R1). It fixes F1, F3, and F6 -- the root of
the central product failure -- and is the difference between the paid number being
trustworthy or not. Everything else (junk gate R4, header hardening R3, the
normalization-cache perf win R2) is important but secondary to getting the count right.

## Everything else, briefly

- **Performance:** the pipeline is ~linear (~1.83 ms/ticket unloaded), NOT O(n^2) --
  the quadratic token-set clusterer is gated off above 2,000 rows. But the submit path
  has **no row cap** and accepts ~200k tickets (50 MiB), which runs 6-24 min
  synchronously (P7). The cost is an uncached field-normalization storm (P1) -- a ~5-10x
  speedup is available by caching per-row keys (R2). Memory (~10x file->RAM) is not binding.
- **Well-defended (do not re-chase):** encodings/mojibake, quoting, delimiter sniffing,
  Zendesk custom fields, and large-file completion are all genuinely robust.
- **Verification gap:** the QA scorecard has zero runtime callers, does not reconcile
  totals or verify quotes verbatim, and never runs on the snapshot (Phase 0 / Slice 5).
- **Presentation + richness:** see PRESENTATION.md.

*Read-only audit. No product code was modified. Fixtures/experiments are in gitignored
`_audit_scratch/`. Deliverables: PIPELINE_MAP.md, FINDINGS.md, REFACTORS.md,
PRESENTATION.md, SUMMARY.md.*
