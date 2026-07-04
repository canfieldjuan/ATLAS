# REFACTORS -- Resolution Audit CSV pipeline (Phase 4)

Deliverable for Slice 4 / #1960. Refactors are recommendations only -- **no product
code is changed in this audit.** Each is justified by a specific finding from
FINDINGS.md (Slices 2-3); anything not traceable to a finding is out of scope.
Effort is a rough engineering estimate; "tests first" = the regression tests that
must exist and stay green before the refactor is trusted.

Ranked by leverage (impact / effort).

## R1 -- Unified semantic clustering pass (fixes F1, F3, F6; the central failure)

- **Finding:** F1/F3 -- the token-set topic partition produces FINAL buckets and the
  embedding booster can only rescue leftover singletons *within* a bucket, so
  same-intent tickets phrased differently are scattered and then dropped below the
  `<2` gate. F6 -- embeddings cannot fix this. This undercounts the #1 issue ~44-58%.
- **Proposed shape:** replace "token-set label seeds the topic partition -> lexical
  sub-cluster -> singleton-only embedding rescue" with **one semantic pass over
  compressed representatives**: embed a per-ticket gist, cluster by cosine over the
  whole set (or within coarse category only when a category column exists), and use
  the lexical/token-set stage ONLY to compress exact/near-duplicates into
  representatives *before* the semantic pass -- never to produce final buckets.
- **Blast radius:** large -- `ticket_faq_markdown.py` clustering core (`:710`, `:787`,
  `:1026`, `:1097`) + the `<2` exclusion (`:862`) + `_topic` (`:1194`). Changes the
  billed `ticket_count` for every report, so it must ship behind a comparison gate.
- **Tests first:** the F1/F3 fixtures become regression tests (12 same-intent -> 1
  cluster of ~12, not 5+7; 25 phrasings -> 1 cluster of ~25). Golden-file the
  `report_model` counts on a labeled corpus so the count change is reviewed, not silent.
- **Effort:** high (~3-5 days incl. the embedding wiring fix R-note below).
- **Note:** pairs with fixing F7 (the wrapper's `del kwargs` dropping `embedding_port`)
  so the semantic pass can actually receive a port.

## R2 -- Uncached field-normalization: memoize keys + precompile (fixes P1, P5, P7)

- **Finding:** P1 -- `_first_value`/`_field_value` re-normalize every key of every row
  on every lookup (30.8M `re.sub` calls at 10k, ~25% of runtime), across ~20 full
  passes; P5 -- no memoization anywhere. This is the whole linear-with-large-constant
  cost and the reason 10k tickets take ~18-73s.
- **Proposed shape:** compute a normalized-key map per row ONCE (dict of
  `_key(k) -> value`) at normalization time; have `_first_value`/`_field_value` do a
  dict lookup instead of a regex scan. Precompile `_key`'s pattern (module-level
  `re.compile`). Optionally collapse the ~15 aggregation passes
  (`support_ticket_input_package.py:404-422`) into one.
- **Blast radius:** medium -- `support_ticket_input_package.py:976/990`,
  `ticket_faq_markdown.py:3381`. Pure performance; output must be byte-identical.
- **Tests first:** a golden-output parity test (same `report_model` before/after) +
  the Slice-3 timing harness as a perf regression (~5-10x speedup expected on P1's share).
- **Effort:** medium (~1-2 days). Highest perf leverage.

## R3 -- Header-detection hardening (fixes C1, H4, C2)

- **Finding:** C1/H4 -- a missing/stripped header row silently consumes row 1 as a
  phantom header (`>=2 non-empty cells` fallback, `campaign_customer_data.py:744-746`),
  losing 20-100% of tickets with no error; C2 -- duplicate column names silently
  last-win.
- **Proposed shape:** require a header confidence signal -- only accept a no-hint
  fallback header when the row looks header-like (short, non-sentence cells), else
  surface a `csv_header_uncertain` warning to the caller and DO NOT silently drop.
  Warn on duplicate normalized column names instead of last-win.
- **Blast radius:** medium -- CSV engine header path; could change which uploads are
  accepted, so ship with the warning first (non-breaking) then tighten.
- **Tests first:** C1/H4/C2 fixtures (headerless/partial/dup) assert a warning + no
  silent loss.
- **Effort:** medium (~1-2 days).

## R4 -- Junk/auto-reply admission gate (fixes F2)

- **Finding:** F2 -- repeated auto-replies form their own #1 billed cluster and more
  than double the support-tax; the publishable gate is PII/question-shape only, no spam check.
- **Proposed shape:** a deterministic auto-reply/spam classifier (regex + heuristics:
  "automatic reply", "out of office", no question, near-identical bodies) that excludes
  a ticket from counting toward a billed cluster (or flags the cluster as
  "auto-generated, excluded from tax"). Keep it visible (report the excluded count),
  do not silently drop.
- **Blast radius:** medium -- a filter before `ticket_count` accrues
  (`ticket_faq_markdown.py:1430`) and before ranking.
- **Tests first:** F2 fixture (5 real + 6 auto-reply -> real ranks #1, junk excluded/flagged).
- **Effort:** medium (~1-2 days).

## R5 -- Large-upload async/chunk + row cap (fixes P7, P2)

- **Finding:** P7 -- the submit path has no row cap (accepts ~200k tickets in a 50 MiB
  blob), and at ~1.8-7 ms/ticket a large upload runs 6-24 min synchronously.
- **Proposed shape:** cap the synchronous path (e.g. reject/queue > N tickets) and
  route large uploads to an async job with progress; document the ceiling. Pairs with
  R2 to raise N.
- **Blast radius:** medium -- the submit endpoint (`api/control_surfaces.py:1814`) + a
  job path.
- **Tests first:** a size-limit test; the Slice-3 ceiling harness.
- **Effort:** medium (~2-3 days if a job runner exists, more if not).

## R6 -- Date locale + signature/quote hygiene (fixes C3, M6, M7)

- **Finding:** C3 -- non-US dates silently transposed; M6 -- no signature/quoted-chain
  stripping (pollutes clustering + leaks PII); M7 -- one unparseable date disables the
  whole dated window.
- **Proposed shape:** detect day-first vs month-first per column (or require ISO with a
  clear error); add a signature/quoted-reply stripper before clustering; make the dated
  window per-row-tolerant instead of all-or-nothing.
- **Blast radius:** small -- `support_ticket_dates.py`, a new stripper in the text path.
- **Tests first:** C3/M6/M7 fixtures.
- **Effort:** low-medium (~1-2 days total).

## Refactors considered and REJECTED

- **Tune the clustering thresholds** (Jaccard 1/3, overlap 0.6, singleton <2). Rejected:
  F10 showed the top-5 is stable under +/-10% threshold perturbation -- the failure is
  *architectural* (fragmentation from the topic partition), not a threshold value, so
  tuning cannot fix F1/F3.
- **Just turn embeddings on** (fix F7's `del kwargs`). Rejected as a standalone fix:
  F6 proved embeddings, even enabled, cannot merge cross-topic fragments -- the booster
  only rescues singletons within a bucket. Necessary as part of R1, useless alone.
- **Optimize the pairwise Jaccard sub-clusterer** (`_question_subclusters`). Rejected:
  P3 measured it LINEAR (b~0.92), not a hotspot -- optimizing it buys nothing.
- **De-duplicate the model-vs-prose metric computation for performance.** Rejected on
  perf grounds: P6 measured it as microseconds (operates on the small item list). It
  remains a correctness/drift concern for Slice 5, not a refactor for speed.

*Recommendations only -- no product code was modified.*
