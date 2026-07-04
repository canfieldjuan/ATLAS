# FINDINGS -- Resolution Audit CSV pipeline (Slices 2-3)

Severity-rated defects from the empirical audit. Slice 2 (#1958) covers Phases 1-2
(CSV parsing + clustering); Slice 3 (#1959) will append performance findings.

Method: adversarial fixtures built in gitignored `_audit_scratch/` and run through
the REAL pipeline functions (`_load_csv_dict_rows_result`,
`build_support_ticket_input_package`, `FAQDeflectionReportService.generate` ->
`build_ticket_faq_markdown`). Every number came from a real run against
`origin/main`. Findings marked **[verified-by-reviewer]** were independently
re-run by the reviewer, not just relayed from an agent. `file:line` anchors are
under `extracted_content_pipeline/` unless prefixed.

`critical` = silent data corruption, wrong numbers in customer-facing output, or
nondeterminism that breaks the scorecard.

---

## CRITICAL

### C1 -- Partial/absent header silently drops tickets and strips subjects, slipping the zero-row guard [verified-by-reviewer]
A headerless (or header-stripped) CSV whose first data row has >=2 non-empty cells
is silently consumed as a phantom header. On a 5-row headerless fixture the parser
returns **4 rows** (20% silent loss); a fully headerless 5-row file returns 0.
Because a row-1 cell often matches a `_TEXT_KEYS` alias, the survivors still yield
text and pass the endpoint's `included_row_count <= 0` guard
(`api/control_surfaces.py:1547`) with **zero warnings** and an undercounted
`source_row_count`. Survivors also lose their subject (it became a column name).
- Evidence: fallback-header acceptance `campaign_customer_data.py:744-746`;
  `_CsvDelimiterCandidate.valid:191-198` (no hint required); dict assembly `:534-542`.
- Repro: `_audit_scratch/fixtures/edge_E1_partial_headerless.csv` ->
  `load_csv_source_rows_result_from_file(...)` returns 4 of 5 rows.
- Impact: 20-100% silent ticket loss on a plausible export mistake, reported as a
  healthy-looking smaller number. **Highest-leverage root of the whole parser.**

### C2 -- Duplicate header column silently discards the earlier column (last-wins)
Two columns with the same normalized name -> the second overwrites the first with
no warning. Realistic for Zendesk exports repeating a column (tags/comments/custom
fields).
- Evidence: `cleaned[cleaned_key] = cleaned_value` overwrite at
  `campaign_customer_data.py:542` (loop `:535-542`).
- Repro: `_audit_scratch/fixtures/st_2f_dup_headers.csv` -> `message = "SECOND message"`.

### C3 -- Non-US dates silently transposed to the wrong day [verified-by-reviewer]
`parse_support_ticket_source_date("02-01-2026")` returns **2026-02-01** (Feb 1). A
UK/EU/AU export of "1 Feb" (day-first) is silently read as Feb 1; day>12
(`13-01-2026`) returns `None`. Drives the dated window / recency basis of the report.
- Evidence: `_US_DATE_FORMATS` (`%m/%d/%Y`, `%m-%d-%Y`, ...) `support_ticket_dates.py:8-13`,
  applied `:39-43`. No locale/day-first detection.
- Repro: `parse_support_ticket_source_date('02-01-2026') -> 2026-02-01`.

### F1 -- Same-intent tickets fragment across topic buckets, fall below the singleton gate, and are dropped from the billed repeat surface [verified-by-reviewer]
12 tickets, all "cannot authenticate / access account", each phrased with low
lexical overlap. Measured: **1 cluster with `ticket_count=5`; the other 7 each land
in their own single-row topic bucket and are excluded as `non_repeat` (7).**
`report_model`: `repeat_ticket_count=5`, `annualized_run_rate_support_cost=$810`
-- this fixture is dateless, so the pipeline uses its **x12 run-rate fallback**
(`faq_deflection_report.py:3232`: `5 x 12 x $13.50`); a *dated* upload annualizes by
`x365/window_days` (`:3228`) instead. Correct clustering (12 repeats) = ~$1,944 on the same
x12 run-rate. **58% of the repeat surface silently dropped** (basis-independent); 7 genuine
repeats mislabeled "asked only once." The dollar figure is a run-rate "if this pace holds,"
not a confident annual loss (see H-x12).
- Evidence: topic partition seeds from the token-set label
  (`ticket_faq_markdown.py:770,787-788`; label `support_ticket_clustering.py:462`);
  the `<2 distinct-source` exclusion `ticket_faq_markdown.py:862` discards the scattered singletons.
- Repro: `_audit_scratch/fixtures/cl_attack1b_frag_exclude.json` ->
  `python _audit_scratch/run_fixture.py fixtures/cl_attack1b_frag_exclude.json`.
- Impact: the product's central claim (repeat questions cluster correctly) is false
  in the fragmentation direction; the #1 issue is undercounted AND partly deleted,
  understating the paid deflection number.

### F2 -- Repeated junk/auto-reply tickets form their own billed cluster, become the #1 fix, and more than double the headline support tax [verified-by-reviewer]
5 real "reset my password" + 6 identical "Automatic reply: Out of Office". Measured:
**the 6 auto-replies become their own cluster, `ticket_count=6`, ranked #1**
(representative "What should I do about automatic monday office out?"); the real
question ranks #2. `repeat_ticket_count=11`, `estimated_support_cost=$148.50`,
`annualized=$1,782` (x12 dateless run-rate, `:3232`); the junk line item is `$81` and is the
#1 priority-fix / drafted-resolution / JIRA item. Legit-only would be $67.50 / $810 run-rate --
**junk inflates the tax by ~$972 (>2x) and owns the top recommendation** (the >2x ratio is
basis-independent; annualized dollars are a run-rate, not a confident annual figure).
- Evidence: publishable gate is PII/question-shape only, not spam
  (`_publishable_customer_question_text:2659`); every group member counts toward
  `ticket_count` (`:1430`).
- Repro: `_audit_scratch/fixtures/cl_attack5b_junk_repeat.json` -> run_fixture + run_cost.

---

## HIGH

### H4 -- Fully headerless file -> 100% loss, reported as "no usable wording" with a wrong count
5 rows -> 0 delivered, all `ticket_row_missing_text`; parser layer fully silent
(`engine_ok`, `source_row_count=4`). The deflection endpoint turns it into a 400
`deflection_submit_no_usable_rows` (misleading reason -- wording WAS present), but
generic ingestion (Path B) gets a silent empty result.
- Repro: `hl_H1_headerless_5rows.csv` (0 delivered) vs `hl_H1b_withheader_5rows.csv` (5).

### H5 -- One over-wide row hard-rejects the entire file (one unquoted comma kills the upload)
A single row with more cells than the header raises `csv_inconsistent_columns` for
the whole file; asymmetric with short rows (silently accepted/padded). One ticket
body with an unquoted comma the exporter failed to quote = zero report.
- Evidence: `campaign_customer_data.py:527-531`. Repro: `st_2e_long_row.csv`.

### F3 -- Highest-volume question fragmented and undercounted even when nothing is excluded
25 auth tickets, 10 phrasings -> one intent split into 6 FAQ line items
(counts 14/3/2/2/2/2, `non_repeat=0`). Top-question undercount 14 vs 25 = **44%**;
the customer sees six "repeat questions" instead of one 25-ticket opportunity.
- Repro: `cl_attack1_fragmentation.json`.

### F4 -- Surface-similar distinct intents over-merge into one vague cluster
10 "cancel my subscription" + 10 "cancel my order" -> 1 cluster, `ticket_count=20`,
representative "What should I do about cancel?" -- one generic answer for two
different workflows. Cause: anchor merge on the shared "cancel" token
(`support_ticket_clustering.py:580-582,587`). Repro: `cl_attack2_overmerge.json`.

### F5 -- A junk row first in row order hijacks the displayed representative and inflates the cluster
Representative is first-match-wins over arbitrary CSV row order (`_question:2372-2375`),
so a junk phrasing that appears first becomes the customer-facing question for a
cluster of real tickets. Repro: `cl_attack5c_junk_rep.json`.

### F6 -- The embedding "semantic rescue" is inert for the core failure mode
Real `all-MiniLM-L6-v2`: OFF == ON, byte-identical, on both fragmentation fixtures.
The booster only runs inside a single topic bucket with >=2 rows
(`ticket_faq_markdown.py:1106,1115-1120`), so it can never merge same-intent tickets
the token-set partition already scattered into separate single-row buckets. A
perfect semantic model cannot fix F1/F3. Repro: `run_embedding.py`.

### F7 -- The deflection wrapper discards any per-call embedding_port [verified-by-reviewer]
`FAQDeflectionReportService.generate(**kwargs)` does `del kwargs`
(`faq_deflection_report.py:1697`) and never forwards a per-call `embedding_port`, so
embeddings cannot be enabled per request. They are on ONLY if the wrapped
faq-markdown service was constructed with one via the host factory
(`atlas_brain/_content_ops_services.py:355` `faq_embedding_port_factory` ->
`TicketFAQMarkdownConfig(embedding_port=...)`, `:293`) -- not "structurally off
regardless of config". And per F6, even when enabled they do not fix the
fragmentation failure.
- Repro: passing `embedding_port` to `generate()` == not passing it (discarded);
  F6's WITH-embeddings control (`run_embedding.py`) changed nothing.

---

## MEDIUM

- **M6 -- No signature / quoted-chain / PII stripping.** HTML is stripped
  (`support_ticket_clustering.py:311-330`) but agent signatures, quoted email chains,
  and inline emails/phones survive into delivered text and dominate lexical gist
  tokens (pollutes clustering + leaks PII). Repro: `ct_3c_signature.csv`.
- **M7 -- One unparseable date disables the dated window for the whole report.**
  `YYYY/MM/DD`, natural-language, and epoch dates parse to `None`; a single such
  included row flips the report to the undated source period. Repro: `ct_3e_dates.csv`.
- **M8 -- Embedded NUL bytes below the 10% ratio pass into delivered text**
  (`_CSV_NUL_REDECODE_RATIO=0.10`). Repro: `enc_1k_embedded_nul.csv`.
- **M9 -- Macro name in the status column -> bucketed "other", undercounts resolved
  rate** (safe direction; rows still included). Repro: `zd_4c_ii_macro_status.csv`.
- **F8 -- Order-dependent presentation.** Same input+order twice -> byte-identical;
  row-shuffle -> markdown differs every time (representative question flips, e.g.
  "login" <-> "access your account", and F5's junk/real swap). Aggregate
  `ticket_count`s stay stable, so scorecard numbers are order-stable but the
  customer-facing questions/quotes are not -- contradicting "deterministic
  positioning" at the presentation layer. Cause: `:2372`,
  `support_ticket_clustering.py:556-560`. Repro: `run_determinism.py`.
- **F9 -- 22% of a realistic mixed inbox excluded** from the repeat/billed surface
  (45 tickets -> `non_repeat=10`). Defensible here (genuine one-offs) but it is the
  same `<2` gate that drops real repeats under F1. Repro: `cl_attack34_mixed.json`.
- **H-x12 -- Raw `annualized_run_rate_support_cost` is a bare x12 run-rate with no span basis [verified-by-reviewer]**
  For a dateless upload the pipeline sets `annualized_run_rate_support_cost = repeat_ticket_count x 12 x $13.50`
  (`faq_deflection_report.py:3232`), which equals the true dated annualization (`repeat x 365/days x $13.50`,
  `:3228`) ONLY when the actual span is ~30.4 days. For a span of T days the fallback is `12T/365` of the dated
  figure: a **year** of dateless tickets reads ~**12x too high** (T=365 -> 12.0x), ~30 days is right, a **week**
  reads ~**0.23x (~4.3x too low)** -- two-directional, not a flat 12x (correcting an earlier draft of this
  finding). **Scope:** the rendered markdown (`:4076-4083`) and PDF (`deflection_pdf_renderer.py:305-312`)
  ALREADY hedge this ("does not infer a monthly or annual reporting period ... If this uploaded batch is monthly
  pace ... Estimate only"); the gap is the **raw `report_model` / API field** surfaced by any downstream consumer
  (dashboard, CRM push, snapshot) WITHOUT that prose. Fix: carry the span basis on the field, or require consumers
  to show the "if monthly pace" caveat. Surfaced by the GPT-5.5 Pro + Codex review (#2002 thread); tracked in #2000 / #1993.

## LOW / informational

- **L10 -- Blank/whitespace rows dropped and NOT counted in `source_row_count`**
  (`:524-525`) -> `source_row_count` is non-reconcilable against the file's line count.
- **L11 -- Latin-1/cp1252 fallback is silent** when it decodes plausibly (no
  "non-UTF-8 used" signal).
- **L12 -- Inconsistent width handling:** over-wide blank row silently dropped, but
  over-wide content row rejects the whole file (H5).
- **L13 -- Short rows truncate trailing fields only** (no field-shift corruption) --
  confirms the map. `:536`.
- **F10 -- Top-5 ranking is NOT fragile to +/-10% threshold changes** (a positive
  result), but overlap `0.6` (`support_ticket_clustering.py:582`), anchor `>=2`
  (`:580`), and singleton `<2` (`ticket_faq_markdown.py:862`) are inline magic
  literals, not configurable. Robust to threshold *values*, fragile to *phrasing*.
- **F11 -- Representatives are frequently synthesized templates**
  ("What should I do about <topic>?"), not verbatim customer wording, unless a
  question-shaped row happens to sit first (which reintroduces F5).

## Well-defended (negative results, recorded so later slices do not re-chase)

- Encodings all decode: UTF-8 BOM, UTF-16 LE/BE (incl. BOM-less via NUL inference),
  latin-1, cp1252, emoji, RTL. **Mojibake is actively avoided** (UTF-8+replace beats
  whole-file latin-1 fallback; `csv_replacement_characters` warning).
- Quoted commas / embedded newlines / doubled quotes / reordered columns /
  semicolon+tab delimiters: correct (stdlib `csv`).
- Zendesk custom fields, quoted multi-value tags, status-flag-only, free-text
  resolution, both-missing: correct; status never gates inclusion.
- Large files complete, no OOM: 500k rows/47MB = 8.34s / 243MB peak (~5x file->RAM
  amplification -> feeds Slice 3).
- Clustering top-5 stable under +/-10% threshold perturbation (F10).

## Slice 3 -- Performance (Phase 3)

Measured on the real chain (`build_support_ticket_input_package` ->
`FAQDeflectionReportService.generate` -> `build_ticket_faq_markdown`) with realistic
free-text fixtures (~256 B/row, 15% HTML, unique source_ids). Scripts in
`_audit_scratch/perf/`. Timing note: the reviewer's confirmed **unloaded** numbers
are ~1.83 ms/ticket (10k = 18.3s, reproduced twice); under heavy concurrent load the
same chain was observed at ~7 ms/ticket (10k = 73s), so real-world time under
contention can be 3-4x the unloaded figure. Scaling is **linear** either way --
growth exponent **b~1.0**, per-ticket cost flat: total 4.2s@2k, 9.2s@5k, 18.3s@10k
unloaded (doubling n -> ~2.0x time).

### P1 -- Uncached field-normalization storm (~30.8M regex calls) dominates the pipeline [verified-by-reviewer]
cProfile @10k: `re.Pattern.sub` = **30.8M calls / 11.3s tottime (25%)**;
`_field_value` (`ticket_faq_markdown.py:3381`) = **15.2s cumulative (33%)**;
`_first_value` (`support_ticket_input_package.py:976`) 390k calls / 10.3s; `_key`
(`:990`) 8.6M calls / 9.2s. Root cause: `_first_value`/`_field_value` re-normalize
**every key of the row on every field lookup** (linear `row.items()` scan calling
regex `_key`/`_compact_key` per field, repeated for dozens of alias groups) with **no
per-row key cache**, across ~20 full O(n) passes end-to-end
(`support_ticket_input_package.py:404-422` is ~15 aggregation passes alone).
Compounding: `_key` passes a string-literal pattern to `re.sub`, forcing ~9.4M
`re._compile` cache lookups. This -- not clustering -- sets the ceiling.
- Severity: HIGH. Fix: cache per-row normalized keys once + precompile the pattern.
  Repro: `_audit_scratch/perf/profile_cprofile.py`, `cprofile_10k.txt`.

### P7 -- The submit path has NO row cap; a 50 MiB blob (~200k tickets) runs synchronously for many minutes [verified-by-reviewer]
`_deflection_submit_max_rows` (`api/control_surfaces.py:1814`) returns the full
`raw_row_count` -- the only bound is `_MAX_DEFLECTION_SUBMIT_BLOB_BYTES = 50 MiB`
(`:167`) ~= **~200k tickets** at ~257 B/row. At the measured linear rate that is
~6 min unloaded / ~24 min under load, ~1.3 GB, in a single request. The practical
30s-request ceiling is only **~4,000 tickets**. A realistic large customer upload
must be chunked or run as an async job; today it will time out or hang.
- Severity: HIGH. Repro: `harness.py` ladder (10k=18.3s, 50k measured ~120-356s).

### P2 -- Token-set clustering is near-quadratic (b~1.88) but hard-gated at 2,000 rows; the skip degrades clustering at scale [verified-by-reviewer]
`_matching_token_bucket` (`support_ticket_clustering.py:567`) compares each row
against every prior token_set -- fit b=1.88 ungated (6.0s@4k, 23.6s@8k, matching the
code's own `:243-249` "~40 min at 35k" comment). Gate `MAX_TOKEN_SET_CLUSTER_ROWS=2000`
(`:249`) caps it ~1.6s then **skips** (rows left uncategorized). The skip is NOT silent
-- it emits `cluster_preview_skipped_large_upload` + `cluster_preview_token_set_row_count`
(`support_ticket_input_package.py:373-385,507-510`), forwarded in the submit response.
But above 2,000 rows the token-set labels are still absent, degrading system-2's topic
partition (observable: `faq_items` jumps 4->62 between 2k and 5k). Feeds Slice-2 F1/F3.
- Severity: MEDIUM. Repro: `scaling_clustering.py`.

### P4 -- ~10x whole-pipeline file->RAM amplification; ~3 full copies co-resident [verified-by-reviewer]
Deep-size: raw rows 2.2x, `source_material` 2.9x, campaign opportunities 4.6x the
serialized input -- all co-resident during `build_ticket_faq_markdown` -> ~9.8x floor
(23.9 MiB @10k, 119.6 MiB @50k py-heap). RSS peak by size: **35 MiB @1k, 46 @2k, 62 @5k,
97 @10k, 357 @50k** (linear in n). `source_row_to_campaign_opportunity` builds a full new
richer copy (1,189 B/row). Memory is NOT the binding constraint -- time crosses 30s ~40x sooner.
- Severity: MEDIUM. Repro: `memory_copies.py`, `harness.py`.

### P3 -- (positive, corrects a Phase-0 concern) the sub-clusterer is LINEAR (b~0.92), not O(n^2) [verified-by-reviewer]
`_question_subclusters` (`ticket_faq_markdown.py:1026`) uses prefix-filtered candidate
nomination + length pre-filter + union-find, not naive pairwise: 0.61s@10k, 1.14s@20k,
b=0.92. Negligible vs the report stage. PIPELINE_MAP 5b framed this as the
pairwise-Jaccard O(n^2) risk; empirically it is not a hotspot.

### P5 -- No memoization anywhere; reruns recompute everything
Zero `lru_cache`/`cached_property` in the 5 core modules -- no cache between snapshot
and full-report on one file, or between reruns; a re-submit repeats 100% of P1's work.
Embeddings off by default (and when on, embed compressed gists, not raw tickets -- no
raw-embedding cost). Severity: MEDIUM (pairs with P1).

### P6 -- Metric double-computation is NEGLIGIBLE for perf (drift concern only)
The only double-computation is the prose `_support_tax_section`
(`faq_deflection_report.py:4034`) re-deriving counts the model `_support_tax_data:3208`
already produced -- and both operate on the small final FAQ `items` list (~8-62), not
raw tickets -> microseconds. The **snapshot does NOT recompute**: `build_deflection_snapshot`
takes the report-model projection path (`:2641-2649`), copying support-tax fields from
the stored model. Clustering runs exactly once per system per file. Phase 0's
"computed twice" stands as a correctness/drift concern (Slice 5), not a perf sink.
- Severity: LOW (perf).

## Single highest-leverage fixes

1. **Clustering fragmentation (F1/F3/F6):** the token-set topic partition producing
   FINAL buckets, with embeddings unable to re-merge across buckets, is the root of
   the central product failure. Fix is architectural (a unified semantic pass over
   compressed representatives), not a threshold tweak -- Slice 4 REFACTORS.
2. **Header detection (C1/H4):** the `>=2 non-empty cells` fallback-header rule
   converts a missing/stripped header into silent uncounted ticket loss.
3. **Junk/spam admission (F2):** no spam/auto-reply gate before a ticket counts
   toward a billed cluster.

*No product code was modified. Fixtures/runners are in gitignored `_audit_scratch/`.*
