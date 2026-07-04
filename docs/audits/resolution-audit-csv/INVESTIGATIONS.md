# INVESTIGATIONS -- per-finding remediation deep-dive

Follow-up to the Resolution Audit CSV audit (arc #1956) and remediation issue #1993.
For EACH finding: root cause traced to the true upstream origin, the exact fix (code
sketch), boundary-probed edge cases, blast radius, tests-first, and effort. Prototypes
for the load-bearing fixes were built in gitignored `_audit_scratch/` and the key
claims were re-run by the reviewer (`[verified-by-reviewer]`). No product code modified.
`file:line` under `extracted_content_pipeline/` unless prefixed.

## Shared roots (fix once, not N times)

| Group | Findings | Single root | Fix |
|---|---|---|---|
| Lexical clustering (no semantic stage) | **F1, F3, F4** (+ F6/F9 downstream) | token-set label promoted to a HARD `topic` partition key (`ticket_faq_markdown.py:788`); `_LOW_SIGNAL_ANCHOR_TOKENS` (`support_ticket_clustering.py:223-242`) contains the auth-intent words, so same-intent tickets can't even anchor-merge | R1 |
| Header detection has no confidence | **C1 = H4** (same bug) **+ C2** | `>=2-cell` fallback-header rule (`campaign_customer_data.py:744-746` + stream twin `:1105`) | R3 |
| Date locale | **C3 -> M7** (chained) | US-only formats + all-or-nothing window gate | R6 |
| Text chokepoint | **M6 + M8** | `support_ticket_plain_text:311-330` scrubs HTML only | R6/PII |
| Uncached field lookup | **P1 + P5** | `_first_value`/`_field_value` re-normalize every key on every lookup; the correct pattern (`_SourceFieldLookup:458`) exists but the hot modules didn't adopt it | R2 |
| Money rendering | **P5-2** (4 defects) | 3 helpers, 2 rounding modes, benchmark re-derived, money baked into model strings | new |

---

## Clustering (F1, F3, F4, F5, F6, F7, F8, F9, F11 + R1)

**Production grounding [verified]:** the embedding booster is OFF by default
(`atlas_brain/config.py:4852` `content_ops_faq_embedding_booster_enabled=False` "until
the mxbai thresholds are live-baselined") -- **the report ships purely lexical today**,
and the codebase itself admits the thresholds are uncalibrated (R1's calibration is a
known-pending prerequisite, not new work).

- **F1 [critical]** -- root is NOT the `<2` gate (`ticket_faq_markdown.py:862`); it is the
  lexical partition: 12 same-intent tickets get 5 different token-set labels
  (`support_ticket_clustering.py:461`, merge rule `:580-596`) -> 5 hard `topic` partitions
  (`:788`) -> each a singleton -> gate excludes 10. Fixing only the gate turns F1 into F3.
  Fix = R1. Blast radius: raises billed `repeat_ticket_count` for every fragmented report;
  ship behind a golden/shadow gate. Effort: high (R1).
- **F3 [high]** -- same root, second lexical stage: even within one topic bucket
  `_question_subclusters` (`:1026`) splits on exact-Jaccard >= 1/3 of gist tokens (`:1077`).
  Note: F3 is **cost-neutral** (fragments stay >=2 so all count) -- harm is presentation
  (10 items not 1) + top-item undercount. Useful for the comparison gate: F3 inputs must
  NOT move the dollar total.
- **F4 [high]** -- `_shared_anchor` (`:658-666`) merges on the single shared token `cancel`
  with no distinguishing tokens. **This constrains R1 more than any finding** (below).
- **F5 [high]** -- `_question:2372` is first-match-wins over row order; the gate
  (`_publishable_customer_question_text:2659`) checks shape+PII, not junk. Fix: rank
  candidates by (customer-wording, on-topic centroid overlap, length, not-auto-reply),
  argmax deterministically; pair with R4. Effort low, shares with F8.
- **F6 [high, verified]** -- spy injection: `config.embedding_port=spy` -> spy called 3x
  but output byte-identical to no-port. The booster (`:1106`) only unions leftover
  singletons *within one topic bucket*; it never compares across the 5 scattered buckets.
  A perfect model cannot fix F1/F3 in this position -- the fix is architectural placement (R1).
- **F7 [high, verified]** -- `generate(embedding_port=spy)` -> spy.calls=0. Wrapper
  `del kwargs` at `faq_deflection_report.py:1697` and never forwards it; the inner service
  already accepts it (`ticket_faq_markdown.py:588`). Fix: forward `embedding_port` (sketch
  in issue). Do it as the FIRST commit of R1 (else it changes nothing -- see F6).
- **F8 [medium, verified]** -- representative + markdown change with row order (counts
  stable). Root: `:2372` first-match + anchor-promotion order-dependence
  (`support_ticket_clustering.py:556`). Fix: deterministic ranking (F5), tie-break on
  content hash not row index.
- **F9 [medium]** -- 22% excluded on a realistic mixed inbox is the DEFENSIBLE side of the
  same `<2` gate (genuine one-offs). It is the control proving the gate is right once the
  upstream partition is right -- which is exactly why R1's merge must run BEFORE the gate.
- **F11 [low]** -- representatives are synthesized ("What should I do about {label}?",
  `:2409`) whenever the strict gate rejects all rows. Couples to F5/F8: rank real wording
  first; label synthesized ones.

### R1 -- unified semantic clustering pass [design corrected + verified]

The audit's R1 ("embed a gist, cluster by cosine over the whole set") is **under-specified**.
Measured separability of cl_attack2 (cancel-subscription vs cancel-order):
`intra-intent cosine min 0.506` vs `inter-intent max 0.672` -> **separation gap NEGATIVE**
(reviewer's independent run: **-0.470**, min 0.236 vs max 0.706). The negative gap proves the
*pairwise* cosine bands overlap -- no single global pairwise threshold classifies every pair
correctly. That alone does NOT prove single-linkage impossibility (single-linkage needs only a
connected chain, not every pair above threshold). **Closed with the MST connectivity bottleneck:**
the two intents JOIN at `merge_distance = 1 - max_inter = 0.295`, but each intent only becomes
internally connected at `connect_distance = 0.535` (its single-linkage MST bottleneck). Since
`connect_distance (0.535) > merge_distance (0.295)`, single-linkage merges the two intents BEFORE
either is internally whole -> **no threshold yields the 2 pure intents** (sweep confirms: no
pure-2 band). So a naive single-linkage semantic pass genuinely reproduces F4 -- now *proven* via
the bottleneck, not inferred from the gap. Property of the data; mxbai-large shifts thresholds but
does not erase the overlap.

**Concrete algorithm:**
1. **Stage A -- demote the lexical stage to duplicate-compression only.** Keep the token-set
   clusterer / `_question_subclusters` but use them SOLELY to fold exact/near-duplicates
   into representatives; stop feeding the label into `:788` as a hard partition key.
2. **Stage B -- one guarded semantic merge over representatives.** Embed each rep's gist
   with the pinned port (mxbai prod / MiniLM proven). **Complete or average linkage -- NEVER
   single** (proven to chain the F4 overlap). Scope to category when a `pain_category`
   column exists. Use the existing mutual-kNN + margin gate (`:1152-1163`, floor 0.78 /
   margin 0.05) as the merge-admission rule, not a raw threshold. Threshold **calibrated**
   on a labeled corpus. Deterministic (fixed model+revision, sorted merge order, CPU-pinned).
3. **Stage C -- `<2` gate on merged clusters** (unchanged) -- now only genuine one-offs fall below.
4. **F7 wired first** so Stage B gets a port.

**Prototype results [verified]:** cl_attack1 (25 same-intent) -> 1 cluster of 25 (vs current
10 items); cl_attack2 -> single-linkage yields no pure-2 partition at any T (MST bottleneck
0.535 > merge 0.295), complete-linkage d=0.50 separates to 2 pure [10,10] but doesn't collapse
*diverse* same-intent -- **the honest trade-off R1 must resolve with calibration, not a
hard-coded constant.** Modules:
`ticket_faq_markdown.py:788,850-868` (Stage A/B), `faq_deflection_report.py:1697` (F7),
`support_ticket_clustering.py` (repurpose to reps; synced file -- edit `atlas_brain/` source
+ `sync_extracted.sh`). A/B behind a flag + golden support-tax on a labeled corpus.
Effort: high (~4-6 days incl. F7 + calibration harness + comparison gate).

---

## Parsing (C1, C2, C3, H4, H5, M6, M7, M8, M9)

- **C1 = H4 [critical]** -- one bug. `_csv_header_index_and_hint:744-746` accepts ANY first
  row with >=2 non-empty cells as the header, no confidence signal (duplicated in the stream
  path `:1105`). C1 = phantom header (subjects become column names); H4 = its 100%-loss tail
  when the phantom names alias no text key. Fix (R3): a `_row_is_header_shaped` heuristic ->
  a non-fatal `csv_header_uncertain` warning (never silent accept); a header-specific endpoint
  reason for H4. **Edge case that decides the design:** a real lowercase single-word header
  (`col1,col2`) is INDISTINGUISHABLE from a headerless data row -> the fix must WARN, not
  reject (ship warning-first, non-breaking; tighten later behind telemetry). ~2 days incl. C2.
- **C2 [critical]** -- dict last-wins overwrite (`campaign_customer_data.py:542`). Fix: detect
  duplicate normalized header names -> warn + keep-first (or suffix). ~0.5 day, bundle with C1.
- **C3 -> M7 [critical, chained]** -- US-only date formats, no locale (`support_ticket_dates.py:8-13`).
  Fix (R6): a per-value `dayfirst` param, decided ONCE per upload by column inference
  (any first-field>12 -> day-first; contradictory -> warn, don't guess). Then M7's window gate
  (`support_ticket_input_package.py:390`, `missing_count==0` all-or-nothing) -> make per-row
  tolerant (fraction threshold, derive window from parseable dates). Do together; C3 unblocks M7.
  Updates the `:1043` golden. ~1.5-2 days.
- **H5 [high]** -- over-wide row raises before the tolerant consistency model runs
  (`:527-531`). Fix: merge spillover into the last column OR drop-and-warn per row; let the
  0.90-consistency check be the single arbiter. Boundary: all-rows-over-wide must still fail.
  ~1 day, isolated from the collapsed-delimiter path.
- **M6 + M8 [medium]** -- text chokepoint (`support_ticket_plain_text:311-330`) scrubs HTML
  only. Fix: `_strip_reply_cruft` (signatures/quoted chains) + PII redaction + C0-control-char
  scrub (`\x00` survives because `\s` doesn't match it). Ties to the PII architecture -- do the
  PII scrub here so it covers the free teaser. **Coordinate with the PII owner; no teaser
  product-shape change without sign-off.** ~1.5-2 days.
- **M9 [medium, low-priority]** -- exact-match status vocab (`:817-829`); a macro-prefixed
  status -> "other" (safe direction). Fix: split on separators + leading-token match. This is
  a CLASSIFIER: mandatory negative probes (`closedeal` must NOT be "closed", `unresolved` must
  NOT be "resolved", `reopened` first). ~0.5 day.

---

## Performance (P1, P2, P4, P5, P7)

- **P1 + P5 -> R2 [high, verified]** -- `_first_value`/`_field_value` re-normalize every row
  key on every lookup (`:976`, `:3381`), `_key` (`:990`) a `re.sub` per call, across ~20 full
  passes. **The correct pattern already exists** (`_SourceFieldLookup:458`) and the two hot
  modules didn't adopt it. Fix: per-row key-index built once + precompile/memoize `_key`.
  **Reviewer-run parity + speedup:** byte-identical parity; **6.1x from a one-line
  lru+precompile, 55x from the per-row key-index.** Load-bearing subtlety: `_first_value` is
  first-non-empty-wins (skips empties, keeps `0`/`False`); a naive `{key:val}` cache breaks
  parity -- the index must preserve empty-skip (proven). ~hours (one-liner) to ~1-2 days (index).
- **P7 [high, verified]** -- root: `_MAX_DEFLECTION_SUBMIT_ROWS = _MAX_DEFLECTION_SUBMIT_BLOB_BYTES`
  (`api/control_surfaces.py:168`) -- the row "cap" is the 50 MiB BYTE value (52.4M), so there
  is effectively no row cap (~200k tickets). Worse: `build_support_ticket_input_package` runs
  INLINE on the event loop (`:1598`, not `to_thread`), blocking the worker; then a GLOBAL
  concurrency gate serializes report generation across tenants. Fix (R5): real row cap ->
  413 (like the inspection path which already caps at 10k); route large uploads to the
  **existing** async scheduler + deflection delivery task; at minimum wrap the inline build in
  `to_thread`. ~2-3 days (runner exists). Pairs with R2 (which raises the safe sync N ~50x).
- **P2 [medium]** -- token-set clustering O(n^2) (b~1.88), gated at 2000 (`:249`); the skip is
  NOT silent (emits `cluster_preview_skipped_large_upload`) but drops the labels above 2000,
  degrading system-2. Fold into R1 (don't fix the quadratic twice; R1 replaces the partition).
- **P4 [medium]** -- ~10x file->RAM, ~3 co-resident copies (`source_row_to_campaign_opportunity`
  builds a full richer dict per row, `:830`). Memory is NOT binding (time crosses 30s ~40x
  sooner). Stream after P7 caps rows; low priority.

---

## Presentation + richness (P5-1..P5-7 + Phase 6)

- **P5-2 [critical, verified]** -- FOUR defects: (1) rounding-mode split (email half-even
  `content_ops_deflection_delivery.py:1637` vs markdown/PDF half-up `:4892`/`deflection_pdf_renderer.py:941`;
  diverges when `ticket_count % 4 == 3`); (2) **sum-of-rounded-rows != rounded-sum** (orthogonal
  to mode -- per-row `_ranked_opportunity_section:4167` rounds each row, headline `:4051` rounds
  the sum -> column $190 vs headline $189, persists even if all helpers agree); (3) benchmark
  re-derived `_model_money(13.50)=$14` on the PDF; (4) money baked into `product_gap_summary`
  string (`:3829`) shown next to the email's own price -> $40 and $41 in one line.
  **Fix [verified]:** one shared helper in the owned package (both renderers already import it),
  **render cents** -- proven for the base support-cost table: rows `['$67.50','$81.00','$40.50']`
  sum to headline `$189.00` exactly (`n x $13.50` is always cent-exact), benchmark renders true
  `$13.50`, all surfaces identical. Cents beats "round the total" and "sum the rounded rows" (the
  latter makes the headline wrong, amplified ~16x by the annualized). **Scope caveat:** cents fix
  the base `n x $13.50` display, NOT universal row-sum reconciliation -- fractional-cent rows (e.g.
  annualized `window=4 -> $1,231.875/row`; two rows render `$1,231.88` summing `$2,463.76` vs true
  `$2,463.75`) still need the raw-total guard below. Kill the baked-money anti-pattern. ~1.5-2 days.
- **P5-7 guard [verified]** -- add a totals-reconciliation guard at `build_deflection_report_artifact:1721`
  (the single chokepoint all surfaces read). **Second-side trap (verified):** the headline uses a
  repeat-only basis (tc>=2) while per-row costs include tc==1, so a naive `sum(all rows)==headline`
  guard FALSE-POSITIVES on any report with a single-ticket question (202.50 vs 189.00). The guard
  must call the **same billed-repeat predicate** used to compute `repeat_ticket_count` (not a
  re-hardcoded `tc>=2` -- that would be a second source of drift); it then correctly catches real
  model drift. Cents rendering makes the base-cost DISPLAY reconcile by construction; the model
  guard catches drift. Also wire the
  CI-only scorecard (zero runtime callers) into runtime.
- **P5-1 [trivial]** -- SEO section priority 20 above ranked (30)/fix-queue (35); it disclaims its
  own value (`:4123`). Fix: `priority=45` (data-only). ~1h.
- **P5-3 [low]** -- `_ticket_text:647` dedups body-vs-subject only on exact-equality; subject-as-
  prefix (common Zendesk) duplicates the quote. Fix: prefix/containment check. ~0.5 day.
- **P5-4 [low]** -- `_publishable_answer_detail:4286` shows a content-free "backed by evidence"
  sentence for proven-but-empty answers. Fix: explicit "no body drafted yet" wording. ~0.5 day.
- **P5-5 [low-medium]** -- no top-5 visual distinction (flat table `:4161`); Priority Fix Queue
  ABSENT from the hosted markdown surface (`surfaces` tuple `:595` lacks `"markdown"`) so markdown
  and PDF ship different section sets. Fix: top-5 separator + a product decision on markdown-vs-PDF
  parity (confirm whether the page renders from `report_model` `"web"` sections or the stored markdown).
- **P5-6 [product-gated]** -- snapshot withholds aggregate `estimated_support_cost`/annualized
  (`snapshot_safe_fields:543`); `locked_questions` carry only `{rank,ticket_count}` (`:796`).
  Adding the annualized number to the free tier is a **monetization decision, not cleanup** --
  the snapshot is a fail-closed allowlist / buyer-facing shape that changes ONLY by operator
  decision (the #1503 precedent). Surface as a proposal.

### Phase 6 richness -- wiring map (corrections in bold)
Ignored columns with concrete wiring: **priority** (no alias tuple exists -- add `_PRIORITY_KEYS`
~`:233`), **channel** (no tuple -- add `_CHANNEL_KEYS`), reply/comment count (comments aliased but
only flattened to text), distinct-requester (`contact_email` aliased, never rolled up), tags/
assignee (routing-only), **CSAT [corrected: avg already computed `:503`; real gap = CSAT-of-answered]**,
status time-in-status. Computable-but-not-computed: **week-over-week trend per cluster** (dates
already parsed, report collapses to one min/max `:4790` -- biggest missed signal), **reopen rate
[corrected: one-liner -- num `:1597` + denom `:1591` both exist]**, first-contact-resolution proxy,
cost-per-cluster over time. API-only ceiling (needs Zendesk Audit/Comments API): macro usage,
status transitions/handle-time, escalation depth, comment threads w/ role+timestamp, CSAT reason.

---

## Sequencing (by leverage / risk)

1. **R2 one-liner** (`lru_cache`+precompile `_key`): hours, 6x, near-zero risk. Do first.
2. **P5-2 cents unification + repeat-basis guard**: ~2 days, kills the trust-destroying money defect.
3. **R3 header (C1+H4+C2)**: ~2 days, warning-first non-breaking.
4. **R2 full key-index**: ~1-2 days, 55x.
5. **R6 (C3->M7)** + **M6/M8 text chokepoint** (PII-coordinated): ~3-4 days.
6. **F7 -> R1 unified clustering + calibration**: ~4-6 days -- the highest-value correctness fix,
   but needs a labeled calibration corpus and resolves the F4 band-overlap trade-off.
7. **R5 large-upload cap/async** (fix the byte-alias, `to_thread` the inline build): ~2-3 days.

## Self-corrections (post external review, 2026-07-04)

Independent adversarial review by GPT-5.5 Pro (portal) confirmed the money math (A1-A3, B1)
and caught real over-claims in the above, corrected in place (tracking #2000):
- **C1** -- the negative separation gap was cited as proof that single-linkage reproduces F4; it
  only proves *pairwise* inseparability. Closed with the MST connectivity bottleneck
  (`connect 0.535 > merge 0.295` -> no pure-2 threshold; sweep confirms). Conclusion stands, now proven.
- **A4** -- "cents reconciles by construction" scoped to base `n x $13.50`; fractional-cent
  annualized rows still need the raw-total guard.
- **B2** -- the reconciliation guard must call the same billed-repeat predicate, not a re-hardcoded `tc>=2`.
- **D1/D2** -- `$810/$1,944` are the dateless `x12` run-rate (`:3232`), not a `x365/window`
  annualization; annualized dollars are labeled run-rates, not confident annual claims (FINDINGS H-x12).

*No product code was modified. Prototypes in gitignored `_audit_scratch/`. The R1 semantic
placement (+ the C1 MST closure), R2 speedup+parity, P5-2 cents reconciliation, guard second-side
trap, and P7 byte-alias were re-run by the reviewer.*
