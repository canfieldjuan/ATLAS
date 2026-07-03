# PIPELINE_MAP — Resolution Audit CSV pipeline (Phase 0)

Deliverable for **Slice 1 / #1957** of the audit arc **#1956**. This is the
read-only **map** of the actual data flow from raw CSV upload to the delivered
Snapshot and Full Report. It is structure, not judgement: severity-rated
defects are deferred to `FINDINGS.md` (Slices 2–3). Where the map surfaces
something a later slice must attack empirically, it is tagged **→ Slice N**.

All `file:line` anchors are under `extracted_content_pipeline/` unless prefixed
`atlas_brain/`, verified against `origin/main` @ `4f2790e6d`.

---

## 0.0 Headline: there are TWO clustering systems, and the one named "clustering" is not where the numbers come from

This is the single most important thing to hold while reading the rest:

- **`support_ticket_clustering.py`** — a deterministic **token-set** clusterer.
  Runs at input-package build time. It only (a) produces preview/diagnostic
  counts and (b) writes a `support_ticket_cluster` **label** onto each row that
  later becomes the **topic seed**. Its overlap/anchor thresholds (0.6, anchor
  logic) shape the topic **label only**; they do **not** decide repeat groups or
  ticket counts.
- **`ticket_faq_markdown.py`** — the **real report grouping**. Produces the
  `ticket_count` per FAQ item that **every dollar figure is multiplied
  against**. Pipeline: topic-partition → lexical sub-cluster → optional
  embedding booster.

Link between them: `support_ticket_input_package.py:370` runs the deterministic
clusterer; `_public_ticket_row` (`support_ticket_input_package.py:928`) keeps
every non-`_`-prefixed key, so `support_ticket_cluster` survives into
`source_material` (`:434`), and `_topic()` reads it first
(`ticket_faq_markdown.py:1200-1202`, `:1218-1225`). **The money is computed in
`ticket_faq_markdown.py`, not in the file called `clustering`.**

---

## 0.1 CSV ingestion

**Two real entry paths, both routed through `atlas_brain/api/control_surfaces.py`, converging on one CSV engine.**

- **A. Paid Resolution Audit ("deflection") path** — `POST /deflection-reports/submit`
  → `submit_deflection_report` (`control_surfaces.py:1483`). Buyer-facing.
- **B. Generic ingestion diagnostics/import** — `POST /ingestion/files/{inspect,import}`
  (`control_surfaces.py:1258`, `:1286`) → `ingestion_diagnostics.inspect_ingestion_file`.

Shared parser core: **`_load_csv_dict_rows_result` (`campaign_customer_data.py:470`)**.

**Path A call path (multipart CSV upload):**
1. `control_surfaces.py:1483` `submit_deflection_report`.
2. `:1493` `_load_deflection_submit_rows_from_request` (dispatch by content-type; blob cap `_MAX_DEFLECTION_SUBMIT_BLOB_BYTES = 50 MiB`, `:162`).
3. `:1646` `_load_deflection_submit_upload_rows` → `tempfile.NamedTemporaryFile(suffix=".csv")` (`:1870`), **streamed to disk in 1 MiB chunks** (`:1912`).
4. `:1886` `_parse_deflection_submit_csv_file` → `load_csv_source_rows_result_from_file` (`campaign_source_adapters.py:568`) → `_load_csv_dict_rows_result`.
5. `:1518` `_deflection_submit_english_rows` (language filter) → `:1530` `_deflection_submit_rows_with_defaults`.
6. `:1539` `build_support_ticket_input_package` (`support_ticket_input_package.py:301`) — normalization.
7. `:369` `assign_support_ticket_clusters_with_diagnostics` — label annotation → `execute_generation` (`:1572`).

**Path B** differs only in staging: `_read_bounded_upload` reads the **whole file into memory** (`control_surfaces.py:2808`, `_MAX_INGESTION_FILE_BYTES = 25 MiB`) before writing a temp file. → Slice 3 (memory).

**Parser facts:**
- Stdlib **`csv.reader`** (not `DictReader`, not pandas), dynamically-built dialect, dict rows assembled manually (`campaign_customer_data.py:493-542`). Field-size limit raised to 16 MiB (`:1227-1234`).
- **Header mapping is name-based + fuzzy, never positional**, in two stages:
  - Physical header-row detection by content (`_csv_header_index_and_hint`, `:731-746`) against a 45-name hint set (`_CSV_HEADER_HINTS`, `:45-90`); leading non-header rows skipped + reported (`csv_leading_rows_skipped`, `:502-507`).
  - Field aliasing (`_normalize_ticket_row`, `support_ticket_input_package.py:530-592`) via ordered alias tuples matched on `_key()` = `re.sub(r"[^a-z0-9]+","",lower)` (`:990-991`) — so `"Ticket Subject"`/`"ticket_subject"`/`"TICKETSUBJECT"` collide.
  - Reordered headers: fine. Duplicate headers: **last-wins** (`:542`). Missing header row: **hard reject** `csv_missing_header` (`:562-567`). Unknown columns: preserved as extra keys.
- **Delimiter sniffed by a custom scorer** (not `csv.Sniffer`): `_CSV_DETECT_DELIMITERS = ",;\t|"` (`:22`); scored on header-hint presence + column-count consistency (`_CSV_DELIMITER_MIN_CONSISTENCY = 0.90`, `:26`); inconsistent width → `csv_inconsistent_columns` reject. → Slice 2 (adversarial structure fixtures).

## 0.2 Normalization

- **HTML strip / entity decode:** `support_ticket_plain_text` (`support_ticket_clustering.py:311-330`) via `_HTMLTextExtractor` then `html.unescape` + whitespace compaction. Applied to titles/body/comments/resolution (`support_ticket_input_package.py:533`, `:546`, `:647-657`).
- **Signature / quoted-email removal: NONE FOUND.** No trailing-signature or `>`-quoted-reply stripping. → Slice 2 (quoted-chain fixtures).
- **Lowercasing:** ticket text is **not** lowercased for storage/output; only inside clustering tokenization (`support_ticket_tokens`, `clustering.py:341`).
- **Dates:** `parse_support_ticket_source_date` (`support_ticket_dates.py:16-44`), stdlib only (no dateutil): ISO, then `date.fromisoformat(text[:10])`, then US formats `%m/%d/%Y|%m/%d/%y|%m-%d-%Y|%m-%d-%y`. Natural-language dates → `None`.
- **Timezone:** naive timestamps are **not** assigned a tz; tz-aware strings keep only `.date()` (time + tz discarded, `support_ticket_dates.py:23-24`). If **any** included row lacks a parseable date, the whole dated window is disabled (`has_valid_date_window` requires `missing_count == 0`, `support_ticket_input_package.py:390-403`). → Slice 2 (mixed date formats).
- **Deduplication: NONE.** Grep across ingestion modules returns zero `dedup|unique|distinct|seen_ids`. Duplicate tickets survive as separate rows and each counts toward `ticket_count`. → Slice 2 (dedup / determinism).
- **Resolution field:** `resolution_text = _first_text(row, _RESOLUTION_TEXT_KEYS)` (`:544`); broad free-text aliases (`:138-161`). Treated as **free-text evidence**, HTML-cleaned + clipped to 500 chars, drives evidence tier (`:595-600`). **Absent/empty → key unset, no error, no drop**; evidence tier falls back; never gates inclusion.
- **Status field:** `_first_text(row, _STATUS_KEYS)` (`:579`), bucketed by `_normalize_status_state` (`:817-829`) into `reopened/resolved/cancelled/open/other`; **unknown vocabulary → `"other"`** (deliberate). Handles Solved/Closed flag, macro name, or free text — all as text buckets. Never gates inclusion.

**Row drop / skip points** (silent-corruption candidates → Slice 2):
- CSV engine: blank-only data rows silently dropped, **not counted** in `source_row_count` (`:524-525`); rows with more cells than header → **whole-file reject** (`:527-531`); `max_rows` truncation counted-but-not-appended (`:543-544`).
- `_normalize_ticket_row`: **no usable text** (no title/body/public comments) → row dropped `ticket_row_missing_text` (`:534-536`, `:342-358`).
- Deflection endpoint: non-English rows dropped when *some* row is language-tagged (`:2503-2514`).
- **Nuance:** the row-level private/internal skip (`private_source_text`) lives only in `source_row_to_campaign_opportunity`, which **the paid deflection path does not call** — Path A honors only per-comment privacy (`_comment_text` drops `public is False`, `:682-683`), not a row-level `is_private`/`is_internal` flag. → Slice 2 (PII/privacy).

## 0.3 Clustering (the real report path)

Ordered stages in `ticket_faq_markdown.py` (entry `build_ticket_faq_markdown`, `:710`):
- **5a Coarse partition** by `(topic, evidence_group_key)` (`:738`, keyed `:788`); topic = `_topic()` (`:1194`, reads the token-set label first); evidence key = `_evidence_group_key(resolution_text)` (`:1228`).
- **5b Lexical sub-cluster** of degraded (`topic:*`) buckets (`_question_subclusters`, `:1026`): exact gist-frozenset match (`:1058-1066`) then prefix-filtered exact-Jaccard ≥ 1/3 (`:1067-1080`), union-find.
- **5c Embedding booster (optional)** (`_apply_embedding_booster`, `:1097`, embed call `:1123`): mutual-nearest-neighbor cosine merge, **applied ONLY to components still singleton after 5b** (`active_indexes = [… if component_sizes[find(index)] == 1]`, `:1115-1119`).
- **5d** sub-cluster with `< 2` distinct source keys → excluded to `non_repeat` (`:862-865`).
- **5e** sort + `max_items` overflow → "other support issues" (`:889-904`).
- **5f** `ticket_count = len(distinct source_ids)` (`:1430`) — **this integer drives all cost math.**

**Critical architecture answer (→ Slice 2 core):** the lexical stage does the
**real, usually final** clustering; the embedding pass is a **leftover-singleton
rescue**. There is **no unified semantic pass over compressed representatives**.
This is the architecture the brief warns fragments high-volume, low-lexical-
overlap questions and undercounts the #1 issue — Slice 2 must prove/quantify it.

**Threshold table (all hardcoded module constants unless noted):**

| Threshold | Value | file:line |
|---|---|---|
| Token-set row cap (skip preview) | 2000 | `clustering.py:249` (param `max_token_set_rows`) |
| Token overlap ratio | ≥ 0.6 | `clustering.py:582` |
| Anchor token min doc-frequency | ≥ 2 | `clustering.py:671` |
| Sub-cluster Jaccard | 1/3 | `ticket_faq_markdown.py:977` |
| Gist token limit | 30 | `ticket_faq_markdown.py:978` |
| Embedding MNN cosine floor | 0.78 | `ticket_faq_markdown.py:979` |
| Embedding MNN margin | 0.05 | `ticket_faq_markdown.py:980` |
| Min distinct sources to keep a cluster | < 2 excluded | `ticket_faq_markdown.py:862` |
| Repeat-ticket threshold (billed) | ticket_count ≥ 2 | `faq_deflection_report.py:4735` |
| **Assisted-contact cost** | **$13.50** | **`faq_deflection_report.py:51`** |

→ Slice 2 runs the ±10% sweep on these.

- **Representative question:** `_question` (`:2367`) = **first-match-wins over group row order** for a row passing `_publishable_customer_question_text` (`:2659`); else synthesizes `"What should I do about <label>"` (`:2409`). A spam/auto-reply ticket **cannot be the displayed question** (must clear the no-PII/publishable gate) but **can be a group member and inflate `ticket_count`** → inflates every `$13.50 × count`. → Slice 2 (junk-inflation) & Slice 5 (grounding).
- **Singletons/residuals:** report path excludes `<2`-source clusters into `non_repeat_ticket_count` (surfaced, not billed, `:862-875`); overflow → "other support issues" (`:902`). Nothing silently dropped or silently billed.

**Determinism (→ Slice 2):**
- `support_ticket_clustering.py` is **order-dependent** — anchor promotion mutates a bucket's key/label based on which row matched first (`:556-561`); same rows in a different order can yield a different anchor label.
- Report path is mostly order-hardened (`sorted` groups `:889`, union-find `min/max` roots), **but** the representative question is first-match-wins on row order (`:2372`), and the embedding MNN float-tie comparisons (`:1148`, `:1157-1161`) are tie-fragile (stable only because the model is CPU-pinned; latent if ever GPU-backed).

## 0.4 Metrics & cost math (every report number traced to source)

- **volume / weighted_frequency:** `weighted_source_volume_by_group` (`ticket_faq_markdown.py:1673`) sums `max(source-weight,1)` per distinct source key; `source_row_weight` (`:1702`) reads `_SOURCE_WEIGHT_KEYS` (`source_weight, search_count, volume, frequency, …`, `:245-259`), **default 1** if absent. `opportunity_score = frequency * (1 + failure_risk_score)` (`:1536`).
- **ticket_count:** **derived, not a CSV column** = distinct source keys per group (`:1430`).
- **cost:** `estimated_support_cost = ticket_count * 13.50` (`_support_cost`, `faq_deflection_report.py:4888-4889`); `_ASSISTED_CONTACT_COST = 13.50` is a **hardcoded Gartner benchmark** (prose `:4049`), **not from the CSV**.
- **handle time: DOES NOT EXIST** — repo-wide grep for `handle_time|aht|minutes_per|hourly|wage` returns nothing. Cost has no time component: flat per-contact rate × repeat count.
- **aggregate:** `_support_cost(repeat_ticket_count)`, `repeat_ticket_count = sum(ticket_count for items with ticket_count≥2)` (`:4728-4736`).
- **annualized (dated window):** `_support_cost(repeat_ticket_count * 365 / source_window_days)` (`:3228-3230`).
- **run-rate (no window):** `_support_cost(repeat_ticket_count * 12)` (`:3232-3233`).

**Double-computation (drift risk → Slice 2/5 reconciliation):**
- Aggregate/annualized/run-rate and repeat/non-repeat counts are each computed **twice** — structured model (`_support_tax_data:3208`) vs markdown prose (`_support_tax_section:4034`) — re-derived independently. Values agree today but drift if only one branch is edited.
- Per-item `estimated_support_cost` is computed in ≥4 places; **`_snapshot_estimated_support_cost` prefers a pre-existing numeric `item["estimated_support_cost"]`** (`:2514-2516`) while the other three always recompute — so the snapshot can surface an overridden/stale value.

## 0.5 Snapshot vs Full Report — ONE path (projection) + defensive recompute fallback

- Runtime entry: `FAQDeflectionReportService.generate` (`faq_deflection_report.py:1676`) → `build_deflection_report_artifact` (`:1721`). The **Full Report `report_model` is computed once**; the **Snapshot is a projection of it**.
- `build_deflection_snapshot` (`:2625`) has two branches: **projection** (`:2642-2649`, `_build_deflection_snapshot_from_report_model`, reads already-computed sections) — taken in production; and a **recompute fallback** (`:2650-2698`) for missing/legacy models that re-derives numbers via the **same helpers** (drift-safe but redundant).
- **`deflection.v1` projection** (`_snapshot_report_model_projection`, `:2833`): per-section allowlist — only sections with `snapshot_safe_fields` survive, only those fields. Exposed: `support_tax` counts, `ranked_questions` rows, `top_unresolved_repeats`, `question_details` (rank/question/evidence-status/scope — **NOT `answer`/`steps`**), plus a single-item teaser exposing the full `answer`+`steps` (`_teaser_full_answer`, `:4694`). Withheld for paid: all markdown, `answer`/`steps`/`evidence_quotes`/`source_ids` (except the teaser), and every section lacking `snapshot_safe_fields`.
- **Customer-facing artifacts** all read the single persisted `report_model` (no delivery-time recompute): delivery email (`atlas_brain/content_ops_deflection_delivery.py:738/:773`), hosted page (persisted `artifact.markdown`), PDF (`atlas_brain/deflection_pdf_renderer.py:169`, reads `stored_deflection_report_model`), MCP snapshot (`atlas_brain/mcp/content_ops_deflection_readonly_server.py:149`). → Slice 5 (presentation + reconciliation).

## 0.6 Verification / scorecard — coverage and gaps

Two independent verifiers, **neither runs at customer-delivery runtime**:

- **(A) `build_deflection_full_report_qa_scorecard`** (`faq_deflection_report.py:1781`) + deterministic harness (`:1851`). **Callers: tests + `scripts/check_deflection_full_report_*` smoke only — zero runtime callers.** Checks: schema_version, required sections/keys present, evidence-export counts equal model counts, surface-observed counts equal model counts + caps respected. **Does NOT:** reconcile totals=sum-of-parts (no `repeat + non_repeat == source_count`, no `cost == count × rate` assertion); verify quotes verbatim (checks evidence **row count** only, never quote content); run on the **snapshot**.
- **(B) `evaluate_support_ticket_generated_content`** (`support_ticket_generated_content_eval.py:607`) — a **different artifact's** linter (marketing landing_page/blog_post, wired at `blog_generation.py:1178`); percentages must equal source-count ratios, ≥1 cluster label must appear. Does not run on the deflection snapshot/report.

**Net:** customer-facing numeric integrity (totals summing, quotes verbatim) is **not gated by either verifier at runtime.** → Slice 5 (scorecard visibility) & SUMMARY.

- **Blind-spot handling is honest:** no grounded answer → `_no_proven_answer_detail` (`:4319`, "No proven answer yet…"); missing quotes stated not faked (`:4360-4368`). One placeholder to watch: `_publishable_answer_detail` (`:4286`) substitutes a generic "backed by evidence" sentence when a proven item's `answer` body is empty — can mask an empty-but-proven answer. → Slice 5.

---

## 0.7 Dead / orphaned code

- `faq_deflection_report.py:3046` **`render_deflection_report`** — dead, zero callers (only `__all__` export `:5570`); superseded by `render_deflection_report_model`.
- `campaign_customer_data.py:1258` `_validate_csv_column_consistency` — 0 references.
- `campaign_customer_data.py:726` `_csv_header_index` — 0 references (superseded by `_csv_header_index_and_hint`).
- `support_ticket_input_package.py:890` `_all_rows_have_dates`, `:949` `_parse_ticket_source_date` — 0 references.
- `campaign_source_adapters.py:537` `load_source_rows_from_file`, `campaign_customer_data.py:375` `FileIntelligenceRepository` — test-only, not on any product path.
- **Effectively inert (not dead):** the token-set module's overlap/anchor thresholds feed the topic **label only**, not `ticket_count`/cost. The embedding booster is **wired but OFF** on the deflection call path — `content_ops_execution.py:1066` and `FAQDeflectionReportService.generate` (`:1698`) never forward `embedding_port`; it engages only if the service was built with `config.embedding_port` (host path, `atlas_brain/_content_ops_services.py:293/:477`, model `mixedbread-ai/mxbai-embed-large-v1`, CPU). → Slice 2 must confirm whether embeddings run in production.

## 0.8 Flags carried into later slices

| Flag | Where | Slice |
|---|---|---|
| Lexical produces FINAL clusters; embeddings only rescue singletons — fragmentation risk | `ticket_faq_markdown.py:1026/:1115` | **2 (core)** |
| No row-level dedup; junk/dup tickets inflate `ticket_count` → `$13.50 × count` | ingestion + `:1430` | **2, 5** |
| Order-dependence (anchor label, first-match representative, float-tie MNN) vs "deterministic positioning" claim | `clustering.py:556`, `ticket_faq_markdown.py:2372/:1148` | **2** |
| Threshold ±10% sensitivity of top-5 | threshold table | **2** |
| Whole-file-into-RAM on Path B; metrics double-computed | `control_surfaces.py:2808`; `:3208`/`:4034` | **3** |
| Embedding booster maybe OFF in production | `content_ops_execution.py:1066` | **2, 3** |
| Scorecard: no totals reconciliation, no verbatim check, not at runtime, not on snapshot | `:1781` | **5, SUMMARY** |
| Snapshot may surface overridden/stale `estimated_support_cost` | `:2514` | **5** |
| Cost = flat $13.50, no handle-time, hardcoded benchmark | `:51` | **5, 6** |

*Phase 0 map only — no product code was modified. Findings with severity are in `FINDINGS.md` (Slices 2–3).*
