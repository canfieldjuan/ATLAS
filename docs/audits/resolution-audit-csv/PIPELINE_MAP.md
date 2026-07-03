# PIPELINE_MAP -- Resolution Audit CSV pipeline (Phase 0)

Deliverable for Slice 1 / #1957 of the audit arc #1956. Read-only map of the
actual data flow from raw CSV upload to the delivered Snapshot and Full Report.
Structure, not judgement: severity-rated defects are deferred to FINDINGS.md
(Slices 2-3). Where the map surfaces something a later slice must attack
empirically, it is tagged "-> Slice N".

All file:line anchors are under `extracted_content_pipeline/` unless prefixed
`atlas_brain/`, verified against `origin/main` @ `4f2790e6d`. (Corrected after a
Codex review of the first draft; every claim below was re-verified against the
code at that commit.)

## 0.0 Headline: two clustering systems; the token-set label seeds grouping, but the billed counts are computed in ticket_faq_markdown.py

- `support_ticket_clustering.py` -- a deterministic token-set clusterer, run at
  input-package build time. It produces preview/diagnostic counts and writes a
  `support_ticket_cluster` **label** onto each row.
- `ticket_faq_markdown.py` -- the report grouping that produces the `ticket_count`
  per FAQ item that every dollar figure multiplies against. Pipeline:
  topic-partition -> lexical sub-cluster -> optional embedding booster.

**The token-set label is NOT merely cosmetic** (corrected). `_topic()` reads the
`support_ticket_cluster` label first (`ticket_faq_markdown.py:1200-1202`, via
`_provided_support_ticket_cluster_topic` `:1218-1226`), and the report groups
rows by `(topic, evidence_group_key)` **before** lexical/embedding sub-clustering
(`:787-788`). So the label seeds the `topic` partition that gates which rows can
group together, and therefore influences `ticket_count`. The final count math
still lives in `ticket_faq_markdown.py`, but `support_ticket_clustering.py` is an
upstream input to it, not an irrelevant sidecar.

## 0.1 CSV ingestion

**Two real entry paths, both in `extracted_content_pipeline/api/control_surfaces.py`,
converging on one CSV engine** (module path corrected -- there is no
`atlas_brain/api/control_surfaces.py`).

- **A. Paid Resolution Audit ("deflection") path** -- `POST /deflection-reports/submit`
  -> `submit_deflection_report` (`api/control_surfaces.py:1483`). Buyer-facing.
- **B. Generic ingestion diagnostics/import** -- `POST /ingestion/files/{inspect,import}`
  (`api/control_surfaces.py:1258`, `:1286`) -> `ingestion_diagnostics.inspect_ingestion_file`.

Shared parser core: `_load_csv_dict_rows_result` (`campaign_customer_data.py:470`).

**Path A call path:** `submit_deflection_report` (`api/control_surfaces.py:1483`)
-> `_load_deflection_submit_rows_from_request` (`:1493`, blob cap 50 MiB) ->
`_load_deflection_submit_upload_rows` (`:1646`) -> tempfile, **streamed to disk in
1 MiB chunks** (`:1912`) -> `_parse_deflection_submit_csv_file` (`:1886`) ->
`load_csv_source_rows_result_from_file` (`campaign_source_adapters.py:568`) ->
`_load_csv_dict_rows_result` -> `_deflection_submit_english_rows` (`:1518`) ->
`build_support_ticket_input_package` (`support_ticket_input_package.py:301`) ->
`execute_generation`.

**Path B** reads the whole file into memory (`_read_bounded_upload`,
`api/control_surfaces.py:2806`, cap 25 MiB) before writing a temp file. -> Slice 3 (memory).

**Parser facts:**
- Stdlib `csv.reader` (not `DictReader`, not pandas), dynamic dialect, dict rows
  assembled manually (`campaign_customer_data.py:493-542`). Field-size limit 16 MiB.
- Header mapping is name-based + fuzzy, two stages: physical header-row detection
  (`_csv_header_index_and_hint`, `:731`) against a 45-name hint set; field aliasing
  (`_normalize_ticket_row`, `support_ticket_input_package.py:530-592`) via ordered
  alias tuples matched on `_key()` = `re.sub(r"[^a-z0-9]+","",lower)`.
- **Missing-header behavior (corrected):** a headerless CSV whose first nonblank
  row has >=2 nonempty cells is recorded as a **fallback header (no hint) and
  ACCEPTED** (`_csv_header_index_and_hint:746`; `_CsvDelimiterCandidate.valid:189`
  does not require a hint). `csv_missing_header` is raised **only** when no row
  has a known hint AND no row has >=2 nonempty cells (`:485`, `:1030`, `:1052`).
  -> Slice 2 (headerless/first-row-as-data corruption).
- Delimiter sniffed by a custom scorer (`_CSV_DETECT_DELIMITERS = ",;\t|"`,
  consistency floor 0.90). -> Slice 2.

## 0.2 Normalization

- HTML strip / entity decode (`support_ticket_clustering.py:311-330`). No
  signature / quoted-email removal exists. -> Slice 2.
- Dates: stdlib only (`support_ticket_dates.py:16-44`) -- ISO, then
  `date.fromisoformat(text[:10])`, then US formats; natural-language -> None.
  Naive timestamps not assigned a tz; tz-aware strings keep only `.date()`.
  If any included row lacks a parseable date, the dated window is disabled. -> Slice 2.
- **Row-level deduplication: none**, but this does NOT mean every duplicate
  inflates counts (corrected). `ticket_count` counts **distinct source keys**
  (`ticket_faq_markdown.py:1339-1340,1430`), `_normalize_ticket_row` preserves a
  stable id (`source_id = _first_text(row, _SOURCE_ID_KEYS) or f"ticket-{row_index}"`,
  `support_ticket_input_package.py:538`), and `build_ticket_faq_markdown` de-dupes
  repeated `(source_id, text)` evidence (`:763-767`). So duplicates that carry a
  **stable `source_id`/`ticket_id` collapse**; inflation happens only for
  duplicates with **missing or changed IDs** (each gets a unique `ticket-<row>` /
  `row:<index>` key). -> Slice 2 (dedup / ID stability).
- **Resolution field:** `resolution_text = _first_text(row, _RESOLUTION_TEXT_KEYS)`
  (`:544`), free-text evidence, HTML-cleaned + clipped to 500. Absent/empty ->
  key unset, no error, no drop; never gates inclusion.
- **Status field (narrowed):** read from the status FIELD only --
  `_STATUS_KEYS = (ticket_status, issue_status, case_status, status, ticket_state,
  state)` (`:234-241`) -- and that field's value is bucketed by
  `_normalize_status_state` (`:817`) into reopened/resolved/cancelled/open/other
  (unknown -> "other"). There is **no** separate "Solved/Closed flag" or
  macro-name parsing; a value like "solved"/"closed" is handled only when it is
  the value of one of those status columns. Never gates inclusion.

**Row drop / skip points** (-> Slice 2):
- CSV engine: blank-only rows silently dropped, not counted (`:524-525`); **long
  rows (more cells than header) hard-reject the whole file** (`:527-531`); **short
  rows (fewer cells) are ACCEPTED with trailing fields padded** (corrected: `value
  = values[i] if i < len(values) else ""`); only a single-cell row containing a
  competing delimiter invalidates the candidate. `max_rows` truncation counted-but-
  not-appended.
- `_normalize_ticket_row`: no usable text -> row dropped `ticket_row_missing_text`
  (`:534-536`).
- Deflection endpoint: non-English rows dropped when some row is language-tagged.
- **Private-row seam (corrected).** The paid path DOES call
  `source_rows_to_campaign_opportunities` inside `TicketFAQMarkdownService.generate`
  (`ticket_faq_markdown.py:617`), and that path HAS a top-level private filter
  (`source_row_to_campaign_opportunity` -> `_is_private_source_row` on
  is_private/is_internal/public, `campaign_source_adapters.py:838-846`,`:1375`).
  The actual gap is upstream: `_normalize_ticket_row`
  (`support_ticket_input_package.py:530`) never reads or copies
  `is_private`/`is_internal` (not in its passthrough whitelist `:177-187`), so on
  the support-ticket-upload path the flag is **stripped before** the downstream
  filter can act -> a top-level private ticket leaks. (Per-comment privacy IS
  handled narrowly: `_comment_text` returns empty only when a comment mapping has
  `public is False` (`:680-684`) -- it does not check `is_private`/`is_internal` on
  comments.) -> Slice 2 (PII/privacy).

## 0.3 Clustering (the real report path)

Ordered stages in `ticket_faq_markdown.py` (entry `build_ticket_faq_markdown`, `:710`):
- **5a Coarse partition** by `(topic, evidence_group_key)` (`:787-788`); topic =
  `_topic()` (reads the token-set label first); evidence key =
  `_evidence_group_key(resolution_text)`.
- **5b Lexical sub-cluster** of degraded `topic:*` buckets only
  (`_question_subclusters`, `:1026`): exact gist match then exact-Jaccard >= 1/3,
  union-find.
- **5c Embedding booster (optional)** applied ONLY to components still singleton
  after 5b (`:1115-1119`). No unified semantic pass over compressed representatives.
- **5d Singleton exclusion (narrowed):** the `<2 distinct-source` exclusion runs
  **only inside the `topic:*` branch** (`:846-848`, `:856-860`).
  **Resolution-scoped groups (`evidence_group_key` set -> `scope=resolution:...`)
  bypass `_question_subclusters` entirely and can render with `ticket_count == 1`**
  (corrected -- not every single-source group is excluded). -> Slice 2.
- **5e** sort + `max_items` overflow -> "other support issues".
- **5f** `ticket_count = len(distinct source_ids)` (`:1430`) -- drives all cost math.

**Critical architecture answer (-> Slice 2 core):** the lexical stage does the
real, usually final clustering; the embedding pass is a leftover-singleton rescue.
No unified semantic pass. This is the architecture that risks fragmenting
high-volume, low-lexical-overlap questions -- Slice 2 must prove/quantify it.

**Threshold table (all hardcoded module constants unless noted):**

| Threshold | Value | file:line |
|---|---:|
| Token-set row cap (skip preview) | 2000 | `support_ticket_clustering.py:249` (param) |
| Token overlap ratio | >= 0.6 | `support_ticket_clustering.py:582` |
| Anchor token min doc-frequency | >= 2 | `support_ticket_clustering.py:658-669` |
| Sub-cluster Jaccard | 1/3 | `ticket_faq_markdown.py:977` |
| Gist token limit | 30 | `ticket_faq_markdown.py:978` |
| Embedding MNN cosine floor | 0.78 | `ticket_faq_markdown.py:979` |
| Embedding MNN margin | 0.05 | `ticket_faq_markdown.py:980` |
| Min distinct sources to keep a `topic:*` cluster | < 2 excluded | `ticket_faq_markdown.py:862` |
| Repeat-ticket threshold (billed) | ticket_count >= 2 | `faq_deflection_report.py:4735` |
| Assisted-contact cost | $13.50 | `faq_deflection_report.py:51` |

(Anchors corrected: the token-set constants live in `support_ticket_clustering.py`;
there is no `clustering.py`.) -> Slice 2 runs the +/-10% sweep.

- **Representative question:** first-match-wins over group row order for a row
  passing `_publishable_customer_question_text` (`:2659`); else synthesized.
  **That gate checks question-shape + customer-heading PII, NOT spam/auto-reply**
  (corrected): a junk/auto-reply row phrased as a normal question with no heading
  PII CAN become the displayed representative -- and any group member (junk or not)
  counts toward `ticket_count`. -> Slice 2 (junk representative + inflation) & Slice 5.
- **Determinism (-> Slice 2):** `support_ticket_clustering.py` anchor promotion is
  order-dependent (`:556-561`); the representative question is first-match-wins on
  row order (`:2372`); embedding MNN float-tie comparisons are tie-fragile (stable
  only because the model is CPU-pinned).

## 0.4 Metrics & cost math (every report number traced to source)

- **weighted_frequency:** `weighted_source_volume_by_group` (`:1673`) sums
  `max(source-weight,1)` per distinct source key; weight from `_SOURCE_WEIGHT_KEYS`,
  default 1.
- **ticket_count:** distinct source keys per group (`:1430`).
- **cost:** `estimated_support_cost = ticket_count * 13.50` (`_support_cost`,
  `faq_deflection_report.py:4888-4889`); `$13.50` hardcoded Gartner benchmark, not
  from the CSV.
- **handle time: does not exist** -- cost has no time component.
- **aggregate/annualized/run-rate:** `_support_cost(repeat_ticket_count)`,
  `* 365/window` or `* 12` (`:3228-3233`).

**Double-computation (drift risk -> Slice 2/5):** aggregate/annualized/run-rate and
repeat/non-repeat counts are computed twice (structured model `_support_tax_data:3208`
vs prose `_support_tax_section:4034`); per-item cost in >=4 places, and
`_snapshot_estimated_support_cost` prefers a pre-existing numeric override (`:2514-2516`).

## 0.5 Snapshot vs Full Report -- ONE path (projection) + defensive recompute fallback

- Full Report `report_model` computed once (`build_deflection_report_artifact`,
  `faq_deflection_report.py:1721`); Snapshot is a projection of it
  (`build_deflection_snapshot:2625` projection branch `:2642-2649`; recompute
  fallback `:2650-2698` uses the same helpers).
- `deflection.v1` projection (`_snapshot_report_model_projection:2833`): per-section
  `snapshot_safe_fields` allowlist. `question_details` exposes rank/question/
  evidence-status/scope but NOT `answer`/`steps` (except a single-item teaser).
- **Delivery artifacts (corrected):** the delivery email and PDF read the persisted
  `report_model` (backend); **the MCP fetch/search path reads the stored snapshot**
  (`snapshot = dict(record.snapshot)`,
  `atlas_brain/mcp/content_ops_deflection_readonly_server.py:181`; search `:134`);
  the **hosted result page is served by the portfolio-ui frontend loading the stored
  artifact** via a `portfolio-ui/api/content-ops/deflection/...` route (exact fetch
  path -> Slice 5). So it is not accurate that every artifact reads the shared
  `report_model`. -> Slice 5.

## 0.6 Verification / scorecard -- coverage and gaps

Two verifiers, neither runs at customer-delivery runtime:
- **`build_deflection_full_report_qa_scorecard`** (`faq_deflection_report.py:1781`)
  -- callers are tests + `scripts/check_deflection_full_report_*` only (zero runtime
  callers). Checks schema/sections/keys present, evidence-export counts == model
  counts, surface counts == model counts. **Does NOT** reconcile totals=sum-of-parts,
  verify quotes verbatim (checks row COUNT only), or run on the snapshot.
- **`evaluate_support_ticket_generated_content`**
  (`support_ticket_generated_content_eval.py:607`) -- a different artifact's linter
  (marketing landing_page/blog_post), not the deflection report/snapshot.

Net: customer-facing numeric integrity (totals summing, quotes verbatim) is not
gated by either verifier at runtime. -> Slice 5 & SUMMARY.

- **Blind-spot handling is honest** (`_no_proven_answer_detail:4319`); one
  placeholder to watch (`_publishable_answer_detail:4286` substitutes a generic
  "backed by evidence" sentence for a proven item with an empty answer body). -> Slice 5.

## 0.7 Dead / orphaned code

- `faq_deflection_report.py:3046` `render_deflection_report` -- no in-repo call
  sites, but exported in `__all__` (`:5570`), so it is external API surface, not
  strictly dead. Superseded internally by `render_deflection_report_model`.
- `campaign_customer_data.py:1258` `_validate_csv_column_consistency`,
  `:726 _csv_header_index` -- 0 references.
- `support_ticket_input_package.py:890 _all_rows_have_dates`,
  `:949 _parse_ticket_source_date` -- 0 references.
- `campaign_source_adapters.py:537 load_source_rows_from_file` -- exported public
  helper (`__all__` `:1473`), test-exercised; the internal API path uses
  `load_source_rows_with_warnings_from_file`. Public surface, not a dead orphan.
- **NOT orphaned (corrected):** `campaign_customer_data.py:376
  FileIntelligenceRepository` is a documented host-facing customer-file adapter
  (exported in `__all__`, documented in `extracted_content_pipeline/README.md:114`
  and `docs/standalone_productization.md:155`), not a test-only orphan.
- **Effectively inert:** the token-set overlap/anchor thresholds shape the topic
  label (which, per 0.0, seeds grouping). The embedding booster is wired but OFF on
  the deflection call path unless the service was built with `config.embedding_port`.
  -> Slice 2 must confirm whether embeddings run in production.

## 0.8 Flags carried into later slices

| Flag | Where | Slice |
|---|---|---|
| Lexical produces FINAL clusters; embeddings only rescue singletons | `ticket_faq_markdown.py:1026/:1115` | 2 (core) |
| Missing/changed-ID duplicates inflate `ticket_count`; stable-ID dups collapse | ingestion + `:763/:1430` | 2, 5 |
| Headerless CSV first-row-as-data accepted; short rows padded | `campaign_customer_data.py:746/:541` | 2 |
| Resolution-scoped single-source groups can render with count 1 | `ticket_faq_markdown.py:846` | 2 |
| Publishable gate is PII, not spam -> junk can be the representative | `:2659` | 2, 5 |
| Order-dependence vs "deterministic positioning" claim | `ticket_faq_markdown.py:2372`, `support_ticket_clustering.py:556` | 2 |
| Private-row flag stripped in `_normalize_ticket_row` before the downstream filter | `support_ticket_input_package.py:530` | 2 |
| Whole-file-into-RAM on Path B; metrics double-computed | `api/control_surfaces.py:2806`; `:3208`/`:4034` | 3 |
| Embedding booster runs only if the FAQ service was built with `config.embedding_port`; deflection dispatch (`content_ops_execution.py:1055`) calls `service.generate()` without forwarding one | `_content_ops_services.py` | 2, 3 |
| Scorecard: no totals reconciliation, no verbatim check, not at runtime, not on snapshot | `:1781` | 5, SUMMARY |
| MCP fetch reads stored snapshot, not report_model | `content_ops_deflection_readonly_server.py:181` | 5 |
| Cost = flat $13.50, no handle-time, hardcoded benchmark | `:51` | 5, 6 |

*Phase 0 map only -- no product code was modified. Findings with severity are in
FINDINGS.md (Slices 2-3).*
