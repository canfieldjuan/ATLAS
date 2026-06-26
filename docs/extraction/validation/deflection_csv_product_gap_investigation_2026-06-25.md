# Deflection CSV-first product-gap investigation (2026-06-25)

Tracker: #1843 (CSV-first product-gap report). Slices: #1844 (S1), #1845 (S2),
#1846 (S3), #1847 (S4). Cross-links: #1386 (launch tracker), #1612 (full-report
surface QA harness).

> Provenance note: this document was committed to the repository on 2026-06-26
> from the investigation captured for #1843, with an independent verification
> pass appended (see "Independent verification" below). Every code reference in
> the evidence matrix was re-checked against the working tree at the head of
> `claude/csv-product-gap-verification-abbwid`. Line numbers are accurate to
> within a few lines unless a drift note says otherwise.

---

## 1. Why this investigation exists

We want to turn the existing FAQ-deflection report machinery into a CSV-first
**product-gap** deliverable without requiring Zendesk API access.

The target claim is deliberately **not** exact UI root cause. It is routeable,
evidence-backed product friction, e.g.:

> Customers repeatedly cannot find where to log in. This likely routes to an
> Auth / Product-UX discoverability owner lane. It repeated N times and costs
> about $X in assisted-contact handling. The source tickets and evidence are in
> the paid export.

Zendesk API / full-thread ingestion remains deferred. This investigation covers
only the CSV-first path and what the current code already supports versus what
must be built.

---

## 2. Evidence matrix

Each row: the claim, the code location, and the verification verdict.

| # | Claim | Location | Verdict |
|---|-------|----------|---------|
| 1 | Support-ticket normalization keeps core ticket fields but drops raw routing metadata (`Group`, `Tags`, `assignee`, `brand`, `organization`). | `extracted_content_pipeline/support_ticket_input_package.py:91-233`, `:503-558` | Confirmed |
| 2 | Only FAQ passthrough keys + product/sub_product/issue/sub_issue context survive normalization. | `support_ticket_input_package.py:177-193` | Confirmed |
| 3 | Public ticket rows are emitted from normalized rows only (underscore-prefixed keys stripped). | `support_ticket_input_package.py:409`, `:842-843` | Confirmed |
| 4 | The generic source adapter, by contrast, preserves all non-empty raw fields. | `campaign_source_adapters.py:830-914` (decisive: `:876-886`) | Confirmed |
| 5 | Hosted deflection submit routes through `build_support_ticket_input_package(...)`. | `api/control_surfaces.py:1499-1505` | Confirmed |
| 6 | CSV (ticket-index) and full-thread importer modes are already separate. | `api/control_surfaces.py:172-177` (`frozenset({"csv","full_thread"})`), `:1899-1932`, `:1941-1957`, `:2120-2145` | Confirmed |
| 7 | Provider fixture contract: full-thread exports can prove answers; ticket-index exports are gap-list only. | `examples/support_ticket_provider_exports/README.md:14-21` | Confirmed |
| 8 | Clustering already yields broad routeable owner lanes (login, billing, export, invite, password, ...). | `support_ticket_clustering.py:37-55`, `:151-222` | Confirmed |
| 9 | Per-gap repeat counts and costs already exist. | `ticket_faq_markdown.py:1316-1318`, `:1401-1404`; `faq_deflection_report.py:48-50`, `:2964-2984`, `:3192-3216` | Confirmed |
| 10 | Date-window logic already exists (would power monthly/batch labels). | `support_ticket_input_package.py:314-384`; `ticket_faq_markdown.py:1414-1435`; `faq_deflection_report.py:4207-4265` | Confirmed |
| 11 | The action queue is routing-capable but currently FAQ-shaped in copy. | `faq_deflection_report.py:3192-3216`, `:3349-3369`, `:3372-3399` | Confirmed (see add-vs-extend note) |
| 12 | Owner lane is currently just `topic or "Unknown"`. | `faq_deflection_report.py:3244-3246` | Confirmed |
| 13 | Representative phrasing is too thin for true customer vocabulary (<= 3 strings). | `faq_deflection_report.py:3359-3369` | Confirmed |
| 14 | Term mappings already exist (customer term -> documentation term). | `ticket_faq_markdown.py:2005-2059` | Confirmed |
| 15 | Evidence export is the uncapped audit lane. | `faq_deflection_report.py:1624-1657`, `:3850-3948` | Confirmed |
| 16 | Delivery email omits owner / evidence-tier / period fields. | `atlas_brain/content_ops_deflection_delivery.py:58-64`, `:507-567` | Confirmed |
| 17 | Existing email key numbers describe handling "in this upload". | `content_ops_deflection_delivery.py:358-420` | Confirmed |
| 18 | Paid result page renders FAQ-style gap cards, not product-gap cards. | `portfolio-ui/api/content-ops/deflection/result-page.js:378-473`, `:545-575` | Confirmed |
| 19 | Hosted allowlists deliberately exclude raw top evidence. | `portfolio-ui/src/types/deflectionReportModel.ts:56-72`, `:168-180` | Confirmed |
| 20 | Contract: free snapshots must not infer answer / evidence / source IDs. | `docs/frontend/content_ops_faq_report_contract.md:559-562` | Confirmed |
| 21 | Email leak tests exist. | `tests/test_atlas_content_ops_deflection_delivery.py:468-518` | Confirmed |
| 22 | Snapshot leak test exists. | `tests/test_smoke_content_ops_deflection_portfolio_result_page.py:309-323` | Confirmed |

---

## 3. Key code references (detail)

### 3.1 Normalization drops routing metadata
`support_ticket_input_package.py` defines a fixed set of alias key tuples
(`_SOURCE_ID_KEYS`, `_TEXT_KEYS`, `_PASSTHROUGH_KEYS`, `_STRUCTURED_CONTEXT_KEYS`,
`_DATE_KEYS`, `_STATUS_KEYS`, `_CSAT_KEYS`, ...). There is no alias slot for
`Group`, `Tags`, `assignee`, `brand`, or `organization`, so those columns are
silently dropped by `_normalize_ticket_row` (`:503-558`). Public rows are then
emitted only from normalized rows via `_public_ticket_row` (`:409`, `:842-843`),
which strips underscore-prefixed internal keys.

By contrast, `campaign_source_adapters.py:876-886`
(`source_row_to_campaign_opportunity`) preserves every non-empty field except
title-collision and private keys. The two ingestion paths are asymmetric by
design; the support-ticket path is the lossy one.

### 3.2 Importer-mode separation already exists
`api/control_surfaces.py:177` defines
`_DEFLECTION_SUBMIT_IMPORTER_MODES = frozenset({"csv", "full_thread"})`, the
request model defaults `importer_mode="csv"`, and the loader branches
`if importer_mode == "full_thread"` to JSON-thread parsing versus CSV parsing.
So S1 does not need to invent a mode split; it needs to preserve routing fields
within the existing CSV mode and label the evidence tier.

### 3.3 Economics are deterministic and already present
- Benchmark: `faq_deflection_report.py:49` -> `_ASSISTED_CONTACT_COST = 13.50`
  (`:50` label `"$13.50"`, narrative cites the "Gartner $13.50 assisted-contact
  benchmark").
- Formula: `_support_cost(ticket_count) = max(0.0, ticket_count * 13.50)`
  (`:4335-4336`), surfaced on the action item as
  `support_cost_formula = "ticket_count * assisted_contact_cost"` (`:3212-3213`).
- The whole report renderer and its markdown dependency
  (`ticket_faq_markdown.py`) contain **zero LLM calls** (verified by grep across
  both files and the transitive path). The "deterministic / no-LLM" property
  holds, so customer PII never leaves the stack.

### 3.4 Clustering owner lanes
`support_ticket_clustering.py` defines `_SINGLE_TOKEN_CLUSTER_LABELS`
(`:209-222`): `billing, cancel, email, export, api, invite, invoice, login,
password, payment, refund, subscription`, plus phrase folds (`:37-55`) and token
folds (`:151-208`) that normalize variants ("sign in"/"log in"/"locked out" ->
`login`, "charge"/"billed" -> `billing`, etc.). Clustering is token-set with a
0.6 overlap threshold and a high-frequency anchor-token fallback. It is
deterministic (no LLM).

---

## 4. Synthetic CSV proof (login discoverability)

A local, no-file-change probe fed five CSV rows asking variants of
"Where is the login button?" through the existing pipeline. Observed output:

| Field | Value |
|-------|-------|
| included rows | 5 |
| repeated item question | `Where is the login button?` |
| topic / current owner lane | `login` |
| repeated cluster ticket count | 4 |
| estimated support cost | `$54.00` |
| status | `Needs answer` |
| source date span | 2026-05-01 to 2026-05-05 |
| evidence | ticket IDs + quotes present in the action item |

Observed limitation: raw `Group` and `Tags` did not survive support-ticket
normalization (matches evidence-matrix rows 1-3), and recommendation copy was
FAQ-shaped rather than product-gap-shaped (rows 11-13).

### 4.1 Economics reconciliation
`$54.00 = $13.50 x 4` exactly. The cost is computed on the **clustered** repeat
count (4), not the raw row count (5, which would yield `$67.50`). This is
internally consistent with `_support_cost` and `_ticket_count` (which prefers the
explicit cluster `ticket_count` over `len(source_ids)`).

### 4.2 Why 5 rows produce a cluster of 4
Clustering is token-set with a 0.6 overlap gate and a >= 2 common-token minimum.
One phrasing variant of the five falls below the overlap threshold against the
main bucket and becomes a singleton, so the repeated cluster carries 4. This is
the same exact-match/token fragmentation behavior already noted as a launch
concern in #1386 ("clustering fragments raw, untagged exports into singletons").
It is expected, not a defect in the proof.

---

## 5. Claim ladder

**Can claim (CSV-first, evidence-backed):**

1. Customers repeatedly ask the same question or describe the same friction.
2. This is a repeated product/support gap.
3. It likely routes to a broad owner lane (Auth, Billing, Reporting, Admin,
   Notifications, Product UX, ...).
4. The repeated contacts carry an estimated handling cost using the
   assisted-contact benchmark.
5. The evidence trail (source ticket IDs + quotes) is available in the paid
   export.

**Must not claim without further evidence:**

1. Exact UI root cause.
2. Exact screen path.
3. A final customer-facing answer backed by public agent resolution evidence.
4. Public/private comment certainty beyond what the CSV columns prove.

---

## 6. Add-vs-extend: which model fields already exist

The `deflection.v1` action item (`faq_deflection_report.py` `_action_item`,
`:3196-3216`) is already routing/triage-shaped, not purely FAQ-shaped. It carries
`owner_lane`, `fix_type`, `csat_signal`, `confidence`, `priority_score`,
`priority_drivers`, `status`, `recommended_action`, `representative_phrasing`,
`ticket_count`, `estimated_support_cost`, `support_cost_formula`, `top_evidence`,
and stable identity keys (`repeat_key`, `cluster_id`, `identity_basis`,
`identity_confidence`). The TS model (`deflectionReportModel.ts`) also already
lists `owner_lane` and `fix_type` in its allowlists.

So the S2 (#1845) candidate-field list splits into two honest buckets:

| #1845 candidate field | Status today | Where |
|-----------------------|--------------|-------|
| `product_gap_summary` | New | (none) |
| `routing_signals` | New | (none) |
| `evidence_tier` | New | (none; `answer_evidence_status` is adjacent, not a tier) |
| `cost_period` | New | (annualized fields are implicit-period) |
| `cost_confidence` | New | (none) |
| `jira_template` | New | (nearest handle: `fix_type` enum, `:3249`) |
| `owner_lane` | Exists, weak | `:3201`, `:3244-3246` (`topic or "Unknown"`) + TS model |
| `owner_confidence` | Partial | subsumed by generic `confidence` (`:3204`, `:3281`) |
| `customer_vocabulary` | Partial | overlaps `customer_wording` + thin `representative_phrasing[:3]` |
| `batch_cost` | Partial | local var in `_support_tax_section` (`:3513`), not exported |
| `monthly_bleed` | Partial | overlaps `annualized_support_cost` / `annualized_run_rate_support_cost` (`:2937`, `:2941`) |

Implication: S2 is more accurately framed as **strengthen `owner_lane` (from
`topic or "Unknown"` to real routing using preserved metadata) + add the six
genuinely-new fields**, not "add owner_lane". S1 (#1844) is honestly framed
because it uses "preserve ... + add evidence tier": routing metadata genuinely
exists downstream and would be preserved/strengthened, and `evidence_tier` is
net-new.

---

## 7. Issue-ready breakdown

- **S1 / #1844 - preserve CSV routing metadata + evidence tier.**
  Add safe routing aliases (`group`, `assignee`, `tags`, `brand`,
  `organization`, `support_platform`, `product_area`, `custom_product_area`) to
  the support-ticket normalizer without breaking the private-text exclusions, and
  add a first-pass evidence/data tier (`csv_index_metadata_only`,
  `csv_customer_text`, `csv_full_thread_resolution_evidence`; the
  `api_full_thread_resolution_evidence` name can exist but stays deferred). This
  is the upstream root: every downstream lane is starved of routing signal
  because normalization drops it.

- **S2 / #1845 - report/action model fields.**
  Strengthen `owner_lane` to consume preserved routing metadata; add the six
  net-new fields (`product_gap_summary`, `routing_signals`, `evidence_tier`,
  `cost_period`, `cost_confidence`, `jira_template`); promote `customer_vocabulary`
  and a labeled period cost from the existing `customer_wording` /
  `annualized_*` material. Keep additive to `deflection.v1`.

- **S3 / #1846 - surfaces.**
  Compact product-gap summary in the delivery email (owner lane, period cost,
  evidence tier, next action, export link); paid/unlocked Product Gap cards on
  the result page. Free/locked snapshot stays unchanged and leak-safe; new fields
  land in paid-only allowlists.

- **S4 / #1847 - fixtures + QA.**
  Enroll the synthetic login probe as a deterministic fixture; add
  `zendesk_ticket_index_only.csv` and `zendesk_full_thread_export.csv` coverage;
  assert routing-metadata survival, evidence-tier labeling, cross-surface
  consistency, and the existing leak boundaries (extends the email-leak and
  snapshot-leak tests already present).

---

## 8. Independent verification (2026-06-26)

An independent verification pass re-checked all 22 evidence-matrix references and
the synthetic-proof economics. Result: code references are **highly accurate**
with no material line drift; the `$54.00 = $13.50 x 4` reconciliation is exact;
the no-LLM property holds for the report renderer and its markdown dependency;
and the add-vs-extend split in section 6 reflects fields that already exist in
`deflection.v1`.

Sequencing watch-item (from #1612): S1 and S2 are both model/ingestion-layer work
with no buyer-visible output until S3, which repeats the horizontal-slice pattern
#1612 flagged. Because `owner_lane` already exists end-to-end (model + TS
allowlist), a thin vertical alternative is available: take `owner_lane` from
`topic or "Unknown"` to real routing using preserved metadata and render it on one
surface in a single slice, before fanning out the remaining net-new fields.
