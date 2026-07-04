# PRESENTATION -- Resolution Audit CSV output + richness (Phases 5-6)

Deliverable for Slice 4 / #1960. Read-only judgement of the real generated artifacts
(email, hosted page, PDF) and the richness ceiling. Positioning/branding are out of
scope. Every number came from a real run dumped to `_audit_scratch/s4_out/`. The
money-reconciliation finding (P5-2) was independently re-verified by the reviewer.
`file:line` anchors under `extracted_content_pipeline/` unless prefixed.

## Phase 5 -- Presentation

### P5-1 Narrative pull -- MIXED (dead spot in the middle)
Section order is priority-sorted (`faq_deflection_report.py:525-720`): Support Tax (10)
-> Source file (15) -> **SEO Targeting List (20)** -> Ranked Question Opportunities (30)
-> action queues (35-39) -> Outcome Diagnostics (40) -> Question Details + Evidence (50).
The dollar hook leads (good), but the **SEO Targeting List sits above the ranked
opportunities and the fix queue** and is the lowest-value section for a support leader
(it restates the questions as "phrases" and disclaims any keyword/volume/rank claim).
Attention dies there; the "what do I fix first" content should be #2.

### P5-2 Number reconciliation -- FAILS: self-contradicting dollar figures on paid artifacts [verified-by-reviewer]
Five money paths disagree because three rounding helpers are used:
`_format_money` (`:4893`, `int(v+0.5)`, half-up) and PDF `_model_money`
(`atlas_brain/deflection_pdf_renderer.py:941`, half-up) vs `_email_money`
(`atlas_brain/content_ops_deflection_delivery.py:1637`, `{:,.0f}`, half-even). Verified
divergence: 40.50 -> **$41** half-up vs **$40** half-even; 13.50 -> **$14** half-up.
Concrete catches on the real fixture (14 repeat tickets: clusters 5/6/3):
- **(a) Ranked cost column ($190) != headline ($189).** The table renders `$68 | $81 | $41`
  (each odd row +$0.50 by half-up: 5->67.50->$68, 3->40.50->$41) summing to **$190**,
  while Support Tax says "about **$189**" (rounds the true sum). Delta grows to ~$3 on
  an 8-row table. A reader who totals the visible column is caught out. `_ranked_opportunity_section:4167` vs `_support_tax_section:4051`.
- **(b) The PDF states the benchmark as "$14," the page says "$13.50."** The data holds
  `13.50` (`_ASSISTED_CONTACT_COST:51`), markdown uses the literal label `"$13.50"`
  (`:52`), but the PDF re-derives via `_model_money(13.50)` = **$14**
  (`deflection_pdf_renderer.py:271,939-945`). The PDF then reads "14 x $14 = $196" against
  its own stated $189, and the true $13.50 never appears on the PDF.
- **(c) One email line shows two prices for the same item.** Real `email.txt`:
  "...3 repeat tickets - **$40** estimated handling ... assisted-contact cost is **$41**...".
  Lead via `_email_money` (half-even -> $40), embedded `product_gap_summary` built with
  `_format_money` (`_action_product_gap_summary:3829`, half-up -> $41). Every odd-count
  item self-contradicts in the email.
- **(d)** Email vs page/PDF diverge by $1 on any .50 total ($148 email vs $149 page/PDF).
- **(e)** `repeat + non_repeat == ticket_source_count` RECONCILES (14+1=15). Negative
  result; but `_non_repeat_ticket_count:4739` adds a summary field to a live singleton
  re-count -- currently non-overlapping, a latent drift path.
- **(f)** annualized `$2,999` = `14 x 365/23 x 13.50` reconciles but is a 15.9x
  extrapolation from a 23-day window (copy hedges "at the same measured daily pace" -- defensible).
  It is a **run-rate**, not a confident annual loss: at n=14 in 23 days the Poisson variance is
  large, so it should be shown with an "if this pace holds" range, not a point figure (D2).

This is the biggest presentation defect: the paid artifacts show numbers a sharp reader
adds up and catches, and there is no runtime guard (P5-7). Fix: one money helper, one
rounding mode, render the benchmark from the single `$13.50` label everywhere.

### P5-3 Evidence grounding -- STRONG (verbatim), one caveat
Every "Complete evidence" block lists real source IDs + verbatim excerpts
(`zd-reset-1 -> "How do I reset my password? ... I am locked out."`); drafted answers
derive from the uploaded `resolution_text` (`_complete_evidence_detail:4351`,
`_source_backing_summary:4855`). Caveats: (1) the representative question can be a
synthesized "What should I do about X?" when no question-shaped row sorts first (Slice 2
F11); (2) `_ticket_text:647` concatenates subject+body, so quotes can read with
duplicated text ("How do I reset my password? How do I reset my password? ...") --
verbatim but awkward.

### P5-4 Blind-spot handling -- HONEST, one watch-item
Unproven clusters render "**No proven answer yet:** No uploaded resolution evidence was
present" (`_no_proven_answer_detail:4319`) -- flagged, not hidden. Watch-item:
`_publishable_answer_detail:4286` (+ PDF `deflection_pdf_renderer.py:582`) substitute the
content-free "This draft answer is backed by uploaded resolution evidence." when a
*proven* item's `answer` body is empty -- a reader could mistake a proven-but-empty item
for a real drafted answer.

### P5-5 Hierarchy of value -- MIXED
Dollar tax leads (good), but: (a) the SEO list outranks the ranked opportunities (P5-1);
(b) **no visual top-5 distinction** -- `_ranked_opportunity_section:4148` emits one flat
rank-ordered table, so with 60+ questions the top rows are not set apart; (c) the
**Priority Fix Queue** (`priority_fix_queue`, `:592-604`, web/pdf/email only) is **absent
from the hosted markdown surface entirely** -- the on-page markdown and the attached PDF
ship different section sets.

### P5-6 Snapshot->report bridge -- PARTIAL: quantifies tickets, gestures on dollars
The free `summary` carries `repeat_ticket_count` but **no `estimated_support_cost` and no
annualized figure** (stripped by the support_tax `snapshot_safe_fields` allowlist,
`:543-551`). `top_questions` carry per-question cost (raw `67.5`), but `locked_questions`
carry **only `{rank, ticket_count}`** -- no dollars, no topic (a 12-cluster run locks 6
ranks = 23 tickets the buyer sees no $ or topic for). There is no aggregate "$X more / N
more clusters" rollup (`build_deflection_snapshot:2683,2778`). The single most persuasive
number -- the ~$2,999/yr annualized tax -- is computed but **withheld from the free
snapshot**, so the upgrade hook is weaker than the report can support.

### P5-7 Scorecard visibility -- MISSED TRUST MECHANISM
`build_deflection_full_report_qa_scorecard:1781` produces an integrity scorecard, but
(a) it has **zero runtime callers**; (b) **no** verification signal is surfaced to the
reader on any surface; (c) even if it ran, it asserts only schema/section/key presence +
count-equality -- **not totals=sum-of-parts**, so it would not catch the $190-vs-$189
defect (P5-2a). Surfacing a "checks passed" signal AND adding a totals-reconciliation
assertion would both build trust and catch P5-2.

## Phase 6 -- Richness gaps

Consumed today by `_normalize_ticket_row:530`: source_id, subject, body, resolution,
public comments, created_at (window only), status (reopened bucket only), csat (negative
count only), group/assignee/tags/product (owner-lane routing only), contact_email, company/vendor.

### List 1 -- Zendesk columns present but IGNORED
(verified: `priority`/`channel` appear nowhere in `support_ticket_input_package.py`)

| # | Ignored column | Signal | Cost | Where |
|---|---|---|---|---|
| 1 | **priority** | Business-criticality of a repeat; a 3-ticket P1 outranks a 6-ticket P4 | LOW | Full report + snapshot badge |
| 2 | **channel** | Origin of the repeat -> phone-heavy needs IVR, web-heavy needs FAQ | LOW | Full report + diagnostics |
| 3 | **public reply/comment count** | Back-and-forth depth = hard-to-resolve; flat $13.50 assumes one contact | MED | Full report -> refine cost model |
| 4 | **requester / repeat-requester rate** | "20 tickets from 3 requesters" vs "from 20" -- account gap vs broad FAQ | MED | Full report per-question |
| 5 | **tags** (routing only) | The team's own taxonomy validates/challenges clustering | LOW | Full report cross-tab |
| 6 | **assignee/group** (owner-lane only) | Which team eats the repeat load | LOW | Owner-lane rollup |
| 7 | **CSAT** (negative count only) | Avg CSAT per cluster + CSAT of answered clusters | LOW | Diagnostics |
| 8 | **status** (reopened count only) | Time-in-status / resolution latency | LOW-MED | Diagnostics |

### List 2 -- Computable from already-parsed data but never computed
1. **Week-over-week trend per cluster** -- `created_at` is parsed (window only,
   `_items_source_date_window:4790`); bucketing by week yields "reset-password up 3x this
   month". **Biggest missed signal; dates already in hand.** MED. Full report + delta path.
2. **Reopen *rate* per cluster** (reopened/total) -- only the absolute is shown. LOW. Diagnostics.
3. **First-contact-resolution proxy** (clusters closed with <=1 agent reply). MED. Full report.
4. **Cost-per-cluster over time** (per-cluster batch cost + #1). MED. Full report + delta.
5. **Distinct-requester / channel-mix rollups** (trivial once List 1 #2/#4 aliased). Snapshot/report.

### List 3 -- Genuinely need the Zendesk API (the CSV tier's honest ceiling)
1. **Macro usage** -- proves an answer fired but is not deflecting (Ticket Audits API).
2. **Audit log / status transitions** -- real handle-time to replace the flat $13.50 (Audits API).
3. **Escalation / reassignment transitions** -- the truly expensive multi-team tickets (Audits API).
4. **Full comment threads w/ role + timestamps** -- real reply count + latency (Comments API).
5. **Satisfaction reason/comment** -- CSV has the score, API has the "why" (Satisfaction API).

**Honest ceiling:** the CSV tier can do volume, per-cluster cost at a flat benchmark,
status/CSAT snapshots, and -- if it read the dropped columns -- priority/channel/requester/
trend (Lists 1-2). It cannot measure true handle-time, macro deflection, or escalation
depth; those need the Audit/Comments API and belong in a future connected paid tier.

## Verdict

The output reads more like a trustworthy, evidence-backed report than a flat dump --
claims trace to verbatim source quotes with IDs, blind spots are labeled honestly, the
dollar hook leads. But **its numbers do not cleanly reconcile**: three rounding helpers
(half-up vs half-even) plus the $13.50 benchmark rendered as $14 produce self-contradicting
figures on the paid artifacts ($40 and $41 in one email line, $190 column vs $189 headline),
with no runtime guard because the QA scorecard has zero runtime callers and does not assert
totals-sum-to-parts. Secondary gaps: the free snapshot withholds the annualized-tax number
that would most motivate the upgrade, and the pipeline silently drops priority/channel/
requester/date-trend signals it already parses or could parse cheaply.

*Read-only. No product code was modified. Probe + real output in gitignored `_audit_scratch/`.*
