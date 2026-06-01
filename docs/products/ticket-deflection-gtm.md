# Ticket Deflection - Go-To-Market Source-of-Truth

Date: 2026-05-23
Archived context: this note was preserved from a local worktree cleanup. Treat
it as historical GTM input, not current implementation truth. Current capability
truth lives in the codebase, merged plan docs, and PR history.

Status: Draft / pre-campaign. Capability + proof + assets are repo-derived facts;
positioning, pricing, channel, and the 30-day goal are operator decisions.

This doc answers a GTM funnel questionnaire from what the Atlas repo actually
contains, so a campaign (landing page, emails, lead magnet, offer, targeting)
can be built on facts instead of hand-waving. Every claim is tagged:

- `[REPO FACT]` - verifiable in the codebase/docs today.
- `[REPO+NOTES - confirm]` - repo signal + prior productization notes suggest a
  default; operator should confirm.
- `[YOUR CALL]` - a business decision the code cannot make.

## How much the repo answers

Of the 13 funnel fields: ~5 answered cold, ~5 partial (confirm), ~3 pure
business calls. The repo nails *what you sell, the mechanism, the proof, and the
assets* (facts about working code). It cannot decide *price, channel, or the
30-day goal*.

## Critical clarification: two ticket-fed products

The repo has two products fed by support tickets. The funnel language fits the
first; the second is adjacent and shares ingestion.

- (A) **Ticket Deflection -> FAQ** (`extracted_content_pipeline/ticket_faq_markdown.py`):
  turn repeat tickets into self-serve FAQ/help content so customers stop
  emailing support. **Built and validated at 50k rows.** This doc is about (A).
- (B) **Churn / at-risk-account prediction** (the B2B reasoning engine): rank
  accounts by churn risk from ticket signals. **Planned, tabled mid-validation**,
  but it has a more-developed GTM kit (ICP, outreach template, discovery script)
  worth reusing if the offer pivots toward retention.

## Filled funnel template

```text
1. What I sell:
   [REPO FACT] A done-for-you / audit service that ingests support tickets and
   generates self-serve FAQ + help-center content (plus a landing page and blog
   post from the same data) to deflect repeat tickets. Closest funnel options:
   "Ticket analysis/audit" + "Support automation" (NOT a chatbot).
   Evidence: extracted_content_pipeline/ticket_faq_markdown.py,
   support_ticket_input_package.py (outputs: faq_markdown, landing_page, blog_post).

2. Is it built yet or still an idea:
   [REPO FACT] The generator is built and validated. NOT built: any ticketing
   API integration (today it takes CSV/JSON/JSONL uploads only). Auth + Stripe
   billing scaffolding exists; tenant-scoping + a real connector are the gaps.

3. Ideal customer:
   [REPO+NOTES - confirm] Repo examples + proof imply mid-market orgs with high
   repeat-ticket volume and a help center. Prior churn-product ICP was
   "B2B SaaS, $5M-$100M ARR, VP Customer Success" - transferable but that's the
   churn angle. For deflection, "high ticket volume" matters more than industry.
   -> Lock the segment.

4. Company size:
   [REPO+NOTES - confirm] Mid-market is the repo-implied default (example data:
   HubSpot/LegacyCRM users). No formal doc. -> Confirm.

5. Industry/niche:
   [REPO+NOTES - confirm] Pipeline is industry-agnostic (generic ticket schema).
   Proof ran on CFPB (financial complaints); examples are logistics + food.
   Works anywhere; the niche is a strategic choice, not a code constraint.

6. Tools/platforms I know or target:
   [REPO FACT] TODAY: any platform's exported tickets (CSV/JSON/JSONL), manual.
   No Zendesk/Gorgias/Intercom API client exists (vendor_registry has Zendesk/
   HubSpot as name-normalization aliases only, not integrations).
   [NOTES] A Zendesk-first adapter is the planned first connector (~5 dev-days),
   then Intercom -> Freshdesk. Do not claim live integrations yet.

7. Best problem I can solve:
   [REPO FACT] Concrete mechanism (not hand-waving): ingest tickets -> cluster
   repeat questions by pain_category -> score by failure-risk + opportunity ->
   extract customer-worded questions -> generate article-style self-serve FAQ
   with next steps. = "Reducing repetitive tickets" + "Improving help-center
   articles." Funnel-ready phrasing:
   "We analyze your last 90 days of tickets, find the top repeat issues, and
   generate self-serve FAQ content that deflects avoidable tickets."

8. Proof/results I have:
   [REPO FACT] Technical proof, not client outcomes (be honest about this):
   - 50,000 real CFPB ticket rows -> bounded FAQ, all scale gates passed,
     1:41.86 wall, 593MB RSS
     (docs/extraction/validation/content_ops_faq_50k_gated_validation_2026-05-23.md)
   - 1,000-row run + a full generate->export->review->update DB lifecycle run
   - A "Support Ticket Deflection" demo with a live DB backend
   NO before/after client numbers yet -> position the first offer as a
   low-friction audit/pilot.

9. Offer I want to sell:
   [REPO+NOTES - confirm] The built path = Option A (Audit) or B (Audit + sprint).
   The code's own default CTA is literally an audit: "Upload Ticket CSV --
   Free Analysis" -> /systems/ai-content-ops/intake. -> Choose A vs B.

10. Price range:
    [YOUR CALL] No committed pricing in the repo. Prior notes have bands for
    OTHER products (Amazon $99-299/mo; B2B churn $2-20k/mo) - do not reuse
    blindly. The funnel's "free audit -> $2,500-5,000 sprint" fits the built path.

11. Lead channel I want to start with:
    [YOUR CALL] The repo gives the assets: it generates landing pages, blog
    posts, and B2B campaigns from ticket data, plus an email MCP server + CRM.
    Prior notes favor cold email + a validation-call kit (reusable outreach
    template + discovery script already written).

12. Assets I already have:
    [REPO FACT - the richest box] FAQ generator (built+validated), landing-page
    generator, blog-post generator, B2B campaign generator, the "Upload Ticket
    CSV" intake CTA + /systems/ai-content-ops/intake route, the live-DB
    deflection demo, email MCP + CRM + invoicing, auth + Stripe billing
    scaffolding, the atlas-intel-ui dashboard, the 50k validation doc (proof),
    and a written outreach template + 30-min discovery script (prior notes).

13. Goal for the next 30 days:
    [YOUR CALL] Repo+notes make these fastest: "validate the offer" (a tabled
    5-prospect-call kit is ready), OR "build a lead magnet" (the free ticket
    audit IS the lead magnet), OR "create a landing page" (the generator does it).
```

## Three honest gaps (so the campaign does not overclaim)

1. **No live integrations.** Upload-based today. A Zendesk connector is ~5
   dev-days, not done. The audit offer should say "export your tickets / send a
   CSV," not "connect Zendesk."
2. **Proof is technical, not outcome.** You can show it runs on 50k real rows
   and produces clean FAQ; you cannot yet say "cut tickets 30% for client X."
   Lead with a free audit that produces *their* FAQ as the proof.
3. **Pricing / channel / goal are not in the code.** Operator decisions.
   Everything upstream of them, the repo already answers.

## Key evidence paths

- Generator: `extracted_content_pipeline/ticket_faq_markdown.py`
- Input adapter: `extracted_content_pipeline/support_ticket_input_package.py`,
  `support_ticket_input_provider.py` (outputs faq_markdown / landing_page / blog_post)
- Examples: `extracted_content_pipeline/examples/support_ticket_sources.csv`,
  `support_ticket_bundle.json`
- 50k proof: `docs/extraction/validation/content_ops_faq_50k_gated_validation_2026-05-23.md`
- Scale-gate harness: `scripts/smoke_content_ops_faq_scale_run.py`
- Operational risk parked: `HARDENING.md` (FAQSCALE-1 - large synchronous runs
  need hosted limits/backpressure/background execution before exposing large
  uploads as synchronous customer requests)
- Adjacent product status: `CLAUDE.md` (Content Ops = active iteration)

## Suggested next step

Lock fields 3/4/5 (ICP), 9 (offer A vs B), and 10 (price). Once those are set,
the pipeline can generate the actual landing-page copy and the audit lead magnet
from the same ticket data, and the prior outreach template + discovery script
can seed the cold-email sequence.
