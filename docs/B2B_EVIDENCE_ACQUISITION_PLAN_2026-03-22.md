# B2B Evidence Acquisition Plan

Date: 2026-03-22

## Scope

This audit verifies which scraper sources can actually improve missing evidence for:

- segment reasoning
- temporal reasoning
- displacement reasoning
- account reasoning

The plan below is based on:

- canonical source registry in `atlas_brain/services/scraping/sources.py`
- source-fit policy in `atlas_brain/services/scraping/source_fit.py`
- parser field mappings in `atlas_brain/services/scraping/parsers/*`
- live Postgres coverage on port `5433` from `b2b_reviews`

## Verified Source Inventory

Canonical source registry contains 18 sources:

- g2
- capterra
- trustradius
- gartner
- peerspot
- getapp
- producthunt
- trustpilot
- reddit
- hackernews
- github
- youtube
- stackoverflow
- quora
- twitter
- rss
- software_advice
- sourceforge

Sources with live rows in `b2b_reviews`:

- reddit: 24,981
- trustpilot: 5,715
- capterra: 1,996
- software_advice: 1,088
- g2: 714
- gartner: 653
- trustradius: 433
- peerspot: 386
- sourceforge: 320
- hackernews: 142
- quora: 137
- twitter: 105

Sources with zero current live rows:

- getapp
- producthunt
- github
- youtube
- stackoverflow
- rss

## Missing Evidence, Precisely

The reasoning pools are no longer blocked by wiring. They are blocked by thin upstream evidence in these fields:

- reviewer identity: title, role, department, decision authority
- company context: company name, company size, industry
- concrete timing: renewal, contract end, evaluation deadline
- pricing context: price increase, spend, seat count
- explicit displacement: alternatives considered, moved-to vendor, migration status
- account proof: company-linked high-intent reviews with quotes

## Verified Live Coverage By Source

Percentages below are from current live `b2b_reviews`, not stale history.

### Segment Fields

- Gartner is strongest for segment identity.
  - title 100.0%
  - company size 100.0%
  - role known 98.9%
  - department known 81.9%
  - buying stage known 54.2%
- TrustRadius is strongest for company-linked segment evidence.
  - title 51.3%
  - company 47.8%
  - company size 71.6%
  - role known 73.9%
  - department known 47.6%
  - company-linked rows 207 of 433
- PeerSpot is strong on role and stage, weaker on parser-captured identity.
  - role known 82.9%
  - buying stage known 57.3%
  - title 0.0%
  - company 4.9%
  - company size 0.0%
- G2 is underperforming relative to parser intent.
  - title 17.4%
  - company 0.4%
  - company size 18.9%
  - role known 34.0%
- Capterra and Software Advice are severely underperforming.
  - Capterra: title 0.0%, company 0.0%, company size 0.0%
  - Software Advice: title 0.0%, company 0.7%, company size 0.0%

### Temporal Fields

- Reddit is the largest timing-signal source by far.
  - active evaluation 6,631 rows
  - concrete timing rows 473
  - support escalation 5.2%
- Trustpilot contributes meaningful support and pricing pain, but weak identity.
  - support escalation 15.5%
  - renewal signal 3.6%
  - concrete timing rows 233
- Hacker News is high-yield for active evaluation and competitor motion, but weak for identity.
  - active evaluation 26.1%
  - competitors 47.2%
- Concrete deadline and renewal evidence is weak across the corpus.
  - deadline signal rate is below 2% for every active source

### Displacement Fields

- Reddit is strongest by absolute competitive volume.
  - competitor rows 8,592
- TrustRadius is strongest by structured competitive rate.
  - competitors in 58.4% of rows
- PeerSpot is also strong for structured alternatives.
  - competitors in 51.6% of rows
- Hacker News is useful for technical replacement chatter.
  - competitors in 47.2% of rows

### Accounts Fields

- TrustRadius is the best current source for company-linked account proof.
  - company-linked 207 rows
  - strategic role rows 177
- Gartner is strongest for strategic-role identification but weak on company linkage.
  - strategic role rows 307
  - company-linked 2 rows
- Reddit has scale but low precision for account linkage.
  - company-linked 600 rows out of 24,981
  - strategic role rows 2,303
- Trustpilot is useful for pain volume, not account precision.
  - company-linked 59 rows out of 5,715

## Parser Reality Check

These are verified by reading the parser implementations directly.

### Sources That Intentionally Lack Company Context

These parsers mostly write `reviewer_company` and `company_size_raw` as `None`, so more scraping volume will not solve segment/account gaps:

- reddit
- hackernews
- twitter
- quora
- trustpilot

These sources are best treated as:

- temporal signal sources
- displacement signal sources
- narrative quote sources

### Sources That Can Provide Identity But Are Underperforming

These parsers explicitly attempt to extract title, company, and size:

- g2
- capterra
- software_advice
- getapp
- trustradius
- gartner
- peerspot

Verified gap:

- TrustRadius, Gartner, and PeerSpot are converting a meaningful share of that structure.
- G2 is converting some of it, but not enough.
- Capterra and Software Advice are barely converting any of it despite parser support.
- GetApp parser is capable, but there are zero live rows because the scraper is currently blocked.

This means:

- for G2, Capterra, Software Advice: parser quality and page extraction are a bigger blocker than raw source availability
- for GetApp: scraper reliability is the blocker

## Ranked Acquisition Plan

### Priority 1: GetApp

Why:

- parser can extract title, company, size, and industry
- source-fit marks it core for crm/support/marketing, project/collaboration, ecommerce, hr, finance, and general B2B
- current live corpus has zero rows, so this is pure missed coverage

Best evidence it can improve:

- segment
- accounts
- pricing
- structured pain quotes

Action:

- restore stable scraping for GetApp first
- then backfill vendors in crm/support/marketing, project/collaboration, ecommerce, hr, and finance

### Priority 2: TrustRadius

Why:

- best current blend of structured identity and competitive context
- strong company linkage
- strong alternatives coverage

Best evidence it can improve:

- segment
- accounts
- displacement

Action:

- increase vendor coverage and page depth where available
- prioritize vendors currently stuck with broad segment labels like `end users`

### Priority 3: Gartner

Why:

- best source for title, size, role, and department
- strongest segment-quality source in the live corpus

Best evidence it can improve:

- segment
- strategic-role targeting

Limit:

- weak company linkage, so it helps less with named accounts

Action:

- expand only where accessible and cost-effective
- use it to sharpen segment playbooks, not account naming

### Priority 4: PeerSpot

Why:

- strong role and stage coverage
- strong alternative/displacement signal
- source-fit marks it core for cloud/devops/security and data/analytics

Best evidence it can improve:

- cloud/security segment reasoning
- cloud/security displacement reasoning

Action:

- expand in cloud, security, infrastructure, observability, and analytics categories

### Priority 5: Reddit

Why:

- highest volume source
- strongest absolute active-evaluation volume
- strongest absolute competitor and quote volume

Best evidence it can improve:

- temporal
- displacement
- quote inventory

Limit:

- identity and company context are intrinsically weak

Action:

- do not rely on Reddit to solve segment/account gaps
- use query tuning to target:
  - renewal
  - price increase
  - outage
  - support escalation
  - switching and migration

### Priority 6: Trustpilot

Why:

- large live corpus
- strong support-escalation rate
- meaningful pricing pain volume

Best evidence it can improve:

- temporal pain triggers
- pricing pressure
- customer-pain quotes

Limit:

- weak reviewer identity and company context by design

Action:

- keep using it for pain and pricing narratives
- do not treat it as a primary segment/account source

## Fix Before Scaling

These are not "scrape more" problems first.

### G2

Verified state:

- parser tries to extract title, company, size, and industry
- live coverage is still weak

Conclusion:

- improve parser/card extraction quality before scaling page depth

### Capterra

Verified state:

- parser tries to extract title, company, size, and industry
- live coverage is effectively zero for those fields

Conclusion:

- this is a parser or extraction-path quality problem, not a source-availability problem

### Software Advice

Verified state:

- parser tries to extract title, company, size, and industry
- live coverage is still near zero

Conclusion:

- fix extraction quality before spending more scrape budget here

## Low Priority Or Conditional

### Hacker News

Useful for:

- technical migration chatter
- active evaluation
- competitor motion

Not useful for:

- company-linked accounts
- firmographic segment precision

### Quora

Useful for:

- lightweight alternative discovery

Not useful for:

- reliable timing
- company context
- segment precision

### Twitter

Useful for:

- fast-moving complaints and public switching chatter

Not useful for:

- identity
- company context
- contract timing

### SourceForge

Useful for:

- cloud/devtools/open-source categories only

Not useful as a general B2B evidence expansion target.

## What To Do Next

### Acquisition Work

1. Unblock GetApp and backfill target verticals where it is a core source.
2. Expand TrustRadius coverage for vendors with weak segment/account evidence.
3. Expand Gartner and PeerSpot selectively for cloud/security and high-value segment gaps.
4. Keep Reddit and Trustpilot focused on timing, pricing, and displacement, not firmographics.

### Extraction Quality Work

1. Audit and fix G2 field capture.
2. Audit and fix Capterra field capture.
3. Audit and fix Software Advice field capture.

### Query Strategy Work

For social sources, prioritize searches that increase concrete timing and pricing evidence:

- `renewal`
- `contract`
- `price increase`
- `support escalation`
- `outage`
- `migrating from`
- `switching from`
- `alternative to`

## Practical Decision Rule

If the goal is better:

- segment pool: prioritize Gartner, TrustRadius, PeerSpot, then fix G2/Capterra/Software Advice
- temporal pool: prioritize Reddit and Trustpilot, then query-tune for renewal and pricing language
- displacement pool: prioritize Reddit, TrustRadius, PeerSpot, Hacker News
- accounts pool: prioritize TrustRadius first, then GetApp once scraper reliability is restored
