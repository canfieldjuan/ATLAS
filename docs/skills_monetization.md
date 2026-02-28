# Behavioral Risk Sensors — Skills, Relationships & Monetization

## What the Sensors Detect

Atlas runs three independent linguistic sensors on any text corpus —
forum posts, negotiation transcripts, press releases, employee reviews,
operational communications — and then cross-correlates their outputs to
find relationships between the signals.

| Sensor | What it measures | Triggered when |
|--------|-----------------|----------------|
| **Alignment** | Collaborative vs adversarial identity language | Adversarial framing dominates (≥30% of combined terms) |
| **Operational Urgency** | Planning-mode vs reactive-mode temporal language | Urgency density ≥ 2× planning density |
| **Negotiation Rigidity** | Absolutist demands vs flexibility markers | Absolutist share ≥50% of combined terms |

---

## The Four Cross-Sensor Relationships

Running sensors in isolation is table stakes. The value comes from what
happens when two or three fire simultaneously on the same body of text.
`correlate()` maps those co-occurrences to four named relationship patterns,
each with a distinct meaning:

### 1. `adversarial_rigidity` — Alignment + Negotiation Rigidity
**What it means:** Identity adversarialism and absolutist positions are
co-active. The parties aren't just using "us vs them" language — they have
also removed all flexibility from their stated positions. Trust and
flexibility collapsed at the same time.

**Why it matters:** This is the earliest reliable indicator that a
negotiation has moved from competitive to combative. Either side can still
walk it back, but the window is narrowing.

---

### 2. `adversarial_reactivity` — Alignment + Operational Urgency
**What it means:** Adversarial framing has escalated into immediate
operational pressure. The divide is no longer theoretical — it is actively
disrupting day-to-day operations.

**Why it matters:** This pattern marks the point where a labor or logistics
dispute transitions from a slow-burn tension into an active operational risk.
Logistics networks, shippers, and insurers need to act before the next
pattern forms.

---

### 3. `reactive_lock` — Operational Urgency + Negotiation Rigidity
**What it means:** Time pressure has spiked precisely as positions have
hardened. There is urgency to resolve the situation, but neither side has
any flexibility left to do so.

**Why it matters:** This is the highest short-term disruption signal. A
situation under time pressure with no available compromise path has one
probable outcome: work stoppage, route failure, or operational shutdown.

---

### 4. `full_friction_cascade` — All Three Sensors
**What it means:** All three friction axes — identity adversarialism,
reactive urgency, and negotiation lock — are simultaneously active and
mutually reinforcing. Each dimension amplifies the others.

**Why it matters:** This is the pre-event cascade signature. Historical
strike and work-stoppage data shows this linguistic fingerprint appearing
in public communications 48–96 hours before the event goes operational.
By the time mainstream media reports it, this pattern has already been
visible in the text for days.

---

## Composite Risk Levels

| Sensors triggered | Risk level | Meaning |
|:-----------------:|:----------:|---------|
| 0 | **LOW** | No friction signals detected |
| 1 | **MEDIUM** | Single axis of friction — monitor, no action |
| 2 | **HIGH** | Cross-sensor relationship active — prepare contingency |
| 3 | **CRITICAL** | Full cascade underway — imminent disruption likely |

---

## Why We Look for These Relationships

Each sensor alone produces noise. A text that says "now" and "urgent" a lot
might just be enthusiastic writing. But when urgency language co-fires with
absolutist negotiation language *in the same body of text*, that is not noise
— that is a structural pattern that precedes operational breakdown.

The relationships answer the question the individual sensors cannot:
**"Is this situation getting worse, or is it isolated friction?"**

The cascade pattern (`full_friction_cascade`) is the most commercially
significant because it provides advance warning that existing financial
models for labor-related disruption risk are not priced correctly.

---

## Monetization Paths

### 1. Risk Intelligence Subscription (B2B SaaS)
**Who buys it:** Freight brokers, 3PLs, ocean carriers, large shippers
(Fortune 500 supply chain teams), risk desks at PE firms with logistics
portfolio companies.

**What they pay for:** A continuous feed of composite risk scores for
monitored carriers, ports, and labor markets — delivered as an API or
dashboard. Alert when any monitored entity crosses from MEDIUM → HIGH or
HIGH → CRITICAL.

**Pricing anchor:** Comparable products (FreightWaves SONAR, DAT iQ,
Resilience360) sell at $15,000–$80,000/year per seat or entity. A
friction-signal layer on top of existing data is a natural upsell at
$500–$2,000/month per monitored company.

---

### 2. Parametric Insurance Trigger (InsurTech)
**Who buys it:** Insurance carriers writing labor disruption, supply chain
interruption, and trade credit policies. Reinsurers building parametric
products.

**What they pay for:** The `full_friction_cascade` pattern as an objective,
auditable trigger for parametric payout or for repricing risk in real time.
Because the signal is linguistic (not a third-party index), it is not
subject to the moral hazard problems of self-reported triggers.

**Pricing anchor:** Data licensing to insurers for use in underwriting
models runs $50,000–$500,000/year depending on exclusivity. A parametric
trigger product earns a fee on each triggered payout event.

---

### 3. Hedge Fund / Quant Research Data Feed (Fintech)
**Who buys it:** Quantitative funds running labor-disruption factor models,
event-driven desks, ESG-overlay funds that need to price labor risk into
equities positions.

**What they pay for:** Ticker-mapped daily scores and relationship-pattern
flags for publicly traded companies, so the `reactive_lock` and
`full_friction_cascade` patterns can be used as alpha signals for put
options, short positions, or position-size reduction ahead of disclosed
labor events.

**Pricing anchor:** Alternative data feeds to quant funds sell at
$25,000–$250,000/year. Event-alpha signals with a documented 48–96-hour
lead time command the upper end of that range.

---

### 4. Consulting & Litigation Support (Professional Services)
**Who buys it:** Management-side labor attorneys, HR consulting firms,
corporate boards navigating active union negotiations, M&A due-diligence
teams assessing acquired workforce risk.

**What they pay for:** A scored linguistic record of a negotiation — showing
when language shifted, which pattern emerged, and how the composite risk
level evolved over time. In litigation, this becomes an expert exhibit. In
M&A, it becomes a workforce-risk line item in the deal memo.

**Pricing anchor:** Expert witness engagements run $5,000–$25,000 per
matter. Consulting retainers for active negotiations run $10,000–$50,000.

---

### 5. Embedded API (Platform / White-Label)
**Who buys it:** Workforce intelligence platforms (Glassdoor, LinkedIn
Talent Insights, Revelio Labs), risk management platforms (Resilience360,
Everstream), ERP vendors adding supply chain risk modules.

**What they pay for:** A white-labeled version of the three sensors plus
`correlate()` embedded in their existing product, billed as a per-call API
or a revenue-share on new SKUs they build on top of it.

**Pricing anchor:** API licensing at $0.001–$0.01 per call; at 10M
calls/month (realistic for an embedded platform partner) that is
$10,000–$100,000/month.

---

## The Data Flywheel

The sensors improve as more labeled text is processed:

```
Text corpus ingested
       │
       ▼
Three sensors score each document
       │
       ▼
correlate() identifies which relationships are active
       │
       ▼
Relationship patterns are matched against known outcomes
(strikes, stoppages, contract failures)
       │
       ▼
Threshold calibration improves — better precision, fewer false positives
       │
       ▼
Higher-confidence signal → higher-value product
```

Every confirmed outcome (e.g., a strike that was preceded by
`full_friction_cascade`) becomes a labeled training point that validates
and sharpens the threshold values. This is the defensible moat: the
historical labeled dataset of pattern → outcome pairs that a new entrant
would need years to replicate.
