# Data Issues Log

Tracking data collection, extraction, and enrichment issues discovered during report audits.
Each issue is framed holistically — focusing on systemic pipeline impact, not individual vendors.

---

## Issue 1: Single-Source Entity Resolution — The "Unnamed Signal" Problem

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports can surface hundreds of evaluation signals but resolve only a tiny fraction to named companies. The extraction pipeline identifies company names only when a reviewer explicitly self-attributes (e.g., "I work at X and we're leaving Y"). The vast majority of signals remain anonymous.

### Downstream Impact

- **Signal density is high, but actionable intelligence is low.** A report may show 71 active churn signals yet name only 1 company — making the report feel thin despite strong underlying data.
- **Battle cards and competitive briefings lack specificity.** Sales teams need to know *who* is moving, not just *that someone is moving*.
- **Trend analysis is weakened.** Without entity resolution, we can't detect patterns like "mid-market SaaS companies are all leaving Vendor X" — the cohort stays invisible.

### Potential Investigation Directions

- Cross-reference Source IDs (HackerNews/Reddit links) with external identity signals (LinkedIn activity, company tech-stack databases, IP-to-Company resolution).
- Enrich at ingestion time rather than report-generation time to avoid repeated re-processing.

---

## Issue 2: Contradictory Narrative Detection — Aggressive Pricing Nuance

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports can label a vendor as a "clear loser" in a competitive landscape while simultaneously surfacing evidence that the same vendor is winning deals back through aggressive discounting (e.g., 90% off). The pipeline does not flag or reconcile these contradictions — both statements appear in the same report without commentary.

### Downstream Impact

- **Misleading competitive positioning.** A battle card that says "Vendor X is losing" while ignoring their fire-sale pricing strategy gives sales reps a false sense of security.
- **Lost strategic insight.** Aggressive discounting is a *signal in itself* — it indicates desperation pricing, budget capture plays, or land-and-expand tactics. Burying it as a footnote wastes high-value intelligence.
- **Report credibility suffers.** Readers who spot the contradiction will question the entire analysis.

### Potential Investigation Directions

- Add a contradiction-detection pass in report generation (or post-processing) that flags when a vendor is simultaneously categorized as "losing" and "winning deals."
- Surface pricing strategy signals as a distinct section in battle cards (e.g., "Competitive Pricing Alerts").

---

## Issue 3: Variable Bloat / Raw Identifiers in Report Output

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports contain raw technical artifacts — numbered section markers like `#1, 13`, `#2, 12`, and full Source ID strings — that leak through to the final output. This makes reports look like database exports rather than polished intelligence briefings.

### Downstream Impact

- **Degrades report readability.** End users (sales reps, strategists) see noise that erodes trust in the product.
- **Signals a template/rendering gap.** The report generation layer is not fully abstracting the underlying data model from the presentation layer.
- **Increases support burden.** Users will ask "what does #1, 13 mean?" — generating unnecessary back-and-forth.

### Potential Investigation Directions

- Audit the report template / LLM prompt to ensure raw IDs and internal markers are excluded or mapped to human-readable labels before rendering.
- Add a post-processing sanitization step that strips or transforms technical identifiers.

---

## Issue 4: De-Anonymization Ceiling — Enrichment Fails on Vague Identifiers

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

The enrichment layer cannot resolve vague organizational descriptors like "Fortune 200 company" or "Big 4 Firm" to specific entities. The intent signal is present in the source data (Reddit, Trustpilot, G2), but the pipeline has no mechanism to map these fuzzy labels to actual companies. This is a variation of Issue 1 (entity resolution), but distinct — here the source *does* provide a partial identifier, and the pipeline still can't close the gap.

### Downstream Impact

- **Near-miss intelligence.** The data is tantalizingly close to actionable — a "Fortune 200" evaluating a switch is valuable, but without knowing *which one*, sales can't act on it.
- **Segment-level analysis is blocked.** We can't aggregate signals by company size, vertical, or account tier if the entities behind vague labels remain unresolved.
- **Report feels speculative.** Citing anonymous archetypes ("a Big 4 firm") reads as hearsay rather than evidence-backed intelligence.

### Potential Investigation Directions

- Build a fuzzy-match enrichment step that cross-references vague descriptors with known tech-stack databases, job postings, or procurement signals to narrow candidates.
- Flag "partially identified" signals separately so reports can at least segment by confidence tier (named → partially identified → anonymous).

---

## Issue 5: Contradictory Signals Between Report Sections — Narrative Coherence Gap

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

Different sections of the same report can assert contradictory facts without reconciliation. For example, a Target Accounts section may show "0 companies considering Vendor Y," while the Head-to-Head section says "Vendor Y is displacing Vendor X." This creates a narrative gap — the displacement signal exists but isn't reflected in the pipeline counts. This is a broadening of Issue 2 (contradiction detection), now observed across *structural sections* of a report, not just within a single narrative.

### Downstream Impact

- **Erodes trust in the report as a whole.** If two sections disagree, readers don't know which to believe.
- **Signals a data flow disconnect.** The displacement graph and the target-account pipeline are likely pulling from different query scopes or time windows, producing inconsistent views of the same reality.
- **Missed early-stage signals.** A displacement trend emerging in reviews may not yet appear in structured "consideration" data — but the report should explain that gap, not ignore it.

### Potential Investigation Directions

- Audit whether displacement signals and target-account counts query the same underlying data or different snapshots/time ranges.
- Add a coherence-check pass that cross-references section assertions and flags contradictions or explains the discrepancy (e.g., "Displacement signal is emerging but no accounts have formally entered evaluation stage yet").

---

## Issue 6: Template Variable Leakage — Code Strings in Report Output

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

Raw template variable names (e.g., `displacement_detail.primary_driver`) are appearing verbatim in rendered report text. This indicates the LLM prompt or report template has unresolved placeholders that are not being substituted before output — the code is "leaking" into the final deliverable. This is related to Issue 3 (variable bloat) but more severe: Issue 3 involved raw IDs/markers, while this involves actual code-level variable paths.

### Downstream Impact

- **Immediate credibility hit.** A report containing `displacement_detail.primary_driver` looks broken, not professional.
- **Indicates fragile template logic.** If one variable leaks, others may be silently failing too — meaning some report sections could be rendering with missing or default data without anyone noticing.
- **Compounds across report types.** If the same template engine serves multiple report types, the leakage pattern likely affects more than just challenger briefs.

### Potential Investigation Directions

- Audit all report templates/prompts for unresolved variable references and ensure fallback/default handling exists for every placeholder.
- Add a post-render validation step that rejects or flags output containing common variable patterns (e.g., anything matching `*.primary_driver`, `*.detail.*`, etc.).

---

*New issues will be appended below as they are discovered.*
