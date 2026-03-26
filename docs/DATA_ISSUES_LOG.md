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

## Issue 7: Unlabeled Ranked Lists — Data Without Context

**Discovered:** 2026-03-26
**Report Type:** Displacement Report
**Status:** Open

### Description

Ranked lists in reports render as bare numbers without labels (e.g., `#2: 66`, `#3: 61`). The underlying data contains the entity names, but the rendering step only outputs the rank position and count — stripping the most important piece: *what* is being ranked. This is a sibling of Issues 3 and 6 (rendering/template gaps), but distinct because the data exists and is correct; it's purely a presentation failure.

### Downstream Impact

- **Entire report sections become useless.** A top-50 list with no labels is dead space — readers can't act on it, reference it, or even understand it.
- **Wastes the strongest signal.** Displacement flow rankings are among the highest-value outputs the pipeline produces. Rendering them without names throws away the payoff.
- **Suggests the serialization layer is dropping fields.** If the LLM or template receives a list of `{rank, name, count}` objects but only prints `{rank, count}`, there may be a field-mapping or context-window issue at play.

### Potential Investigation Directions

- Check whether the ranked data passed to the LLM/template includes entity names or only IDs that require a separate lookup.
- If names are present in the data, audit the prompt/template to ensure the name field is referenced in the output format.
- If names are missing, trace back to the query layer to confirm the join to the vendor registry is happening.

---

## Issue 8: Competitor Names Extracted as Feature Drivers — Misclassification Opportunity

**Discovered:** 2026-03-26
**Report Type:** Displacement Report
**Status:** Open — Flagged as Potential Feature

### Description

The driver extraction pipeline sometimes surfaces competitor names as displacement drivers (e.g., "Flexibility and depth comparable to Tableau" classified as a 6% driver). This happens because reviewers use well-known products as shorthand for feature expectations — "I want Tableau-like dashboards" — and the extraction treats the entire phrase as a driver string.

### Downstream Impact

- **Not a bug in isolation, but needs classification.** This is actually valuable signal — it reveals which products serve as the *mental benchmark* in a category. Knowing that users measure alternatives against a specific product is strategic intelligence.
- **Without classification, it muddies driver analysis.** If competitor-as-benchmark signals are mixed in with true feature drivers (pricing, UX, integrations), aggregate driver breakdowns become noisy. "Tableau" isn't a *feature* — it's a *reference point*.
- **Opportunity for a new signal type.** Separating "benchmark mentions" from "feature drivers" would create a distinct intelligence layer: which products define the category standard in users' minds.

### Potential Investigation Directions

- Add a classification step in driver extraction that detects when a driver string contains a known vendor/product name and tags it as "benchmark reference" vs. "feature driver."
- Surface benchmark references as their own report section (e.g., "Category Benchmarks — products users compare against").
- Cross-reference with the vendor registry to auto-detect product names in driver strings.

---

## Issue 9: Zeroed-Out Metrics — Boolean/Null Parsing Failure

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open

### Description

Computed metrics (e.g., "Recommend Ratio") render as `0` for every vendor in the scorecard, regardless of review volume. A vendor with 500+ reviews cannot have a 0% recommendation rate — this points to a parsing failure where a boolean, null, or missing field is being coerced to `0` instead of being calculated or flagged as unavailable.

### Downstream Impact

- **Immediate credibility killer.** The Vendor Scorecard is positioned for C-Suite and Strategic Partnerships — the most scrutinizing audience. A column of zeros will be spotted instantly and discredit the entire report.
- **Silent data corruption pattern.** If one metric is zeroed out, others may be too — but less obviously. Metrics that *should* be low might appear correct by coincidence while actually being wrong.
- **Breaks comparative analysis.** The scorecard's core value is vendor-vs-vendor comparison. If a key metric is uniformly zero, that comparison dimension is completely eliminated.

### Potential Investigation Directions

- Trace the "recommend" field from source ingestion → extraction → aggregation → report rendering to find where the value drops to 0 or null.
- Check whether the source data uses a non-standard representation (e.g., "Yes"/"No" string instead of boolean, or a nested field the extractor isn't reaching).
- Add validation that flags any metric that is identical across all vendors in a scorecard — statistically impossible uniformity is a reliable bug signal.

---

## Issue 10: Uncategorized Pain Points — The "Other" Bucket Problem

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open

### Description

The pain-point extraction pipeline produces "Other" as a top category for some vendors. In a scorecard designed for strategic decision-makers, "Other" is a wasted slot — it means the LLM could not confidently classify the complaint into an existing taxonomy and gave up. This is especially damaging when "Other" is the *#1 pain point*, as it implies the pipeline doesn't understand the most common complaint.

### Downstream Impact

- **Wastes the most valuable real estate in the report.** The top pain point slot in a scorecard drives strategic narratives. Filling it with "Other" communicates nothing.
- **Hides actionable intelligence.** The underlying reviews that got bucketed into "Other" likely *do* contain classifiable complaints — technical debt, integration complexity, contract lock-in, poor support — but the extraction gave up too early.
- **Skews competitive comparison.** If Vendor A's top pain is "Pricing" and Vendor B's is "Other," the comparison is meaningless. Vendor B looks better by default because its real pain is hidden.

### Potential Investigation Directions

- Force the classification model to choose from a defined taxonomy rather than allowing an "Other" escape hatch. If confidence is low, use a secondary classification pass or expand the taxonomy.
- Post-process: when "Other" exceeds a threshold (e.g., >10% of complaints for a vendor), re-run classification on those reviews with a more granular prompt.
- Audit what's actually *in* the "Other" bucket — it may reveal taxonomy gaps (categories the pipeline doesn't have yet).

---

## Issue 11: Emerging Displacement Signals in Unexpected Segments

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open — Flagged as Potential Feature

### Description

The scorecard surfaces low-volume but strategically significant displacement signals in unexpected product segments (e.g., a community/collaboration tool threatening an enterprise meeting platform — not the expected enterprise competitors). These signals are currently buried alongside high-volume flows without differentiation.

### Downstream Impact

- **High-value signal hidden in plain sight.** Unexpected displacement patterns (a tool from an adjacent category encroaching) are often the earliest indicators of market disruption. They deserve amplification, not burial.
- **Segment misattribution.** The pipeline treats all displacement equally — it doesn't distinguish "enterprise competitors swapping" from "category boundary erosion." These are fundamentally different strategic signals.
- **Missed narrative opportunity.** A scorecard that says "your competition isn't just who you think it is" is far more valuable to a C-Suite audience than one that confirms known rivalries.

### Potential Investigation Directions

- Tag displacement flows with a "same-category" vs. "cross-category" flag based on vendor registry metadata.
- Surface cross-category displacements in a dedicated "Emerging Threats" or "Category Boundary Shifts" section.
- Weight low-volume cross-category signals higher in strategic reports, as they indicate early-stage disruption.

---

*New issues will be appended below as they are discovered.*
