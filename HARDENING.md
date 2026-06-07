# HARDENING.md

Park non-blocking hardening discoveries here when they are not required for the
current thin slice to function. Newest entries go first.

Do not use this file to defer issues that break the slice's real flow, AGENTS
contract, tests, CI, security, or data truthfulness. Those must be fixed inline
or the slice must stop.

When starting a slice, scan this file for entries touching the same ownership
lane or files. Fix only entries required for that slice to function; otherwise
leave them parked and mention the reason in the plan's Deferred section if they
were considered. Periodically drain stale entries or promote them into the debt
register under `docs/technical-debt/`.

## Entry Format

```md
## YYYY-MM-DD

### <short title>
- File/location:
- Description:
- Why it matters:
- Effort: S / M / L
- Category: correctness / polish / tech-debt / security
- Owner/session:
- Found during:
```

## Parked Items

## 2026-06-07

### Landing-page variants pass audits but are not meaningfully distinct
- File/location: `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-landing_page.json` and landing-page generation variant prompting.
- Description: Gate A exported three landing-page variants with distinct hero headlines, but the same title and largely similar body copy; one page repeats the same "35 out of 36 tickets were question-shaped" point in adjacent paragraphs.
- Why it matters: Audit-clean variants are not necessarily marketable variants. Review/export needs to prove meaningful whole-page differences, not only different hero text.
- Effort: M
- Category: correctness
- Owner/session: Codex Gate A live output-quality proof
- Found during: PR-Gate-A-Live-Output-Quality-Proof review

### Blog output uses debug-style source narration instead of publishable prose
- File/location: `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-blog_post.json` and blog generation prompt/quality gates.
- Description: The approved blog row opens with "The uploaded CSV contains 36 support-ticket rows...", which reads like internal data narration rather than a publishable customer-facing article.
- Why it matters: A generated blog can pass structural checks while still being unshippable marketing content.
- Effort: M
- Category: correctness
- Owner/session: Codex Gate A live output-quality proof
- Found during: PR-Gate-A-Live-Output-Quality-Proof review

### Gate A needs a messy-ticket grounding rerun
- File/location: Gate A validation harness input selection / future validation artifact.
- Description: The committed Gate A proof used the clean `support_ticket_saas_demo_sources.csv` fixture where nine clusters each contain exactly four tickets. That proves the live path, but does not stress grounding on lopsided clusters, junk rows, missing fields, or noisy real tickets.
- Why it matters: Grounding quality is easiest on the tidy demo fixture; the next acceptance run should use messy real support data before treating Gate A as product-cleared.
- Effort: M
- Category: correctness
- Owner/session: Codex Gate A live output-quality proof
- Found during: PR-Gate-A-Live-Output-Quality-Proof review

### Brand-voice second-person guidance is not consistently honored
- File/location: `extracted_content_pipeline/brand_voice.py` audit surfaced from live blog and sales-brief exports.
- Description: The Gate A profile requested `preferred_pov=second_person`. The exported blog draft and one sales brief had `brand_voice_audit.passed=false` with `preferred_pov_second_person_not_detected`.
- Why it matters: the stored profile reaches the prompt, but live output can still miss the requested voice; the UI should not present brand voice as applied without exposing or repairing these misses.
- Effort: M
- Category: correctness
- Owner/session: Codex Gate A live output-quality proof
- Found during: PR-Gate-A-Live-Output-Quality-Proof

### Sales brief live generation drifts from requested renewal brief type
- File/location: `extracted_content_pipeline/sales_brief_generation.py`
- Description: Gate A requested `inputs.brief_type=renewal`; all three exported sales briefs stored `brief_type=pre_call` because the model's parsed JSON won over the configured default.
- Why it matters: review/export proves persistence, but the output contract can ignore the operator's requested sales-brief mode.
- Effort: S
- Category: correctness
- Owner/session: Codex Gate A live output-quality proof
- Found during: PR-Gate-A-Live-Output-Quality-Proof

## 2026-05-29

### atlas-intel-ui npm audit vulnerabilities
- File/location: `atlas-intel-ui/package-lock.json`
- Description: `npm ci` reports 6 dependency audit findings (2 moderate, 4 high).
- Why it matters: Dependency vulnerabilities can become deploy-time security exposure, but resolving them may require package upgrades outside this UI rendering slice.
- Effort: M
- Category: security
- Owner/session: Codex FAQ deflection report UI slice
- Found during: PR-FAQ-Deflection-Report-UI-Readonly

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
