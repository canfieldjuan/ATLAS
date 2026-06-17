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

## 2026-06-17

### CSV source-row admission accepts machine JSON in mapped message fields
- File/location: `scripts/evaluate_csv_admission_threshold_evidence.py` breakage matrix and `extracted_content_pipeline/ingestion_diagnostics.py` CSV source-row admission.
- Description: A row whose mapped `Message` value is a machine JSON blob such as `{"event":"ticket_created","id":123}` is currently accepted as usable customer text.
- Why it matters: Machine payloads in a mapped body field can inflate usable-row coverage and make parser admission look cleaner than the report input really is.
- Effort: M
- Category: correctness
- Owner/session: Codex content-ops/deflection-parser-testing
- Found during: PR-Deflection-Parser-Breakage-Evidence-Runner

## 2026-06-16

### Rotate archived IndexNow key
- File/location: Historical `atlas-intel-next` / `_ARCHIVED_atlas-intel-next/scripts/indexnow.ts`
- Description: The archived IndexNow script previously contained a real IndexNow key. The branch tip now requires `INDEXNOW_KEY`, but the old key remains in git history and should be rotated or replaced at the IndexNow key-hosting location.
- Why it matters: IndexNow keys are low-sensitivity and semi-public by design, but rotating the historical value closes the loop on a real key surfaced by secret scanning.
- Effort: S
- Category: security
- Owner/session: Codex security/workflow
- Found during: PR-Security-Guardrail-CI review

### Pin remaining mutable workflow action refs
- File/location: `.github/workflows/*.yml`
- Description: The workflow posture audit now reports existing mutable GitHub Action refs as warnings, and `claude.yml` has been owner-gated and SHA-pinned. Existing product/check workflows still use mutable action tags and should be drained through a dedicated pinning or Dependabot-triage slice.
- Why it matters: Mutable action tags can become CI compromise paths if a third-party action is compromised or a tag is repointed.
- Effort: M
- Category: security
- Owner/session: Codex security/workflow
- Found during: PR-Security-Guardrail-CI review; narrowed during PR-Workflow-Action-Pin-OIDC-Audit

### Burn down advisory security scanner backlog
- File/location: `.github/workflows/security_guardrails.yml`, GitHub code scanning results for Semgrep, Trivy, Checkov, pip-audit, and OSV.
- Description: The first adoption run surfaced existing Semgrep findings, Trivy HIGH/CRITICAL config findings, and dependency CVEs. This PR keeps the sweep informative so `main` does not launch permanently red, but the findings still need triage and ratcheting to blocking gates.
- Why it matters: Advisory scans are only useful if the backlog is burned down and the gates are tightened once known debt is fixed or explicitly waived.
- Effort: L
- Category: security
- Owner/session: Codex security/workflow
- Found during: PR-Security-Guardrail-CI review

### Pin or retire floating ASR dependency audit input
- File/location: `requirements.asr.txt`
- Description: `requirements.asr.txt` installs `nemo_toolkit[asr]` from `NVIDIA/NeMo@main`, so pip-audit would resolve a moving upstream dependency graph on every scheduled run. The advisory pip-audit matrix excludes this file entirely, which means the ASR dependency stack has zero CVE coverage until the requirement is pinned to a tag/commit or retired.
- Why it matters: Security scans need deterministic inputs; a floating VCS requirement can fail or change results without any Atlas code change, but excluding it is still a conscious CVE-coverage gap that must be closed.
- Effort: M
- Category: security
- Owner/session: Codex security/workflow
- Found during: PR-Security-Guardrail-CI review

### Rotate credentials exposed in historical `.env`
- File/location: Historical commit `d63a9b77b9727766e14e523626c22dd6c1c80da8`, file `.env`
- Description: Full-history Gitleaks adoption scan found redacted provider credentials in an old committed `.env`, including Stripe, Anthropic, OpenRouter, Reddit, Firecrawl, Stack Overflow, Product Hunt, CAPTCHA, Apollo, SignalWire, Google Calendar, Resend, and Google API-style keys.
- Why it matters: Any real key committed to git history must be treated as exposed even if the file is now ignored or deleted. CI can block new leaks with a baseline, but provider-side rotation/revocation is still required.
- Effort: M
- Category: security
- Owner/session: Codex security/workflow
- Found during: PR-Security-Guardrail-CI

## 2026-06-14

### Support-ticket tokenizer still over-strips some es-ending words
- File/location: `extracted_content_pipeline/support_ticket_clustering.py`
- Description: The tokenizer fix in `PR-Deflection-Question-Label-Quality` stopped `Atlas` and `status` from becoming `atla` and `statu`, but the remaining final-`s` rule can still turn terms such as `series` and `kubernetes` into `serie` and `kubernete`.
- Why it matters: Rare product or technical terms could still degrade cluster previews or source-policy labels if they become the dominant fallback token.
- Effort: S
- Category: polish
- Owner/session: Codex content-ops/deflection-product-proof
- Found during: PR-Deflection-Question-Label-Quality review

## 2026-06-13

### portfolio-ui npm audit vulnerabilities
- File/location: `portfolio-ui/package-lock.json`
- Description: `npm ci` in `portfolio-ui` reports 3 dependency audit findings (1 moderate, 2 high).
- Why it matters: Dependency vulnerabilities can become deploy-time security exposure, but resolving them may require package upgrades outside this repeat-metric copy slice.
- Effort: M
- Category: security
- Owner/session: Codex deflection/clustering
- Found during: PR-Deflection-Repeat-Metric-Alignment

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

### Brand-voice second-person guidance is not consistently honored
- File/location: `extracted_content_pipeline/brand_voice.py` audit surfaced from live blog and sales-brief exports.
- Description: The Gate A profile requested `preferred_pov=second_person`. The exported blog draft and one sales brief had `brand_voice_audit.passed=false` with `preferred_pov_second_person_not_detected`.
- Why it matters: the stored profile reaches the prompt, but live output can still miss the requested voice; the UI should not present brand voice as applied without exposing or repairing these misses.
- Effort: M
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
