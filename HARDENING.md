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
- Found during:
```

## Parked Items

## 2026-05-22

### Consider a wider advisory-lock hash key
- File/location: `extracted_content_pipeline/api/generated_assets.py`, `_landing_page_repair_lock`
- Description: `hashtext()` is 32-bit, so unrelated draft lock keys could theoretically collide.
- Why it matters: A collision would create a false `409` for an unrelated draft repair.
- Effort: S
- Category: correctness
- Found during: PR-Landing-Page-Repair-Cost-Guard review

### Revisit repair lock connection hold time
- File/location: `extracted_content_pipeline/api/generated_assets.py`, `repair_landing_page_draft`
- Description: The advisory-lock connection stays checked out while the LLM repair runs.
- Why it matters: This is acceptable for operator-triggered repair, but higher repair volume could turn LLM latency into pool pressure.
- Effort: M
- Category: tech-debt
- Found during: PR-Landing-Page-Repair-Cost-Guard review
