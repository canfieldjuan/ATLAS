# ATLAS-HARDENING.md

Parked non-blocking hardening discoveries for the **Atlas blog / deep-dive
content pipeline** work (the `content-ops/blog-*` ownership lanes). Kept separate
from the root `HARDENING.md` so this session's parked items don't collide with
the concurrent content-ops-station sessions. Newest entries first.

Same rules as root `HARDENING.md` (AGENTS.md §3d): do NOT park anything that
breaks the slice's real flow, the AGENTS contract, tests, CI, security, or data
truthfulness -- those are fixed inline. When starting a slice, scan this file for
entries touching the same lane/files; fix only what the slice needs, else leave
parked and note in the plan's Deferred.

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

### LLM may still narrate a partial source set despite the generalized field (D5 residual)
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`, blog generation payload + the external `digest/b2b_blog_post_generation` skill (system prompt).
- Description: D5 stops feeding the per-platform `source_distribution.sources` list to the LLM payload and provides a generalized `source_description`, but the model could still infer/mention platform names from quote `source_name`s or chart labels in the payload, or just not use `source_description`. LLM-fidelity is empirically unverified (same class as the D6 lead-in drift).
- Why it matters: the deterministic lever is closed, but corpus-wide adherence needs checking; this is what the Phase-2 deep pass should verify (and a detector could flag prose source-lists that omit a quoted source).
- Effort: M
- Category: correctness
- Found during: D5

### Unused `_top_source_summary` helper after D5 dropped its only caller
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L2461 (and mirror).
- Description: `_build_coverage_snapshot_note` no longer calls `_top_source_summary`; it's now dead code. (It also had a latent bug — it was fed the `{sources, verified_count, community_count}` dict instead of `{name: count}`.) Remove next pass.
- Why it matters: dead code + a latent bug; removing it is cleanup, deliberately out of D5's thin scope.
- Effort: S
- Category: tech-debt
- Found during: D5

### Deep-dive strengths/weaknesses chart fallback cannot show true strengths
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L8092 (and the byte-identical `extracted_content_pipeline` mirror), the `if len(strengths) + len(weaknesses) < 3 and signals:` fallback in `_blueprint_vendor_deep_dive`.
- Description: when the product profile is too thin, the "Strengths vs Weaknesses" chart is built only from pain-category signals, which carry weakness data only -- so the `strengths` series is always 0 (a one-sided chart). D4 made the bucketing truthful (all pain -> weaknesses) but did not give the fallback a real strengths source.
- Why it matters: thin-profile deep-dives render a two-axis chart that can structurally never populate one axis. A real fix pulls strengths from profile.strengths or positive-mention counts so the chart is genuinely two-sided.
- Effort: M
- Category: polish
- Found during: D4

### "Strengths vs Weaknesses" chart title is two-sided while fallback data is one-sided
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L8124 (`chart_id="strengths-weaknesses"` title) and the rendered chart titles in published deep-dive posts.
- Description: the chart title implies a strengths-vs-weaknesses comparison; in the signals fallback the chart shows weaknesses only. Title/content mismatch for thin-profile deep-dives.
- Why it matters: minor reader confusion -- a "vs" framing with one empty side. Pairs with the entry above; resolving the strengths source likely resolves this too.
- Effort: S
- Category: polish
- Found during: D4
