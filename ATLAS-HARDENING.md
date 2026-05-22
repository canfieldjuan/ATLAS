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
