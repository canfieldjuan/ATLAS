# Long-Form Creative Content — Parked Backlog

Date: 2026-05-02

This document captures the long-form creative-content product direction
(AI-narrated short stories / serialized fiction / YouTube-style narrative
channels) and the data-flow / writing-quality findings that surfaced during
the discussion. **Nothing in this file is in flight.** It is parked until the
extracted content pipeline reaches 100% standalone operability, per the
remaining work tracked in `remaining_productization_audit.md`.

## Why this is parked

Standalone readiness is currently ~86% (12 manifest-mapped Atlas-shaped files
still in the boundary, see audit doc). Until the campaign-core spine is fully
product-owned and zero copied-from-Atlas surface remains in the sellable path,
no new product direction starts.

## The product hypothesis

Listening to AI-narrated YouTube story channels surfaces two consistent tells:

1. **Voice is monotone** with no fluctuation — narrator-level emotion is flat
   across a 20-minute story.
2. **Prose is descriptive but emotionally even** — even adjective-rich
   sentences land without rhythm or intentional cadence shifts.

If a small team can fix both, they can produce content that out-performs the
current AI-narration pack. The fixes:

- **Voice:** ElevenLabs (v3 alpha supports inline audio tags such as
  `[whispers]`, `[laughs]`, `[hesitant]`). With embedded performance markers in
  the generated text, the voice layer carries the emotion deterministically.
- **Writing:** the moat. Sentence-rhythm + specificity is what produces felt
  emotion. Default LLM output settles into even cadence and hedged language.
  Fixing this is prompt-engineering work driven by examples, not instructions.

## Pre-build research before any code

The single highest-leverage move before writing any new code:

- Pull 30-50 top videos from one tight niche (true-crime narrative,
  Reddit-style relationship drama, sci-fi short fiction, etc.).
- Transcribe each.
- Measure: hook structure, paragraph length distribution, sentence-length
  variance, dialogue ratio, sensory-detail density, cliffhanger placement.
- That dataset becomes the style spec encoded in the generation prompt as
  examples.

Without this dataset, the prompt engineering is guesswork.

**Niche lock-in note:** prompt engineering is genre-specific. Building one
pipeline that writes "any kind of story well" is not the right framing.
Pick a niche, dogfood it end to end, then generalize.

## Today's data-flow (for reference)

When the next slice begins, the swap point is already small. Each existing
generator reads from Postgres tables specific to its domain:

| Generator | Reads from | Where |
| --- | --- | --- |
| Blog posts | `product_reviews`, `complaint_reports`, `blog_posts` | `autonomous/tasks/blog_post_generation.py:78-244, 493-600` |
| Complaint content | `product_pain_points` (pain_score >= 4.0) | `autonomous/tasks/complaint_content_generation.py:135` |
| B2B campaign emails | `b2b_reviews`, `b2b_campaigns`, churn views | `autonomous/tasks/b2b_campaign_generation.py:3816-3891` |
| Vendor briefings | `vendor_targets`, `b2b_intelligence`, evidence vault | `autonomous/tasks/b2b_vendor_briefing.py:3102-3146` |
| **Campaign generation (standalone)** | **JSON/CSV files** | **`campaign_customer_data.py:77-96`** |

`campaign_customer_data.py` is the only file/object-driven loader in the
package and is the right model to copy when retargeting the pipeline for
narrative input.

`EXTRACTED_PIPELINE_STANDALONE=1` skips LLM calls and batch execution but does
not mock DB reads — the existing tasks still expect a live Postgres. Any
narrative loader must be file/object-driven from the start.

## Minimum viable seam (when work resumes)

Approximately 150 LOC across two files, no subsystem rewiring:

1. New `load_narrative_prompts_from_source(...)` next to
   `campaign_customer_data.py:77-96`. Normalizes a narrative input record
   (story idea + character/voice persona + target length + hook style) into
   the dict shape `_gather_data` already returns.
2. One conditional in `blog_post_generation.py:107` (`_gather_data`) for
   `topic_type == "custom_narrative"` that calls the new loader instead of the
   SQL fetchers.
3. New skill markdown in `skills/digest/` (existing prompt registry handles
   loading).

LLM batching, cost tracking, retries, idempotent fingerprints, prompt
caching, and model routing all reuse without modification.

## Reusable as-is (zero changes)

- `pipelines/llm.py`, `services/b2b/anthropic_batch.py`, `campaign_llm_client.py`
- Token-aware compression in `campaign_sequence_context.py:29-80` and
  `reasoning/evidence_engine.py`
- Postgres storage abstraction (`campaign_postgres.py`,
  `storage/database.py`, `storage/models.py`)
- Skill registry (`skills/registry.py`)
- Fingerprint / idempotent retry (`autonomous/tasks/_b2b_batch_utils.py`)

## Domain-locked (will need replacement for stories)

- Recipient/target selection: `services/vendor_target_selection.py`,
  `autonomous/tasks/campaign_suppression.py`
- B2B evidence engine: `reasoning/evidence_engine.py:27-100`,
  `reasoning/archetypes.py`
- Email sequence progression: `campaign_sequence_progression.py:20-60`
- B2B prompt content: `skills/digest/b2b_campaign_generation.md:1-60`,
  `autonomous/tasks/b2b_campaign_generation.py:74-102`
- Email template renderer: `templates/email/vendor_briefing.py` (primitives
  reusable; email shape itself is not)
- Churn-signal-gated generation: `services/campaign_reasoning_context.py`

## Punch list for a 50k-word novel pipeline (parked)

When this work eventually starts, six new files / ~1200 LOC:

1. `skills/digest/novel_chapter_generation.md` — input
   `{chapter_num, outline_section, character_state, world_context, prior_summary, reader_cohort}`,
   output
   `{chapter_markdown, word_count, pacing_score, continuity_issues}`.
2. `storage/story_models.py` — `StoryChapterDraft` plus worldbuilding /
   character-state JSON blobs.
3. `story_sequence_progression.py` — replaces email open/click gates with
   chapter-completeness gates.
4. `reasoning/story_evidence_engine.py` — swap vendor/churn rules for
   `character_consistent`, `timeline_coherent`, `setting_described`.
   Rule-engine shape from `reasoning/evidence_engine.py:27-79` is reusable.
5. `services/story_reasoning_context.py` — mirror
   `campaign_reasoning_context.py`; return compressed worldbuilding + arcs +
   prior-chapter synthesis under the existing 512-token budget.
6. `templates/story/chapter_renderer.py` — chapter / TOC assembly. Reuse
   HTML-escape and section-break primitives from
   `templates/email/vendor_briefing.py:9-24`.

Plus `story_generation.py` mirroring `campaign_generation.py:68-150` for
orchestration.

## Voice integration (parked)

ElevenLabs v3 alpha supports inline audio tags. The integration point is at
content-render time (post-generation, pre-publish): the generation prompt is
trained to embed `[whispers]`, `[laughs]`, `[hesitant]`, etc. inline; the
voice layer reads them and carries the emotion. This is not a separate
service; it is a render-time pass over generated text.

## Resume condition

This work resumes only after the campaign-core spine is fully product-owned
per `remaining_productization_audit.md` "Next Concrete Slice" — migrating
producer flow from copied `b2b_campaign_generation.py` into
`CampaignGenerationService` using the normalized ports.
