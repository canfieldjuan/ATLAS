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

## Confirmed structural rules from external research

The following are documented patterns from successful "strange, dark, and
mysterious" channels (MrBallen et al). Encodable directly into the
generation prompt and into deterministic post-generation gates.

### Five-beat script model with timing

| Beat | Window | Function |
| --- | --- | --- |
| Fast Hook | 0:00 - 0:45 | Cold open. Shocking statement / teaser of worst moment / provocative question. <=3 second attention grab. |
| Character Humanization | 0:45 - 3:00 | Establish "who" and "where" through a personal lens. Make the audience care before tragedy. |
| Contextual Build | 3:00 - 10:00 | Chronological. Omit outcome / "why". Each beat adds pressure or stakes. |
| Escalation | 10:00 - end | Rapid succession to climax. Pacing and vocal speed build tension. No resolution disclosure. |
| Delayed Payoff | resolution | Final reveal + clean wrap addressing remaining questions. |

### Hard invariants (deterministic gates)

- **10% rule:** the outcome / perpetrator identity / final reveal must not
  appear in the first 90% of the script. This is a checkable invariant, not
  a vibe — a `story_evidence_engine.py` gate can auto-reject a draft whose
  outcome-keywords land before the 90% mark.
- **The Prefacing Rule:** any technical or historical setup that cannot be
  woven smoothly into the narrative must be surfaced *before* the Fast Hook.
  Prevents flow-breaking exposition mid-story.

### Prompt-engineering constraints (rhetorical rules)

Drop these directly into the generation prompt:

- "Write from the limited first-person perspective of the victim or a
  witness, not as an omniscient observer."
- "Include specific sensory details — smell of damp earth, distant
  floorboard creaking — to enable active mental simulation in the audience."
- "Do not mention the final outcome or perpetrator identity until the final
  10% of the script."
- "Use ellipses (...) to indicate dramatic pauses and CAPITALIZATION for
  emphasized words to guide the synthetic narrator's delivery."

### TTS vocal cue convention

The integration glue between text generation and ElevenLabs:

| Marker | Meaning to TTS layer |
| --- | --- |
| `...` | Dramatic pause / intentional silence |
| `WORD` (all caps) | Emphasized stress |
| `[whispers]`, `[laughs]`, `[hesitant]` | ElevenLabs v3 alpha audio tags (inline emotional control) |

The generation prompt must instruct the LLM to embed these markers; the
voice render layer then carries the emotion deterministically.

### Voice persona shortlist (verify before use)

Reportedly tuned for mystery / long-form narration. **Verify each name
against the live ElevenLabs voice library before treating as ground truth**
— LLM-generated catalog details are a known hallucination class. Spend
30 minutes in the actual library, listen to samples, confirm IDs.

| Candidate | Reported character | Reported source |
| --- | --- | --- |
| Graham / Rupert | Authoritative British male; "BBC documentary effect" | ElevenLabs |
| Adam Stone | Smooth, deep, late-night-radio | ElevenLabs |
| John Doe | Gravely storyteller, audiobook tone | ElevenLabs |
| David Castlemore | Mystery / thriller storytelling | ElevenLabs |
| Ian Cartwell | "Mystery Storyteller" - tuned for suspense | ElevenLabs |
| Nathaniel | British, calm, "mysterious tone" reflective | ElevenLabs |

### Uncanny-valley regression cases (test against)

Concrete failure modes documented in audience criticism. Useful as
acceptance tests for any voice integration:

- Voice pauses mid-sentence as if hitting a period when it shouldn't
- Monotone delivery over content that should carry emotional weight
  (e.g., upbeat tone narrating a tragedy)
- Generic, unrelated stock footage (e.g., 8-second clip of a car on a
  motorway) used as filler

If a generated video exhibits any of these, retention dies.

### Source corpus

Concrete starting datasets identified by top creators:

- `r/nosleep` (long-form first-person horror)
- `r/creepypasta` (internet-native short horror)
- True-crime case databases

Lesser-known-but-trending stories outperform retold famous cases.

### Visual pacing constants (parked until video pipeline)

Programmable scene-rule constants for the eventual video assembly layer.
Not needed for text-only v0; saved here because these specific numbers
require a stopwatch-on-reference-videos research session to derive.

| Element | Value | Function |
| --- | --- | --- |
| Slow dread scenes | 8.0 - 15.0 sec | High shot length lets fear "breathe" |
| Panic / action scenes | 4.0 - 6.0 sec | Rapid cuts simulate disorientation |
| Zoom-in speed | 1% - 2% | Constant slow motion maintains focus |
| Grain / film flicker overlay | 5% - 20% | Signals "eerie" atmosphere |

### Channel taxonomy (niche-selection menu)

Pick ONE for v0. Prompt engineering is genre-specific; building one
pipeline that writes "any kind of story well" is the wrong framing.

| Channel | Style | Engagement lever | Best fit for |
| --- | --- | --- | --- |
| @MrBallen | First-person, delayed payoff | Emotive narration, ex-SEAL credibility | Narrative immersion, "strange" mysteries |
| @ThatChapter | Witty, research-heavy, Irish narration | Humor + meticulous case breakdowns | Bingeable case reports |
| @BaileySarian | "Murder, Mystery & Makeup" | Levity-darkness juxtaposition | Casual storytelling |
| @ExploreWithUs | Documentary, interrogation-focused | Bodycam + legal procedure | Investigative immersion |
| @LazyMasquerade | Eerie, atmospheric, detached | Internet lore, paranormal | Late-night moody viewing |
| @MrNightmare | List-based horror, direct | Themed series (Walmart, Night Shifts) | Short, intense horror |

### Tonal guardrails

- **Humanise the victim.** Use names, give them personality and life
  context. The most successful channels celebrate the victim rather than
  treating them as subject of a "disturbing" story.
- **Cultural / community accuracy.** Distorted portrayals of rural areas
  or specific cultures invite legal and reputational risk; respectful
  tone is a long-term monetisation requirement, not just an ethical one.

### Business-case unit economics

| Niche | RPM range | Engagement profile |
| --- | --- | --- |
| True Crime | $10 - $20 | High watch time, intense research focus |
| Mystery / Unsolved | $10 - $18 | High repeat viewership, active comments |
| Paranormal / Ghost | $8 - $15 | Loyal subscribers, "creepy" atmosphere |
| Creepypasta | $8 - $12 | Binge-watchable, internet-native audience |

RPMs in this niche run 2-3x platform average due to deep engagement and
long watch times.

## Resume condition

This work resumes only after the campaign-core spine is fully product-owned
per `remaining_productization_audit.md` "Next Concrete Slice" — migrating
producer flow from copied `b2b_campaign_generation.py` into
`CampaignGenerationService` using the normalized ports.
