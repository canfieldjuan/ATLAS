# Podcast Repurposing Landing Page Strategy — Demo-First Productized Offer

Date: 2026-05-03 (updated)

This document is the buyer-facing strategy for the podcast repurposing
offer that uses the AI Content Ops system as its engine.

**Status:** the engineering required to honour the page promise is now
shipped. The product can ingest a transcript, extract the strongest
ideas, and generate per-format drafts (newsletter, blog, LinkedIn post,
X thread, shorts script). See `host_install_runbook.md` for the host
install path and the new `run_extracted_podcast_*.py` CLIs at the
repository root.

**Buyer-facing language convention:** never reference "AI Content Ops,"
"the pipeline," or any internal machinery. The buyer sees a service that
turns one podcast episode into a week of usable content. The engine stays
under the hood.

## Page type and length

**Short-to-medium productized service page. Not full long-form VSL.**

The buyer can understand the offer instantly: "I give you one episode.
You give me multiple usable content assets." That is simple. The page
does not need to educate them for 3,000 words.

The demo / output samples carry the sale. Page length is a conversion
trade-off; long-form is overkill at this price point and adds friction.

Build the page modularly so the lower sections can be expanded into
long-form later if a higher-priced tier emerges. Do not pre-commit to
long-form before the price point proves it needs it.

## Above the fold

**Headline:**
> Turn One Podcast Episode Into a Week of Ready-to-Post Content

**Subhead:**
> Paste your episode link and get publish-ready newsletters, blog posts,
> LinkedIn posts, X threads, and short-form scripts — written from the
> actual episode and matched to your voice.

**Primary CTA:**
> Repurpose My First Episode

**Secondary CTA:**
> See Sample Outputs

**Immediately below the CTAs**, a sample transformation rendered as
visual cards or tabs:

```
Input:
Podcast Episode: "Why Most Businesses Struggle to Use AI"

Outputs:
Newsletter · Blog Post · LinkedIn Post · X Thread · Shorts Script
```

The visual transformation IS the demo. Click-to-expand each output card
and show the actual generated asset.

## Below the fold (tight; no padding)

### 1. Problem

> Your best podcast ideas are trapped inside long-form episodes.
>
> You record the episode, publish it, maybe post the link once, and move
> on. But inside that episode are newsletter ideas, blog posts, social
> clips, quote posts, and short-form scripts your audience would actually
> consume.

### 2. Deliverables

Every episode can become:

- Email newsletter
- SEO blog post
- LinkedIn post or carousel outline
- X thread
- Shorts / Reels / TikTok script
- Pull quotes
- Show notes
- Promo captions

### 3. Demo / sample outputs (the most important section)

Show 2-3 example episodes across different niches with a niche picker:

- Business podcast
- Coaching / personal brand podcast
- Health / wellness podcast
- Real estate / finance podcast
- Faith / mindset podcast

Let users click their category and see sample assets in *that* niche.
This is the podcast version of the "niche picker" UX pattern: massive
personalization lift, low engineering cost.

### 4. Voice consistency

Reframe the quality system as a buyer-facing benefit:

> **Your ideas. Your tone. Not generic AI content.**
>
> We don't just summarize your episode. We pull out the best arguments,
> stories, hooks, and teaching moments — then shape them into content
> that still sounds like you.

### 5. How it works (four steps, no machinery talk)

1. Paste your episode link
2. We extract the strongest ideas
3. We turn them into platform-ready assets
4. You review, edit, and publish

**No pipeline diagram. No reasoning-layer talk. No "AI Content Ops"
language.**

### 6. Pricing

This offer is closer to a productized service than self-serve SaaS at
first. Tier structure:

| Tier | Price | Includes |
| --- | --- | --- |
| **First Episode** | $149 | 1 episode, 5 repurposed assets, 2-3 business day delivery — low-risk paid sample |
| **Monthly Repurposing** | $597/mo | Up to 4 episodes/month, 5 assets per episode (newsletter, blog post, LinkedIn post, X thread, short-form script), voice-matched formatting |
| **Authority Package** | $997/mo | 4-6 episodes/month, 7-10 assets per episode, content calendar, pull quotes, show notes, promo captions, priority turnaround |

## The four-question test

The first version of the page should answer only four questions:

1. What do I give you?
2. What do I get back?
3. Will it sound like me?
4. How much does it cost?

If a section on the page does not answer one of these four questions,
it should be cut from v1.

## Central positioning

Use this as the through-line:

> **Your podcast is not one piece of content. It's the raw material for
> your entire content engine.**

This is the bridge from "I record one episode a week" to "I have a full
content system" without making the buyer care about the machinery
underneath.

## What carries the sale

- **The demo**, not the copy. Output samples decide the conversion.
- **Voice consistency proof.** The single biggest objection in this
  category is "it'll sound like generic AI." Pre-empt this by showing
  side-by-side: episode quote → repurposed asset that uses the host's
  actual phrasing and arguments.
- **Niche-specific samples.** Generic samples convert worse than
  niche-matched ones because the buyer needs to see content for their
  audience, not someone else's.

## Copy moves

- **Lead with the asset, not the speed.** "Get a publish-ready
  newsletter from your last episode" beats "AI repurposing in 90 seconds."
- **Show, don't tell, on quality.** Any hand-written copy on the page
  must not sound like AI, since the product's whole promise is
  non-generic content. Hand-write or edit aggressively.
- **The first paid tier ($149) is a foot-in-the-door, not a profit
  driver.** Conversion to monthly is the metric that matters; price
  the sample to remove friction.

## Engineering status (was: prerequisites)

The two engineering prerequisites originally flagged for this offer are
both addressed in v0:

1. **Voice-match quality** — the format-repurpose skill accepts a
   `voice_anchors` payload (tone descriptors, banned phrases, style
   examples) which the prompt threads through to the generation call. A
   deterministic per-format quality validator
   (`services/podcast_quality.py`) blocks placeholder tokens, banned
   phrases, and per-format structural violations before drafts persist.
   The validator runs after every LLM call; failures are saved with
   `status='needs_review'` for human triage rather than silently
   discarded.
2. **Multi-format output with deterministic per-format templates** —
   the format-repurpose skill encodes per-format structural rules
   (newsletter / blog / linkedin / x_thread / shorts) inline and the
   generator loops one LLM call per format. Per-format `max_tokens`
   budgets are configured in `PodcastRepurposeConfig`. Format set is
   locked to the five from this strategy doc by a CHECK constraint on
   `podcast_format_drafts.format_type`.

Voice fine-tuning (embedding-based style transfer, persona-conditioned
generation) remains a v1.5+ deferral. Hosts that need stronger voice
matching today can ship style examples in `voice_anchors.style_examples`
and override the format-repurpose skill via the `--skills-root` CLI flag
without forking the package.

## Build artifacts shipped

For reviewers and operators:

- Migrations: `storage/migrations/270_podcast_transcripts.sql`,
  `271_podcast_extracted_ideas.sql`, `272_podcast_format_drafts.sql`.
- Library: `podcast_ports.py`, `podcast_transcript_data.py`,
  `podcast_postgres_import.py`, `podcast_extraction.py`,
  `podcast_postgres.py`, `podcast_postgres_extraction.py`,
  `podcast_idea_data.py` (BYO seam),
  `podcast_repurpose_generation.py`, `podcast_postgres_repurpose.py`,
  `podcast_example.py` (offline deterministic LLM),
  `services/podcast_quality.py`.
- Skills: `skills/digest/podcast_idea_extraction.md`,
  `skills/digest/podcast_format_repurpose.md`.
- CLIs: `scripts/run_extracted_podcast_transcript_import.py`,
  `scripts/run_extracted_podcast_idea_extraction.py`,
  `scripts/run_extracted_podcast_repurpose_generation.py`.
- Tests: `tests/test_extracted_podcast_transcript_import.py`,
  `tests/test_extracted_podcast_extraction.py`,
  `tests/test_extracted_podcast_repurpose_generation.py`,
  `tests/test_extracted_podcast_quality.py` (43 tests).
