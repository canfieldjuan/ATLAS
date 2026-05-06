# Evidence-to-Story Engine — Parked Design (Nonfiction Track)

Date: 2026-05-06

This document is the **nonfiction sibling** of
[`long_form_creative_backlog.md`](long_form_creative_backlog.md). Same parked
status, same genre lock-in, same resume condition. The only divergence is
truth handling: where the creative-backlog doc plans for fiction (or
fiction-styled storytelling inspired by true crime), this doc plans for
**evidence-traceable nonfiction** — same true-crime / mystery / "strange,
dark, and mysterious" YouTube long-form narration niche, but with every
claim sourced and every dramatization labeled.

> **Parked.** Resume condition is the same as the sibling doc: the
> `extracted_content_pipeline` campaign-core spine must be fully
> product-owned (`remaining_productization_audit.md` "Next Concrete
> Slice") before any new product direction starts.

## v0 cut (locked)

The minimum trusted loop that proves the engine works. Locked
2026-05-06. Anything outside this list is explicitly out of scope
for v0.

| Decision | v0 setting |
| --- | --- |
| Niche | true-crime / mystery / "strange, dark, and mysterious" YouTube long-form narration |
| Truth mode | **Narrative Nonfiction only.** Strict Documentary and Dramatized True Story land later. |
| Script length | **5–7 minutes (~1,000–1,500 words).** Shorter than the sibling doc's 10-minute target — less surface area for claim drift in the first loop. |
| Loaders | **Two: YouTube transcripts + news article URL/text.** All other loaders (court records, police reports, OCR, voice notes, handwritten notes, scraped DBs, user timelines) are deferred. |
| Drafting | Single-pass. Multi-pass orchestration loop lands when a length or accuracy regression demands it. |
| Audio render | **Out of scope for v0.** Stage 14 (ElevenLabs render) is a separate manual step. The v0 deliverable is an audio-*ready* script with embedded TTS markers and a voice-direction sidecar. |
| Surface | CLI only. Web UI / Slack-bot approval gates land later. |
| Inferred claims | **Soft-rewrite policy (see "Inferred-claim handling" below).** Internally tag, externally rewrite to hedged language. |

**v0 input → output contract:**

```
INPUT:
  - 1 YouTube transcript (URL or .vtt/.srt)
  - 1 news article (URL or pasted text) about the same case

OUTPUT:
  - source records         (one per ingested artifact)
  - claim ledger           (typed + sourced, see schema below)
  - timeline               (ordered events with confidence bands)
  - 3 story angles         (with evidence-strength scores)
  - selected outline       (5-beat, claim-budget per beat)
  - narration script       (5-7 min, embedded TTS markers, soft-rewritten inferences)
  - citation validation    (every script claim resolves to a ledger row)
  - voice-direction sidecar (per-section tone / pacing / emphasis)
```

This is the contract the v0 build must satisfy. Anything else is a
follow-on PR.

## Why a sibling doc

The parked creative backlog is silent on four design surfaces that become
load-bearing the moment we publish a story about a real person:

1. **Truth handling** — no Strict Documentary / Narrative Nonfiction /
   Dramatized True Story mode separation, no labeling convention.
2. **Citations and evidence preservation** — `story_evidence_engine.py`
   in the punch list checks fiction continuity
   (`character_consistent` / `timeline_coherent` / `setting_described`),
   not source-claim-confidence-quote-verification.
3. **Source-data variety** — covers `r/nosleep`, `r/creepypasta`, true-
   crime case databases. Silent on PDFs, court records, interview
   transcripts, news articles, voice notes.
4. **Guided workflow with approval gates** — describes a pipeline,
   not a user-facing flow with stage-by-stage human approval.

These four are the difference between "AI writes a story" and
"evidence-backed nonfiction storytelling, source-traceable, narration-
ready." The sibling preserves everything else from the creative backlog:
genre lock-in, orchestration loop, prompt rules, voice-tag convention,
unit economics.

## What stays unchanged from the creative backlog

Lift directly without modification:

- **Genre:** true-crime / mystery / "strange, dark, and mysterious"
  YouTube long-form narration. Channel taxonomy menu in the sibling
  doc still applies.
- **v0 output shape:** 10-minute YouTube mystery scripts, 2–3k words,
  single-pass-where-possible.
- **Five-beat script model:** Fast Hook → Character Humanization →
  Contextual Build → Escalation → Delayed Payoff.
- **Hard invariants:** the 10% rule, the Prefacing Rule.
- **Prompt-engineering constraints:** first-person limited POV,
  specific sensory details, ellipses-for-pauses, CAPS-for-emphasis.
- **TTS vocal cue convention:** `...` for pauses, ALLCAPS for stress,
  ElevenLabs v3 inline `[whispers] [laughs] [hesitant]` tags.
- **Voice persona shortlist** (with the same "verify against the live
  ElevenLabs library before treating as ground truth" caveat).
- **Tonal guardrails:** humanize the victim, cultural/community
  accuracy.
- **Unit economics:** $10–$20 RPM in true crime, 2–3× platform average
  due to deep engagement and long watch times.
- **Multi-pass orchestration scope:** outline planner → chapter
  planner → drafter → continuity checker → revision pass → long-context
  memory compression. 12–15 new files / ~3,000–4,500 LOC. Model-mix
  cost discipline.

## What this doc adds

### 1. Truth modes (three, labeled)

Every story is generated under exactly one mode. The mode is part of
the story record, surfaced on the published artifact, and gates which
prompts and validators run.

| Mode | Allowed | Required | Use case |
| --- | --- | --- | --- |
| **Strict Documentary** | Only sourced facts. No invented dialogue. No reconstructed scenes. | Every claim cites a source. | Court-adjacent cases, named living subjects, sensitive cultural ground. |
| **Narrative Nonfiction** | Real facts. Pacing, transitions, emotional framing shaped for narrative. | Every fact cites a source. Framing/transitions are unsourced but flagged in the source map. | Default mode for v0. Most YouTube true-crime content lives here. |
| **Dramatized True Story** | Reconstructed dialogue and scene-level color allowed when the underlying event is sourced. | Any reconstructed scene is labeled "[reconstructed]" in script and "[reconstructed for narrative]" on-screen at render time. | Cases where the source record is thin and the listener experience requires interior dialogue. |

The mode is stored on `StoryDraft` and consumed by:

- The drafter prompt (different system prompts per mode)
- The fact-check stage (different invariants per mode)
- The render layer (mode-specific watermarks / labels)

### 2. Citation ledger (the missing primitive)

Every claim that lands in a script traces back to a source row.
Without typing, the ledger becomes a soup of "everything is a claim"
and the fact-check gate cannot be specific. Hence: every claim is
both confidence-banded *and* type-classified.

#### Claim-type taxonomy (eight types)

`reveal` is a first-class type so the **10% rule** becomes a
deterministic claim-placement gate (not a keyword scan).
`reconstructed` is included because Dramatized True Story is a
planned future mode; it is **excluded from v0** (see the v0 build
contract).

| Type | What it asserts | Example | Default confidence allowed in Narrative Nonfiction |
| --- | --- | --- | --- |
| `factual` | Verifiable real-world assertion | "The car was found near the river." | verified |
| `timeline` | Time / sequence assertion | "The 911 call came in at 8:43 PM." | verified |
| `entity` | A person, place, or organization assertion | "John Doe lived in Chicago." | verified |
| `emotional_inference` | Internal state attributed to a real subject | "She felt isolated." | inferred (must be soft-rewritten — see below) |
| `disputed` | A claim where sources disagree | "The defense says the meeting never happened." | disputed |
| `reveal` | The story's final reveal / outcome / perpetrator identity | "Police arrested the neighbor." | verified (placement-gated by the 10% rule) |
| `transition` | Author voice / scene-setting / pacing — not a factual assertion | "Hours passed before anyone noticed." | unknown (does not require a source row, but is logged) |
| `reconstructed` | Dialogue or scene-level color invented to render a real event | "He whispered, 'I'm coming home.'" | only allowed in Dramatized True Story mode (post-v0) |

The fact-check stage runs different invariants per type. `factual`,
`timeline`, `entity`, and `reveal` claims must have a verbatim quote
in the ledger or be rejected. `emotional_inference` claims must have
a soft-rewrite applied (see next section). `disputed` claims must
cite at least two sources and surface the disagreement on-screen.
`reveal` claims also fire a placement gate so the outcome cannot
appear in the first 90% of the script. `transition` claims are
logged for placement gates but do not require a source quote.
`reconstructed` claims fail the gate in Strict and Narrative modes
and are labeled at render time in Dramatized.

#### Schema

```
claim_id          uuid
story_id          uuid
text              the assertion as it appears in the script (post-rewrite)
claim_type        factual | timeline | entity | emotional_inference
                  | disputed | reveal | transition | reconstructed
source_id         fk -> sources (nullable for `transition` only)
quote             verbatim excerpt from the source (nullable for `transition`)
source_locator:                — structured (no more single-string locators)
  url             string?
  paragraph       int?         — 0-indexed paragraph in the source text
  timestamp       string?      — ISO-8601 duration into a transcript ("00:14:23")
  quote_offset    int?         — 0-indexed char offset within the source text
confidence        verified | inferred | disputed | unknown
mode_constraint   strict | narrative | dramatized
rewrite_applied   bool — true when the inferred-claim soft-rewrite ran
original_text     null | the pre-rewrite assertion (audit trail)
inserted_by       extraction_pass | drafter | revision | reviewer
verified_by       null | reviewer_id
verified_at       null | timestamp
```

A script cannot pass the fact-check gate unless every claim of type
`factual | timeline | entity | disputed | reveal | reconstructed`
has a row, and every `emotional_inference` claim has
`rewrite_applied=true`.

In Strict Documentary mode, every claim must have
`confidence='verified'`. In Narrative Nonfiction, `inferred` is
allowed only on `emotional_inference` claims, and only after the
soft-rewrite. In Dramatized, `reconstructed` segments are linked to
the underlying source event but flagged separately for the render
layer.

The ledger also gates the **10% rule** (outcome cannot appear in the
first 90% of the script) at the claim level: the `reveal` claim type
is tagged at extraction time, and the deterministic gate from the
sibling doc's `story_evidence_engine.py` checks claim placement, not
keyword placement.

#### Inferred-claim handling (soft-rewrite policy)

For Narrative Nonfiction (the v0 mode), inferred emotional content
is allowed internally but must be rewritten before it lands in the
script. The policy is **Option B: internal tag, external rewrite to
hedged language.** This avoids both extremes:

- Hard-rejecting inferred claims would gut the narrative — half the
  appeal of true-crime narration is the felt-emotion layer.
- Letting inferred claims through verbatim ("She was terrified") would
  let the system make nonfiction *sound* good while quietly inventing
  interior states.

Soft-rewrite examples:

| Original (rejected) | Soft-rewritten (allowed) |
| --- | --- |
| "She was terrified." | "Based on the reporting, the situation appears to have left her afraid and isolated." |
| "He knew what he had done." | "Investigators later concluded he understood the consequences of his actions." |
| "The neighbors didn't care." | "No neighbor came forward to police in the days that followed." |

The drafter prompt embeds the soft-rewrite rule. The revision pass
verifies that no `emotional_inference` claim landed in the script
without `rewrite_applied=true`. The `original_text` column preserves
the pre-rewrite assertion for audit.

### 3. Source-data variety

Beyond the sibling doc's `r/nosleep` / `r/creepypasta` / case-database
sources, ingest paths for:

| Source type | Loader | Notes |
| --- | --- | --- |
| News articles | URL or PDF → reader-mode extraction | Strip ads/nav before extraction |
| Court records | PDF → OCR → structured extraction | Preserve case number, jurisdiction, date |
| Interview transcripts | Plain text or `.srt` / `.vtt` | Preserve speaker labels and timestamps |
| YouTube transcripts | URL → existing transcript service | Channel + episode metadata as locator |
| Scraped case databases | Existing scraping stack (`atlas_brain/services/scraping/`, 23 modules + parsers/) | Reuse, do not rebuild |
| Police reports | PDF → OCR → structured | Mark sensitive fields; redact before publishing |
| Voice notes | Audio → existing ASR (`asr_server.py`) → text | Preserve recording timestamp |
| Handwritten notes | Image → OCR | Confidence is `inferred` until reviewer verifies |
| User-supplied timelines | CSV / form input | Maps directly to `claim_id` rows |

Loader contract: every loader emits `(text, source_metadata,
locators[])`. The metadata is normalized by a single shared shaper
(mirrors `campaign_customer_data.py:77-96`'s file/object-driven shape
from the sibling doc) and lands as a `Source` row. Per-claim locators
are written to the citation ledger by the extraction stage.

### 4. Guided workflow with approval gates

Pipeline runs as 14 stages, each with an approval boundary the user
can pause at. The system never auto-publishes:

```
1.  Upload data           → Source rows persisted.
2.  Extract facts         → Claim rows + locators populated.
3.  Build timeline        → Ordered event list with confidence bands.
4.  Identify entities     → People, places, organizations, with aliases.
5.  Detect story angle    → 3 candidate angles with evidence strength.
6.  User selects angle    → Approval gate.
7.  Generate outline      → Five-beat structure + claim-budget per beat.
8.  User approves outline → Approval gate.
9.  Draft script          → First-person limited POV, embedded TTS markers.
10. Fact-check claims     → Every script claim resolves to ledger row;
                            mode-specific invariants enforced.
11. Continuity check      → Reuses sibling doc's continuity checker.
12. Revision pass         → Drafter rewrites against fact-check + continuity findings.
13. User approves script  → Approval gate.
14. Render audio          → ElevenLabs v3 with inline tag interpretation.
```

Stages 1–5 are deterministic / extractive. Stages 7, 9, 12 are LLM
generations (multi-pass orchestration loop from the sibling doc).
Stages 10–11 are deterministic gates with LLM advisory.

### 5. State expansion (on top of the sibling doc)

The sibling doc covers Story state and Generation state. The
nonfiction track adds:

- **Evidence state** — claim ledger above; tracks `verified` /
  `inferred` / `disputed` / `unknown`.
- **Voice state** — narration arc per script section (target tone,
  pacing, breath timing, emphasis points). Stored as a sidecar
  alongside the script, consumed by the render layer.
- **User state** — preferred mode, preferred channel-taxonomy
  archetype, past angle choices. Used to bias angle proposals.
- **Production state** — explicit `draft → fact-check → revised →
  approved → rendered → published` state machine. The current
  parked backlog implies this; the nonfiction track requires it
  because of the legal-review handoff.

### 6. Orchestration adds (on top of the sibling doc)

The sibling doc's orchestration loop covers planning, drafting,
continuity, revision, long-context memory. The nonfiction track adds:

- **Extraction agent** (stage 2) — pulls claims from raw source text
  with strict-output schema. Smaller, faster model.
- **Angle proposer** (stage 5) — three candidate angles with evidence
  strength. Smaller model, structured output.
- **Fact-check validator** (stage 10) — deterministic resolver
  against the citation ledger. No LLM in the hot path; LLM is advisory
  for ambiguous resolutions.
- **Reconstruction labeler** (Dramatized mode only) — flags
  reconstructed dialogue / scenes for the render layer.
- **Citation linter** (cross-cutting) — every script claim must
  resolve to a ledger row before the revision pass exits.

Model-mix discipline carries over: smaller models for extraction /
angle proposal / continuity / fact-check resolution; larger model
only for drafting and revision.

### 7. Endgame framing

Same as the sibling doc: **YouTube long-form narration in the
true-crime / mystery niche**, monetized via ad revenue at the $10–$20
RPM band. The nonfiction track adds:

- **Defensibility through accuracy.** Channels that misrepresent real
  cases face takedowns, demonetization, and legal exposure. Source-
  traceable nonfiction is a long-term moat against the AI-narration
  pack, not just a quality lever.
- **Reuse path for adjacent products.** Same engine, different output
  surface: documentary scripts, podcast episodes, court-style case
  reports, Substack-style longform. None of those are v0; flagging
  them so the citation-ledger schema is shaped to support them.

## Honest scope

The sibling doc estimates **12–15 new files / ~3,000–4,500 LOC** for
the multi-pass orchestration loop alone. The nonfiction adds:

| Component | New work |
| --- | --- |
| Source-loader shaper + per-source loaders | ~6 loaders × ~100 LOC + shared shaper |
| Citation ledger (schema + writer + linter) | ~400 LOC |
| Truth-mode prompt variants (3 modes) | prompt-engineering effort, not LOC |
| Fact-check validator + reconstruction labeler | ~600 LOC |
| Angle proposer + extraction agent | ~500 LOC |
| Approval-gate UI surface | net-new; out of scope for v0 (CLI-driven for v0) |

Realistic combined estimate when both docs land: **18–22 new files /
~5,000–6,500 LOC**, plus prompt-engineering effort that scales with
the number of agents.

## v0 cut

See the **v0 cut (locked)** section pinned near the top of this doc
for the binding scope and the input → output contract. The
implementation-ready spec lives in
[`evidence_to_story_v0_build_contract.md`](evidence_to_story_v0_build_contract.md),
with a fixture scaffold under
[`fixtures/evidence_to_story_v0_golden/`](../fixtures/evidence_to_story_v0_golden/).
Resume condition is unchanged: the campaign-core spine must be
product-owned per `remaining_productization_audit.md` before any of
this starts.

## Name shortlist (parked)

PR #308 didn't claim a product name and the doc shouldn't either
until v0 ships. Captured here so the conversation isn't lost.

| Candidate | What it leans into | Risk |
| --- | --- | --- |
| **StoryLedger** | The citation ledger IS the product. Most descriptive of the actual moat. | Sounds like an accounting tool. |
| **TraceScript** | Catchier; "trace" implies source-traceability. | Doesn't say "story" out loud. |
| **ProofNarrative** | Says the quiet part out loud (sourced narrative). | Clunky as a brand. |
| **SourceFrame** | "Frame" implies the narrative scaffolding around sources. | Generic; collides with photography / framing terms. |

Decision rule when the time comes: pick the name that survives the
"explain it to a true-crime YouTube creator in one sentence" test.
StoryLedger and TraceScript both pass; the other two need a second
sentence.

## Open questions (for the resume conversation)

These are the calls to make when this work resumes — not now:

1. **Citation ledger storage.** Postgres table per story vs single
   shared table with `story_id` foreign key. Likely the latter; flag
   for confirmation.
2. **OCR provider.** Tesseract local vs cloud OCR. Cost vs accuracy
   tradeoff for court-records ingestion.
3. **Sensitive-data redaction.** Names of victims of unsolved cases,
   minor children, sealed records. Default-deny vs opt-in surfaces.
4. **Approval-gate UX.** Web UI vs CLI vs Slack-bot. v0 = CLI;
   production-grade = web UI. Slack-bot is the cheap middle ground.
5. **Mode escalation rules.** When does a Narrative Nonfiction draft
   require auto-promotion to Strict Documentary? (Probably: when a
   named living subject crosses a threshold of mentions, or when a
   claim is tagged `disputed`.)

## Cross-references

- Sibling doc with the genre / orchestration / voice / unit-economics
  scaffolding: [`long_form_creative_backlog.md`](long_form_creative_backlog.md)
- Resume gate (campaign-core spine cleanup):
  `remaining_productization_audit.md` "Next Concrete Slice"
- Existing reusable infrastructure (zero changes for v0):
  `pipelines/llm.py`, `services/b2b/anthropic_batch.py`,
  `campaign_llm_client.py`, `campaign_sequence_context.py`,
  `campaign_postgres.py`, `storage/database.py`, `storage/models.py`,
  `skills/registry.py`, `autonomous/tasks/_b2b_batch_utils.py`
