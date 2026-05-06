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

Every claim that lands in a script traces back to a source row:

```
claim_id          uuid
story_id          uuid
text              the assertion as it appears in the script (post-rewrite)
source_id         fk -> sources
quote             verbatim excerpt from the source
locator           page / paragraph / timestamp / url-fragment
confidence        verified | inferred | disputed | unknown
mode_constraint   strict | narrative | dramatized
inserted_by       extraction_pass | drafter | revision | reviewer
verified_by       null | reviewer_id
verified_at       null | timestamp
```

A script cannot pass the fact-check gate unless every claim in it has
a row. In Strict Documentary mode, every claim must have
`confidence='verified'`. In Narrative Nonfiction, `inferred` is
allowed but is rendered with a "based on reporting from..." inline
tag. In Dramatized, reconstructed segments are linked to the
underlying source event but are flagged separately.

The ledger also gates the **10% rule** (outcome cannot appear in the
first 90% of the script) at the claim level: the reveal-claim is
tagged at extraction time, and the deterministic gate from the
sibling doc's `story_evidence_engine.py` checks claim placement, not
keyword placement.

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

For the **first** evidence-to-story slice, narrow to:

- **One niche:** strange-dark-mysterious / true-crime YouTube,
  matching the sibling doc.
- **One mode:** Narrative Nonfiction. Strict and Dramatized land
  later.
- **Two source loaders:** YouTube transcripts + news article URLs.
  The other 7 listed loaders defer.
- **CLI-driven workflow:** stages 1–13 as scripts; stage 14 (audio
  render) is a separate manual step. UI lands later.
- **Single-pass drafting** (matches the sibling doc's pragmatic v0
  for short-form scripts). Multi-pass orchestration arrives only when
  novel-length or 20-minute documentary lengths are in scope.

Resume condition is unchanged: the campaign-core spine must be
product-owned per `remaining_productization_audit.md` before any of
this starts.

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
