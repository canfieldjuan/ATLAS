# Evidence-to-Story Engine — v0 Build Contract

Date: 2026-05-06

Implementation-ready contract for v0 of the Evidence-to-Story engine.
Pairs with the design doc at
[`evidence_to_story_engine.md`](evidence_to_story_engine.md) and the
parked creative sibling at
[`long_form_creative_backlog.md`](long_form_creative_backlog.md).

> **Parked.** Implementation does not start until the
> `extracted_content_pipeline` campaign-core spine is fully product-owned
> per `remaining_productization_audit.md` "Next Concrete Slice".

**Implementation status:** Stage 1 has started in the separate
`extracted_evidence_to_story` product boundary. The product-owned
`extracted_evidence_to_story/sources.py` loader and
`scripts/build_evidence_to_story_sources.py` CLI implement the deterministic
manifest-to-`sources.json` step only. Stages 2-9 remain unimplemented until
their own slices land.

When v0 work begins, **every section below must be true at completion**.
This doc exists to make sure each new file has a reason to exist before
it is written.

---

## 1. v0 scope

| Decision | v0 setting |
| --- | --- |
| Niche | true-crime / mystery / "strange, dark, and mysterious" YouTube long-form narration |
| Truth mode | **Narrative Nonfiction only** |
| Script length | **5–7 minutes (~1,000–1,500 words)** |
| Loaders | **Two:** YouTube transcript + news article URL/text |
| Drafting | **Single-pass.** No revision loop in v0. |
| Audio render | **Out of scope.** Deliverable is an audio-*ready* script with embedded TTS markers and a voice-direction sidecar. |
| Surface | **CLI only.** Five human approval gates (see §4). |
| Inferred claims | **Soft-rewrite (Option B):** internal tag, external rewrite to hedged language. |
| Output | A `story_package/` directory with 9 files (see §3). |

The minimum trusted loop:

```
1 YouTube transcript + 1 news article
  → source records
  → claim ledger (typed + sourced)
  → timeline + entities
  → 3 angles → 1 selected → 5-beat outline
  → 5–7 min narration script with embedded TTS markers
  → citation validation report
  → voice-direction sidecar
```

---

## 2. Non-goals

Explicitly **not** in v0:

- Strict Documentary mode
- Dramatized True Story mode (and `reconstructed` claim type)
- Multi-pass orchestration (outline → draft → critique → revise)
- Continuity-checker pass (single-pass drafting does not need one)
- Revision pass
- Long-context memory compression (the sibling doc's chapter-level concern)
- Audio render via ElevenLabs
- Web UI / Slack-bot approval gates
- Court records / police reports / OCR / handwritten notes / voice notes
- Scraped case-database loaders (defer; reuse `atlas_brain/services/scraping/` later)
- Auto-publish anywhere
- Multi-niche generalization ("works for any kind of story")
- Multi-narrator / character-voice formats
- User-state persistence (preferences, past projects)
- Web app or API surface — v0 is internal-tool-shaped

If a feature isn't on the v0 scope list and isn't on this non-goal list,
it does not exist for v0 yet. Surface the question in the first PR.

---

## 3. Output package shape

A v0 run produces a directory:

```
story_package/
├── sources.json            — ingested sources, normalized
├── claims.json             — citation ledger (typed + sourced)
├── timeline.json           — ordered events with confidence bands
├── entities.json           — people / places / organizations + aliases
├── angles.json             — 3 candidate angles with evidence-strength scores
├── selected_outline.json   — chosen angle's 5-beat outline + claim budget
├── script.md               — narration-ready 5–7 min script with TTS markers
├── voice_direction.json    — per-section tone / pacing / emphasis sidecar
└── validation_report.json  — pass/fail per acceptance test (see §9)
```

Files are written in the order produced. Stage failures abort the run
and leave a partial package on disk with `validation_report.json`
describing what completed.

---

## 4. Stage contracts

Nine stages. Five approval gates. Every stage has a deterministic
input/output JSON contract.

| # | Stage | Kind | Approval gate after? |
| --- | --- | --- | --- |
| 1 | Load sources | deterministic | ✅ — review `sources.json` |
| 2 | Extract claims | LLM (structured-extraction tier) | ✅ — review `claims.json` |
| 3 | Build timeline | deterministic + LLM advisory | — |
| 4 | Identify entities | LLM (structured-extraction tier) | — |
| 5 | Propose 3 angles | LLM (structured-extraction tier) | ✅ — review `angles.json`, pick one |
| 6 | Generate outline | LLM (writing tier) | ✅ — review `selected_outline.json` |
| 7 | Draft script | LLM (writing tier) | — |
| 8 | Validate citations | deterministic + LLM advisory | ✅ — review `validation_report.json` |
| 9 | Generate voice direction | LLM (structured-extraction tier) | — |

The five gates are CLI-driven: stage exits with a banner pointing at the
file the user should review and the command to resume the next stage.
No web UI. No Slack-bot. No daemon.

### Stage 1 — Load sources

**Input** (CLI flags + a manifest):

```json
{
  "story_id": "case_2026_05_06_a",
  "sources": [
    {"type": "youtube_transcript", "title": "...", "url": "...", "text_path": "inputs/yt_transcript.txt"},
    {"type": "news_article",       "title": "...", "url": "...", "text_path": "inputs/article.txt"}
  ]
}
```

**Output** — `sources.json`:

```json
{
  "story_id": "case_2026_05_06_a",
  "sources": [
    {
      "source_id": "src_yt_01",
      "type": "youtube_transcript",
      "title": "...", "url": "...",
      "text": "...",
      "metadata": {"duration_sec": 1830, "channel": "...", "published_at": "..."}
    },
    {
      "source_id": "src_news_01",
      "type": "news_article",
      "title": "...", "url": "...",
      "text": "...",
      "metadata": {"author": "...", "publication": "...", "published_at": "..."}
    }
  ]
}
```

Loaders implement a single `load(spec) → SourceRecord` Protocol. Two
concrete loaders for v0; everything else fails with
`UnsupportedSourceType`.

### Stage 2 — Extract claims

**Input:** `sources.json`.

**Output** — `claims.json`. See §5 for the schema. A typed, sourced
claim ledger. Every `factual | timeline | entity | disputed | reveal`
claim must carry a `source_id` and a `quote`. `transition` claims have
`source_id=null`. `emotional_inference` claims must be soft-rewritten
(see §5).

### Stage 3 — Build timeline

**Input:** `claims.json` (filtered to `claim_type='timeline'`).

**Output** — `timeline.json`:

```json
{
  "story_id": "...",
  "events": [
    {
      "event_id": "evt_01",
      "occurred_at": "2018-03-12T20:43:00Z",
      "occurred_at_precision": "minute",
      "description": "911 call placed",
      "claim_ids": ["clm_..."],
      "confidence": "verified"
    }
  ]
}
```

`occurred_at_precision` ∈ `{exact_time, minute, hour, day, month, year, decade, unknown}`.
Conflicts between sources surface as multiple events with the same
`description` and different `confidence`.

### Stage 4 — Identify entities

**Input:** `sources.json` + `claims.json`.

**Output** — `entities.json`:

```json
{
  "story_id": "...",
  "entities": [
    {
      "entity_id": "ent_01",
      "kind": "person",
      "canonical_name": "John Doe",
      "aliases": ["Johnny", "J. Doe"],
      "attributes": {"role": "victim", "location": "Chicago, IL"},
      "claim_ids": ["clm_..."]
    }
  ]
}
```

`kind` ∈ `{person, place, organization, event}`. Sensitive-data
redaction (minor children, sealed records) happens here, gated by a
deny-list config; redacted entities still get an `entity_id` but their
`canonical_name` is replaced with a placeholder and the redaction is
logged in `validation_report.json`.

### Stage 5 — Propose 3 angles

**Input:** `claims.json` + `timeline.json` + `entities.json`.

**Output** — `angles.json`:

```json
{
  "story_id": "...",
  "angles": [
    {
      "angle_id": "ang_01",
      "headline": "...",
      "thesis": "...",
      "evidence_strength": 0.82,
      "supporting_claim_ids": ["clm_..."],
      "missing_evidence": ["motive_for_X"],
      "five_beat_fit": {
        "fast_hook": "...",
        "character_humanization": "...",
        "contextual_build": "...",
        "escalation": "...",
        "delayed_payoff": "..."
      }
    }
  ]
}
```

`evidence_strength` is a 0.0–1.0 score derived from the count and
confidence of supporting claims. Minimum threshold for a candidate is
`0.4`; below that the proposer flags insufficient material rather than
fabricating an angle.

**Approval gate:** user picks one `angle_id` and re-runs.

### Stage 6 — Generate outline

**Input:** `claims.json` + `entities.json` + selected `angle_id`.

**Output** — `selected_outline.json`:

```json
{
  "story_id": "...",
  "angle_id": "ang_01",
  "five_beats": [
    {
      "beat": "fast_hook",
      "window_seconds": [0, 45],
      "summary": "...",
      "claim_budget": ["clm_..."],
      "must_not_reveal": ["clm_reveal_01"]
    }
  ],
  "reveal_claim_id": "clm_reveal_01",
  "reveal_target_position": 0.92
}
```

`reveal_target_position` ∈ `(0.9, 1.0]` enforces the 10% rule.

**Approval gate:** user reviews + resumes.

### Stage 7 — Draft script

**Input:** `selected_outline.json` + `claims.json` + `entities.json`.

**Output** — `script.md`:

```markdown
---
story_id: ...
angle_id: ang_01
mode: narrative_nonfiction
length_target_sec: 360
length_actual_sec: 355
---

## Fast Hook

It was the LAST night anyone would ever see her... [hesitant] and
nobody noticed she was gone for three days.

<!-- claim: clm_001 (factual, src_yt_01@01:23) -->
<!-- claim: clm_002 (emotional_inference, rewrite_applied) -->
...
```

Script embeds claim references as HTML comments so the validator can
trace every assertion back to a ledger row without surfacing the
references in the audio.

### Stage 8 — Validate citations

**Input:** `script.md` + `claims.json`.

**Output** — `validation_report.json`. See §9 for the acceptance-test
schema.

**Approval gate:** user reviews + decides ship vs. revise.

### Stage 9 — Generate voice direction

**Input:** `script.md` + `selected_outline.json`.

**Output** — `voice_direction.json`:

```json
{
  "story_id": "...",
  "sections": [
    {
      "beat": "fast_hook",
      "tone": "tense, restrained",
      "pace": "slow-to-medium",
      "pauses": ["after first sentence", "before final hook line"],
      "emphasis": ["LAST", "three days"],
      "tts_tags": ["[hesitant]"],
      "narrator_intensity": 0.6
    }
  ]
}
```

This is a sidecar; v0 does not call ElevenLabs. The render layer reads
this file when audio rendering is added in v1.

---

## 5. Claim ledger schema

The most important pre-code primitive. Without typed claims the
fact-check gate cannot be specific.

### v0 claim-type taxonomy (seven types)

`reconstructed` is intentionally **excluded** in v0 because Dramatized
True Story mode is out of scope.

| Type | What it asserts | Example | Default v0 confidence |
| --- | --- | --- | --- |
| `factual` | Verifiable real-world assertion | "The car was found near the river." | verified |
| `timeline` | Time / sequence assertion | "The 911 call came in at 8:43 PM." | verified |
| `entity` | A person, place, or organization assertion | "John Doe lived in Chicago." | verified |
| `emotional_inference` | Internal state attributed to a real subject | "She felt isolated." | inferred (must be soft-rewritten) |
| `disputed` | A claim where sources disagree | "The defense says the meeting never happened." | disputed |
| `reveal` | The story's final reveal / outcome / perpetrator identity | "Police arrested the neighbor." | verified |
| `transition` | Author voice / scene-setting / pacing — not a factual assertion | "Hours passed before anyone noticed." | unknown |

`reveal` is a first-class type so the **10% rule** (outcome cannot
appear in the first 90% of the script) becomes a deterministic
claim-placement gate, not a keyword scan.

### Schema

```
claim_id          string   (e.g. "clm_001")
story_id          string
text              string   — assertion as it appears in the script (post-rewrite)
claim_type        enum     — one of the seven v0 types above
source_id         string?  — fk -> sources.json; null only for `transition`
quote             string?  — verbatim excerpt; null only for `transition`
source_locator:                — structured (no more single-string locators)
  url             string?
  paragraph       int?     — 0-indexed paragraph in the source text
  timestamp       string?  — ISO-8601 duration into a transcript ("00:14:23")
  quote_offset    int?     — 0-indexed char offset within the source text
confidence        enum     — verified | inferred | disputed | unknown
mode_constraint   enum     — strict | narrative | dramatized
                             v0 always writes "narrative"
rewrite_applied   bool     — true iff emotional_inference soft-rewrite ran
original_text     string?  — pre-rewrite assertion (audit trail)
inserted_by       enum     — extraction_pass | drafter | revision | reviewer
verified_by       string?  — reviewer_id; null in v0 (no human verification yet)
verified_at       string?  — ISO-8601 timestamp
```

### Per-type fact-check invariants

| Type | Requires `source_id`? | Requires `quote`? | Requires `source_locator.{paragraph,timestamp}`? | Other |
| --- | --- | --- | --- | --- |
| `factual` | yes | yes | at least one | — |
| `timeline` | yes | yes | at least one | also produces an event in `timeline.json` |
| `entity` | yes | yes | at least one | also produces an `entity_id` |
| `emotional_inference` | yes | yes | at least one | `rewrite_applied=true` mandatory before script lands |
| `disputed` | yes — at least 2 distinct `source_id` across the disagreement set | yes | at least one per source | both sides surface in `validation_report.json` |
| `reveal` | yes | yes | at least one | placement gate fires (see §6) |
| `transition` | no | no | no | logged for placement gates only |

### Inferred-claim soft-rewrite policy

Internal tag, external rewrite to hedged language. The drafter prompt
embeds the rule; the validator enforces `rewrite_applied=true` on every
`emotional_inference` claim before the script lands.

| Original (rejected) | Soft-rewritten (allowed) |
| --- | --- |
| "She was terrified." | "Based on the reporting, the situation appears to have left her afraid and isolated." |
| "He knew what he had done." | "Investigators later concluded he understood the consequences of his actions." |
| "The neighbors didn't care." | "No neighbor came forward to police in the days that followed." |

`original_text` preserves the pre-rewrite assertion for audit.

---

## 6. Validation rules

Deterministic gates. Each rule emits a `pass | warn | fail` finding in
`validation_report.json` with the offending `claim_id` or script offset.

| Rule | What it checks | Fail behavior |
| --- | --- | --- |
| **CITE-001** | Every script claim reference resolves to a `claim_id` in `claims.json`. | fail |
| **CITE-002** | Every `factual | timeline | entity | disputed | reveal` claim has a non-null `source_id` and `quote`. | fail |
| **CITE-003** | Every `emotional_inference` claim has `rewrite_applied=true` and a non-null `original_text`. | fail |
| **CITE-004** | Every `disputed` claim has at least 2 distinct `source_id`s. | fail |
| **REVEAL-001** | The `reveal_claim_id` does not appear in script positions before `reveal_target_position` (default 0.92). | fail |
| **PREFACE-001** | Any technical/historical setup that cannot be woven smoothly is surfaced before the Fast Hook. | warn (advisory) |
| **STRUCTURE-001** | Script contains all five beats in order: fast_hook → character_humanization → contextual_build → escalation → delayed_payoff. | fail |
| **NARRATION-001** | Script contains at least N pause markers (`...`) and emphasis markers (CAPS) per beat. Thresholds in config. | warn |
| **NAMED-001** | No claim asserts criminal/wrongful conduct against a named living person without a `source_id` of type `news_article`, `court_record`, or `verified_official_statement`. v0: only `news_article` qualifies. | fail |
| **REDACT-001** | No redacted entity surfaces in the script body. | fail |

Stage 8 runs every rule. Any `fail` aborts the package as
`status="failed"`; the user must re-run after fixing.

---

## 7. Sample fixture requirements

A "golden case" fixture lives at
[`extracted_evidence_to_story/fixtures/evidence_to_story_v0_golden/`](../fixtures/evidence_to_story_v0_golden/).

Selection criteria for the case (one human-judgment call before any
v0 build):

| Criterion | Why |
| --- | --- |
| Has at least 1 long-form YouTube treatment | Source loader requires this |
| Has at least 1 reputable news article on the same case | Cross-source claim verification |
| Clear, factual timeline of events | Stage 3 has something to anchor on |
| Clear final reveal/outcome | Stage 6 + REVEAL-001 require this |
| Not legally contested at time of fixture creation | Avoids fact-check moving-target |
| Not involving minors | Avoids redaction edge cases in v0 |
| Not famous (no Wikipedia top-line) | Avoids audience prior knowledge baking in expectations the engine can't see |
| Real, not synthesized | The engine must work against real messy text, not a clean fixture |

The fixture's `inputs/manifest.json` follows the Stage-1 input shape
above. Expected outputs under `expected/` are populated by an
authoritative manual run; future runs are diffed against `expected/`
in CI. v0 does not gate CI on full-text equality (LLM nondeterminism
makes that wrong) — instead, the diff checks shape, claim-count
ranges, and acceptance-test pass/fail parity.

---

## 8. Model routing

Per-stage tier. Concrete model IDs are decided at v0 build time;
this table fixes the *roles*.

| Stage | Tier | Why |
| --- | --- | --- |
| 1. Load sources | deterministic | No reasoning needed |
| 2. Extract claims | structured-extraction | High-volume, schema-bound, cost-sensitive |
| 3. Build timeline | deterministic + structured-extraction (advisory) | Mostly merging stage-2 output |
| 4. Identify entities | structured-extraction | Schema-bound |
| 5. Propose 3 angles | structured-extraction | Three concise outputs against a fixed schema |
| 6. Generate outline | writing | Genre-specific structural reasoning |
| 7. Draft script | writing | The moat — sentence rhythm, sensory detail |
| 8. Validate citations | deterministic + writing (advisory only) | Hot path is rule-based; LLM only for ambiguous resolution |
| 9. Generate voice direction | structured-extraction | Schema-bound sidecar |

**Tier-to-model mapping (footnote, decided at build time).** Atlas has
both cloud and local options. Likely defaults at v0:

- `structured-extraction` → Haiku 4.5 (cloud) or Qwen3:14B (local)
- `writing` → Sonnet 4.6 or Opus 4.7 (cloud); local fallback only if the host explicitly opts in via config

Cost reconciliation goes through the existing
`extracted_llm_infrastructure/services/cost/` ledger from day one — no
ad-hoc cost tracking. Every LLM call writes a `llm_usage` row.

---

## 9. Acceptance tests

The engine "passes" v0 if **every** test below resolves to `pass` on
the golden fixture. Each maps to one or more validation rules from §6.

| ID | Test | Maps to rule(s) |
| --- | --- | --- |
| AT-1 | Every factual script claim maps to a claim ledger row | CITE-001 |
| AT-2 | Every claim ledger row maps to a source quote with locator | CITE-002 |
| AT-3 | The final reveal does not appear before the final 10% of the script | REVEAL-001 |
| AT-4 | The outline follows the five-beat structure | STRUCTURE-001 |
| AT-5 | The script is narration-ready (pause markers + emphasis markers present per beat) | NARRATION-001 |
| AT-6 | No unsupported named-person accusations are added | NAMED-001 |
| AT-7 | Emotional inferences are softened, not stated as facts | CITE-003 |
| AT-8 | Validation report clearly shows passed/failed checks (this doc's own contract — `validation_report.json` is human-readable) | meta |

`validation_report.json` shape:

```json
{
  "story_id": "...",
  "status": "passed | failed",
  "summary": {"passed": 7, "warned": 1, "failed": 0},
  "findings": [
    {
      "rule_id": "CITE-002",
      "level": "pass | warn | fail",
      "claim_id": "clm_017",
      "script_position": 0.34,
      "message": "..."
    }
  ],
  "acceptance_tests": [
    {"test_id": "AT-1", "result": "pass"},
    ...
  ]
}
```

---

## 10. Future v1/v2 notes

What v0 deliberately defers and the order of arrival likely:

1. **ElevenLabs render integration (v1).** Voice-direction sidecar →
   audio. The v0 sidecar shape is designed so the render layer is a
   pure function of `script.md + voice_direction.json`.
2. **Strict Documentary mode (v1).** Tighter `confidence='verified'`
   gate, drops `emotional_inference` allowance, requires `verified_by`
   on every claim.
3. **Dramatized True Story mode (v2).** Re-introduces `reconstructed`
   claim type and the on-screen `[reconstructed for narrative]` label.
4. **Multi-pass orchestration (v2).** Outline → draft → critique →
   revise. Triggered when v0 retention numbers prove the moat exists
   but length scales beyond what single-pass produces well.
5. **Loader expansion (v1).** Court records, police reports (with
   OCR), interview transcripts (with speaker labels), scraped case
   databases (reusing `atlas_brain/services/scraping/`).
6. **Web UI / Slack-bot approval gates (v1).** CLI is the v0 surface;
   the gate boundaries are designed so a UI can wrap each.
7. **User state (v2).** Preferred mode, channel-archetype bias, past
   angle choices.
8. **Adjacent output surfaces (v2+).** Documentary scripts, podcast
   episode scripts, court-style case reports, Substack-style
   longform. Same engine, different render templates. Out of scope
   until v0 retention proves the niche.

---

## Resume condition

This work resumes only after the campaign-core spine is fully
product-owned per `remaining_productization_audit.md` "Next Concrete
Slice" — migrating producer flow from copied
`b2b_campaign_generation.py` into `CampaignGenerationService` using
the normalized ports.
