# How Atlas Ships Marketing Content — The Same Operating Model, Adapted

> A sister to `docs/ai_dev_operating_model.md`. Same spine — plan a thin artifact,
> let mechanical gates catch every repeatable failure, keep the builder and reviewer
> separate, turn every miss into a gate — adapted to content marketing, where the
> artifact is structured copy, the "test" is the market, and the value compounds in a
> living record of what worked.

The dev model and this one share a thesis: **make the machine catch every failure
mode a script can decide, so human judgment is only ever spent where a script can't.**
Two things change in content, and they change the shape of the whole loop:

1. **The artifact is structured data, not freeform code** — so the contract is a
   *schema*, and the schema is the interface (no UI required).
2. **The real test runs after you ship, and it's probabilistic** — so the lifecycle
   doesn't end at publish, and the source of truth is a *living record of outcomes*,
   not a write-only archive.

---

## 1. The crew model: writer + editor, per channel lane

Work splits into **channel/campaign lanes** (e.g. "lifecycle email", "SEO blog",
"paid social"). Each lane is a pair:

- **Writer session** — fills the brief + asset schema, drafts the copy, runs the gates.
- **Editor session** — reviews *independently* against the brief and posts one verdict:
  **BLOCKER / MAJOR / NIT / LGTM**.

Same reason as code: a model editing its own copy inherits its own blind spots. The
editor re-judges against the brief and the brand-voice contract from scratch, never
rubber-stamps. The difference from code review: the editor spends most of its judgment
on **taste and persuasion** (is this on-brand, does it earn the click), because the
mechanical gates have already settled consistency and compliance.

---

## 2. The unit of work: a brief + an asset schema (the contract *is* the interface)

Nothing ships without a **contract first** — but the contract is a filled schema, not
a prose plan. It has two layers, kept distinct on purpose:

- **The brief (intent).** `objective`, `audience`, `funnel_stage`, `one_message`,
  `proof`, `offer`/`CTA`. This is the strategy the editor judges *good* against.
- **The asset schema (structure).** Typed fields per output type — an email is
  `subject / preheader / body_blocks[] / cta / utm`; a blog is
  `title / meta / outline_h2[] / claims[] / internal_links[]`; an ad is different
  again. This is what the gates validate.

Folding strategy into structure is the failure mode: you get copy that is
schema-valid and soulless. Keep both layers; gates run mostly against the asset
schema, the editor judges against the brief.

**The schema is the UI.** You don't need a webpage — you need a typed document the
session fills and the gates validate. Driving the workflow is: fill (or approve) the
schema → gates run → editor → publish. Once the contract exists, most operator turns
are "approve / next"; you only step in at the two things that can't be mechanized:
**taste** and the eventual **verdict**.

---

## 3. Pre-publish gates: what you *can* know before shipping

The gates enforce only what is knowable in advance. Two tiers:

**Deterministic (a regex/registry can decide):**

| Gate | Fails when… | Source of truth it checks against |
|---|---|---|
| **Claim / cross-channel consistency** | a stat, price, product name, or CTA disagrees with canon | the **claim/messaging registry** |
| **Compliance** | a required disclaimer is missing or a banned/regulated claim appears | compliance ruleset (**fails closed**) |
| **Structure / SEO** | title/meta length, heading shape, keyword, link, or word-budget out of spec | the asset schema |
| **Grounding** | a factual claim cites no source | the registry / provided evidence |

**LLM-judge (needs a rubric, not a regex):**

| Gate | Scores |
|---|---|
| **Voice / tone drift** | the draft against the **brand-voice contract** (lexicon, banned words, reading level, POV, formality) |

The split mirrors the dev model's honesty: deterministic gates settle
consistency/compliance/structure; the LLM-judge plus the editor handle voice and
persuasion. **What neither can settle pre-publish is whether it will work** — that is
the market's job (§4). Atlas already has scaffolding here: `extracted_quality_gate`
(deterministic packs: blog / campaign / witness-specificity / evidence-coverage /
source-quality) and brand-voice profiles (`content_ops_brand_voice_profiles`).

---

## 4. The part code doesn't have: publish is the midpoint, not the finish

Code's terminal state is real — merge plus green CI means done, move on. Content has
no such moment, because **the test runs after you ship and answers days later,
probabilistically.** So the lifecycle has a tail code lacks:

```
brief+schema → draft → gates → editor → PUBLISH → measure → verdict → record
```

- **Publish** ships to the channel (UTM-tagged so the measurement is attributable).
- **Measure** waits for the delayed signal (open/click/conversion/engagement).
- **Verdict** is a human call the data informs: *worked / didn't / why* — it can't be
  automated, because it depends on signal that didn't exist at ship time.
- **Record** writes that one-line verdict to the living ledger (§5).

This tail is where the value compounds. A model that stops at "publish" has shipped a
piece; a model that closes the loop has learned something reusable.

---

## 5. The source of truth: a *living* record, not a write-only archive

This is the sharpest divergence from the dev model. In code, the merged artifact is
the truth and the plan is disposable — we literally archive-and-forget plans on merge.
In content, the published piece is *not* the valuable retained thing; **the outcomes
and the patterns that won are**, and you read them *before every new brief*. So the
content source of truth is the inverse of `plans/archive/`: it is consulted constantly,
never retired. It is the analogue of `CANONICAL.md`, not of the plan archive.

Three living docs:

- **Campaign ledger (lightweight verdict log).** One record per campaign/piece:
  `hypothesis · assets · headline metric · worked-didn't-why`. Deliberately cheap, so
  it actually gets kept — and queried at the start of every brief ("what subject line
  won for this segment?").
- **Claim / messaging registry.** Canonical stats, prices, positioning, product names.
  The consistency gate checks copy against this; it is content's `CANONICAL.md`.
- **Brand-voice contract.** The voice rules the voice-drift gate scores against —
  content's `AGENTS.md`.

The raw drafts and process *are* archivable-and-forgettable like plans; only the
outcomes and patterns live on. Conflating the two is the trap — you don't need draft
#6 from three campaigns ago, but you very much need to know that campaign's angle
converted.

---

## 6. Two flywheels: kill the repeat, clone the winner

Code's flywheel is purely negative: a mistake that cost a cycle becomes a gate, so it
never recurs. Content keeps that one **and adds a positive twin**, which is arguably
its engine:

- **Negative (same as code).** A miss that slips — off-brand line, inconsistent stat,
  missing disclaimer — becomes a new gate: a banned phrase, a registry entry, a rubric
  criterion. It can't recur.
- **Positive (new).** A *winner* — an angle, format, or subject that converted —
  becomes a **reusable prior**: a template, a schema default, a line in the brief's
  "what's worked here" preamble. It gets deliberately repeated.

Every published piece earns a verdict (§4); the verdict feeds whichever loop applies;
both front-load into the next brief. Quality compounds in both directions at once:
fewer bad pieces, more pieces built on what already won.

---

## 7. Honest limits — where this model can't be mechanical

Same discipline as the dev doc's "Known gaps": name the residual, don't pretend it's
covered.

- **The effectiveness test is delayed and probabilistic.** Pre-publish gates can
  guarantee consistency, compliance, structure, and on-brand voice — never "this will
  convert." The flywheel still works, just on a slower, noisier loop than CI's
  two-minute verdict. Attribution noise (seasonality, audience mix, sample size) means
  a single piece's "verdict" is a weak signal; patterns emerge over many.
- **Taste isn't fully mechanizable.** More of the gate budget here is LLM-judge than
  regex, so the model leans harder on the judgment layer — which makes the dev doc's
  rule **"external-bot/LLM findings are advisory, never auto-applied"** matter *more*,
  not less. A naive "auto-apply every gate suggestion" loop degrades copy.
- **The ledger rots if the verdict step is skipped.** The whole positive flywheel
  depends on someone actually recording *worked/didn't/why*. Keeping the ledger
  lightweight is the defense (a heavy ledger is the one that silently stops being
  filled) — but it still rests on the same human-discipline floor the dev model admits.

---

## Why it works — the principles worth stealing

1. **The contract is the interface.** A brief + asset schema, filled and gated — not a
   UI. Press enter to advance; step in only for taste and the verdict.
2. **Gate what you can know before shipping; let the market test the rest.** Consistency,
   compliance, structure, voice are pre-publish; effectiveness is post-publish.
3. **Separate the writer from the editor.** Independent judgment against the brief beats
   self-review.
4. **Publish is the midpoint, not the finish.** Measure, give a verdict, write it down —
   the loop the value lives in.
5. **Two flywheels.** Kill the repeat (miss → gate) *and* clone the winner (win → prior).
6. **Keep one living source of truth, queried before every brief.** Outcomes and
   patterns live on; drafts are disposable. The opposite of a write-only archive.

The net effect mirrors the dev model: the operator runs writer/editor pairs across
channels and moves fast, because the system mechanically refuses to let any of them ship
off-brand, inconsistent, or non-compliant copy — and every new way one *could* miss, or
every angle that *won*, becomes one more thing the next brief starts with.
