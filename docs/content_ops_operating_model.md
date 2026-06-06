# How Atlas Ships Marketing Content — The Same Operating Model, Adapted

> A sister to `docs/ai_dev_operating_model.md`. Same spine — plan a thin artifact,
> let mechanical gates catch every repeatable failure, keep the builder and reviewer
> separate, turn every miss into a gate — adapted to content marketing, where the
> artifact is structured copy, the "test" is the market, and the value compounds in a
> living record of what worked.
>
> **Diagram note.** `content_ops_operating_model.svg` shows the *simplified* 7-stage
> flow. This writeup is canonical and goes further (a 0→11 lifecycle, a four-part gate
> stack, the review contract). A v2 diagram — a clean top-level flow plus a separate
> "review contract" canvas — is **deferred** rather than crammed onto one page.

The dev model and this one share a thesis: **make the machine catch every failure
mode a script can decide, so human judgment is only ever spent where a script can't.**
Two things change in content, and they change the shape of the whole loop:

1. **The artifact is structured data, not freeform code** — so the contract is a
   *schema*, and the schema is the interface (no UI required).
2. **The real test runs after you ship, and it's probabilistic** — so the lifecycle
   doesn't end at publish, and the source of truth is a *living record of outcomes*,
   not a write-only archive.

And one thing the dev model gets wrong if you copy it naively, corrected here: **the
LLM is not a judge.** Code has hard surfaces (types, tests, imports) where a model can
sometimes adjudicate. Content is squishier — tone, persuasion, promise clarity — and a
model will happily be convinced by its own polished draft. So the model never *decides*
in this pipeline. It produces **structured review evidence** a human must resolve.

---

## 1. The crew model: writer + editor, per channel lane

Work splits into **channel/campaign lanes** (e.g. "lifecycle email", "SEO blog"). Each
lane is a pair:

- **Writer session** — fills the brief + asset schema, drafts the copy, runs the gates.
- **Editor session** — reviews *independently* against a frozen contract and posts an
  accountable decision (§5). Most of the editor's judgment goes to **taste and
  persuasion**, because the mechanical gates have already settled consistency and
  compliance.

Same reason as code: a model (or human) editing its own work inherits its own blind
spots. The editor re-judges from scratch and never rubber-stamps.

---

## 2. The unit of work: a brief + an asset schema (the contract *is* the interface)

Nothing ships without a **contract first** — but the contract is a filled schema, not a
prose plan. Two layers, kept distinct on purpose:

- **The brief (intent).** `objective`, `audience`, `funnel_stage`, `reader_problem`,
  `primary_promise`, `proof`, `offer`/`CTA`, `channel`, `risk_tier`, `success_metric`.
  This is the strategy the editor judges *good* against.
- **The asset schema (structure).** Typed fields per output type — an email is
  `subject / preheader / body_blocks[] / cta / utm`; a blog is
  `title / meta / outline_h2[] / claims[] / internal_links[]`. This is what the gates
  validate.

Folding strategy into structure is the failure mode: you get copy that is schema-valid
and soulless. Keep both layers; gates run mostly against the asset schema, the human
judges against the brief.

**The schema is the UI.** You don't need a webpage — you need a typed document the
session fills and the gates validate. Once the contract exists, most operator turns are
"approve / next"; you only step in at the things that can't be mechanized: **taste**,
the **brief's own quality**, and the eventual **verdict**.

---

## 3. The lifecycle: 0 → 11 (publish is the midpoint, not the finish)

Code's terminal state is real — merge plus green CI means done. Content has no such
moment, because the test runs *after* you ship and answers days later, probabilistically.
So the lifecycle has a tail code lacks, and a triage gate code rarely needs:

```
 0 · Triage            Should this asset even exist?
 1 · Brief + schema    Audience, promise, offer, CTA, channel, metric, risk tier
 2 · Evidence packet   Registry claims, proof points, prior winners, examples
 3 · Writer drafts     Fields first, prose second
 4 · Contract gates    Schema · claims · compliance · structure · tracking   (3A/3B)
 5 · Model-assisted    Adversarial pass: drift, ambiguity, weak proof, genericness (3C)
     review
 6 · Editor review     Forced rule coverage, cited evidence, exception logging   (3D)
 7 · Experiment +      Hypothesis, metric, attribution window, links, UTM,
     publish QA        launch readiness
 8 · Publish           Ship to channel
 9 · Measure           Wait the window. No premature victory dances.
10 · Verdict           Worked / didn't / inconclusive — structured reason
11 · Compound          Miss → gate · win → prior · override → calibration ·
                       claim change → registry
```

### Stage 0 — "Should this exist?"

Most content requests are organizational anxiety wearing a hat. Before any brief, a
cheap triage decides **create / clone-an-existing-winner / defer / reject** from:
audience segment, lifecycle stage, business goal, expected behavior change, channel,
opportunity size, reuse potential, risk level, *why now*. This is the gate that keeps
the pipeline from becoming an efficient machine for producing polished landfill.

### Stage 7 — the experiment contract (defined *before* ship)

The measurement plan is frozen *before* the asset publishes, or the verdict ledger rots
into "the graph went up and nobody wants to argue." Required fields: `hypothesis`,
`primary_metric`, `secondary_metric`, `attribution_window`, `audience/segment`,
`control_or_comparison`, `min_sample_size` (rough is fine), `what_counts_as_worked`,
`what_counts_as_inconclusive`, `decision_if_it_works` / `decision_if_it_doesn't`.

---

## 4. The gate stack — four sub-gates, deterministic where it can be

The old single "Content gates" box splits into four, so the deterministic parts stay
deterministic and the squishy parts are *assisted* but never automated into fake
certainty:

| Sub-gate | What it checks | Type | **Who decides** |
|---|---|---|---|
| **3A · Schema** | required fields, asset format, channel constraints, word count, CTA present | deterministic | system |
| **3B · Claims / compliance** | claims match registry, disclaimers, pricing/product names, banned phrases, required legal language, source freshness | deterministic (+ source mapping) | system |
| **3C · Model-assisted review** | voice drift, promise clarity, specificity, persuasion weakness, reader objections, genericness | LLM-assisted | **editor** |
| **3D · Human editor decision** | the accountable call | judgment | **editor** (effectiveness → market) |

Deterministic gates settle what a regex/registry can decide. **3C does not adjudicate** —
it lints and argues; the editor resolves. Effectiveness is settled only post-publish, by
the market. Atlas already has substrate here: `extracted_quality_gate` (deterministic
packs: blog / campaign / witness-specificity / evidence-coverage / source-quality) and
brand-voice profiles (`content_ops_brand_voice_profiles`).

### The claims map (highest-value 3B addition)

The draft auto-produces a **claims map** — every meaningful claim extracted and
classified against the registry:

```
Claim:            "Save 30% on all plans"
Registry ID:      pricing.discount.q4
Approved wording: "Save up to 30% on eligible annual plans"
Risk level:       high
Expiration:       2026-01-15
Appears at:       hero, line 1
Status:           FAIL (mismatch: "all plans" vs "eligible annual plans")
```

Not subjective. This is the system quietly preventing marketing from inventing a
lawsuit — and it makes every claim trivially re-checkable when the registry changes.

### The LLM does not say "LGTM" — it fills a coverage matrix

3C's model is given a narrow job and forced into JSON, never prose approval:

> You are not approving this draft. You produce review evidence. Cite exact draft
> spans. Cite exact rule IDs. Mark every required rule `PASS` / `FAIL` / `NOT_APPLICABLE`.
> Missing evidence = `FAIL`. Do not invent rules. Do not rewrite unless asked.

```json
{
  "review_id": "...",
  "ruleset_versions": { "brief": "...", "brand_voice": "...", "claim_registry": "...",
                        "compliance": "...", "channel_schema": "..." },
  "rule_coverage": [
    { "rule_id": "VOICE-01", "status": "PASS", "draft_span": "...", "reason": "...", "confidence": 0.82 },
    { "rule_id": "CLAIM-03", "status": "FAIL", "draft_span": "...", "registry_conflict": "registry says X, draft says Y", "severity": "blocker", "confidence": 0.94 }
  ],
  "issues": [
    { "issue_type": "unsupported_claim", "severity": "blocker", "rule_id": "CLAIM-03",
      "draft_span": "...", "why_it_matters": "...", "recommended_action": "revise|remove|escalate" }
  ],
  "approval_recommendation": "DO_NOT_APPROVE"
}
```

**Adversarial pass.** A second model-assisted prompt (different prompt, maybe different
model, same inputs) does *only* one thing: find the strongest reason this should not
ship — overclaims, ambiguity, reader objections, promise/CTA mismatch, generic stretches,
missing proof, voice slips. It is the annoying reviewer who catches what everyone missed.
Still not a judge.

> **Deferred / advanced (my call, not the reviewer's):** treating *model disagreement*
> as an orchestrated signal (LLM-A pass vs LLM-B fail → route to human, log override as
> calibration data) is elegant but the highest-complexity, lowest-marginal-value piece.
> Deterministic blockers always block and model-found blockers always go to a human
> regardless — so we get most of the benefit without the orchestration. Park it.

---

## 5. The review contract — the anti-drift mechanism

The guarantee against drift **cannot come from the reviewer** (humans drift, models
drift, rules drift). It comes from the *interface*. A reviewer is never handed a draft
and asked "thoughts?" — that is how you get vibes. They are handed a **Content PR**:

1. **Brief snapshot** — immutable at review time (id, asset type, audience, lifecycle
   stage, reader problem, primary promise, offer, CTA, channel, risk tier, success metric).
2. **Rule packet** — versioned and frozen: brand-voice contract version, claim-registry
   version, compliance pack version, schema version, channel-rules version.
3. **Draft.**
4. **Claims map** (§4).
5. **Gate results** — pass / fail / needs-human-judgment.
6. **Required reviewer coverage matrix** — the reviewer cannot approve until every
   required row is resolved with cited evidence:

   | Rule ID | Rule | Reviewer must provide | Status |
   |---|---|---|---|
   | VOICE-02 | Avoids hype language | quote or issue | pass/fail |
   | CLAIM-04 | Pricing claims match registry | system evidence | pass/fail |
   | PROMISE-02 | Primary promise appears early / above fold | quote location | pass/fail |
   | RISK-01 | No unsupported competitive claim | claims-map evidence | pass/fail |

**The rule: no required rule passes silently.** Missing coverage = fail.

### Comment discipline

Every reviewer comment must attach to one of: a brief requirement, a brand rule, a claim
registry item, a compliance rule, a channel constraint, a performance hypothesis, or a
named editorial-judgment category. So instead of "can we make this punchier?" you get:

```
Category: Promise clarity   Severity: medium
Issue:    Opening delays the reader benefit until paragraph 3.
Evidence: "<span>"          Suggested fix: move the outcome into the first sentence.
```

> **My one softening of the reviewer here:** drive-by polish notes shouldn't be *illegal*,
> they should be a **NIT** category (the dev model already has BLOCKER/MAJOR/NIT/LGTM).
> A NIT is still categorized and still non-blocking — that preserves cheap taste input
> without letting "feels off-brand" masquerade as a blocker.

### Decision states (incl. "approve with exception")

`BLOCKED` · `REVISION REQUIRED` · `APPROVED` · `APPROVED WITH EXCEPTION` · `ESCALATED`.

Sometimes a piece should ship despite a soft-rule violation — but never *invisibly*. An
exception logs: `exception_rule`, `reason`, `owner`, `expiration`, `should_this_update_the_rule?`.
This keeps tribal knowledge out of Slack, where knowledge goes to die.

### Four review targets (in order)

| Target | Question |
|---|---|
| Draft vs brief | Did the writer satisfy the contract? |
| Brief quality | Was *this* the right promise, audience, offer, CTA? (sometimes the brief was garbage in a blazer) |
| Market result | Did the asset actually work? |
| System learning | Should we update a gate, registry, prior, or calibration set? |

---

## 6. Risk tiering — so the process isn't bureaucratic sludge

Not every asset earns the same review burden. The risk tier (set in the brief, §2) routes
the work — this is the primary lever against over-engineering harmless copy:

| Tier | Example | Required review |
|---|---|---|
| **Low** | internal nurture email, low-claim social | automated gates + editor |
| **Medium** | lifecycle email with pricing/product claims | gates + editor + claims check |
| **High** | regulated/legal/medical/financial, aggressive competitor comparison | gates + editor + compliance/legal |
| **Critical** | public launch, major paid campaign, exec byline | full review + experiment contract + postmortem |

---

## 7. Measure → verdict → compound (where the value lives)

- **Measure** waits for the (delayed, noisy) signal defined by the experiment contract.
- **Verdict** is a human call the data informs — *worked / didn't / inconclusive* — but
  the **"why" is structured**, drawn from a **failure taxonomy**: wrong audience · weak
  offer · weak hook · unclear promise · claim-credibility gap · CTA friction · landing
  mismatch · timing · channel mismatch · deliverability · too generic · too complex ·
  insufficient proof · compliance weakened the message · inconclusive data.
- **Compound** routes the verdict into the flywheels (§9).

Structured "why" is what makes verdicts *reusable* instead of anecdotal: you don't just
learn "this failed," you learn *which type* of failure keeps recurring.

---

## 8. The source of truth: a *living* record (now a quartet)

The sharpest divergence from the dev model. In code, the merged artifact is truth and
the plan is disposable — we archive-and-forget plans on merge. In content, the published
piece is *not* the retained value; **the outcomes and winning patterns are**, read
*before every new brief*. This is the inverse of `plans/archive/`: consulted constantly,
never retired. It is content's `CANONICAL.md`, not its plan archive.

- **Campaign ledger (lightweight verdict log).** One record per piece:
  `hypothesis · assets · headline metric · worked-didn't-why (taxonomy)`. Deliberately
  cheap, so it actually gets kept and queried.
- **Claim / messaging registry.** Canonical stats, prices, positioning, product names —
  the source the 3B gate and claims map check against.
- **Brand-voice contract.** The voice rules the 3C drift review scores against.
- **Review calibration library** *(new — the missing anti-drift piece)*. Worked examples
  with verdicts + reasoning: approved / rejected / borderline / known-defect / good-voice
  / voice-drift / overclaim / weak-vs-strong persuasion. Without examples, "brand voice"
  is a séance. This reference set anchors *both* human and model reviewers, and overrides
  (§5) feed new examples into it.

The raw drafts *are* archivable-and-forgettable like plans; only outcomes, patterns, and
calibration examples live on.

---

## 9. Two flywheels — kill the repeat, clone the winner

Code's flywheel is purely negative: a mistake becomes a gate. Content keeps that and adds
a positive twin:

- **Negative.** A miss → a new gate (banned phrase, registry entry, rubric criterion).
  Sharpened by the failure taxonomy: **if 3+ assets fail for the same reason, that
  becomes a gate** — 3+ "unclear promise" → a Promise-Clarity gate; 3+ "insufficient
  proof" → a Proof-Density gate; 3+ "CTA friction" → a CTA-Specificity gate.
- **Positive.** A *winner* → a reusable prior (template, schema default, brief preamble),
  deliberately repeated.
- **Overrides** (approve-with-exception) → calibration examples, so the *judgment layer*
  itself compounds.

Every published piece earns a verdict; the verdict feeds whichever loop applies; all
front-load into the next brief. Quality compounds in three directions at once: fewer bad
pieces, more pieces built on what won, and sharper review.

---

## 10. Honest limits — where this can't be mechanical

- **The effectiveness test is delayed and probabilistic.** Pre-publish gates can't
  promise "this will work." They promise something narrower and honest: *"this is
  coherent, compliant, on-contract, measurable, and not obviously doomed."* Note the
  precise wording — effectiveness is **not *provable* pre-publish, but some failure modes
  *are* knowable** (weak, confusing, unsupported, misaligned, generic). Attribution noise
  means a single verdict is a weak signal; patterns emerge over many.
- **Taste isn't fully mechanizable**, which is exactly why the model is **evidence, not
  judge**, and why the dev doc's rule — **external-bot/LLM findings are advisory, never
  auto-applied** — matters *more* here, not less. A naive auto-apply loop degrades copy.
- **The ledger and calibration library rot if the discipline lapses.** The positive
  flywheel and the calibration set both depend on someone recording the verdict and
  curating examples. Keeping the ledger lightweight is the defense (a heavy ledger is the
  one that silently stops being filled) — but it still rests on a human-discipline floor.

---

## Why it works — the principles worth stealing

1. **The contract is the interface.** A brief + asset schema, filled and gated — not a UI.
2. **Triage before you build.** "Should this exist?" beats efficiently producing landfill.
3. **The LLM produces evidence, not verdicts.** Deterministic gates block hard failures;
   models surface structured risks and cite spans; humans make accountable decisions;
   the market settles effectiveness.
4. **Force complete review coverage.** No required rule passes silently; every comment is
   categorized and evidenced; overrides are logged.
5. **Risk-tier the burden.** Heavy review for what can hurt you, light for what can't.
6. **Publish is the midpoint.** Define the experiment first, measure, give a structured
   verdict, record it.
7. **Two flywheels.** Kill the repeat (miss → gate, on a 3+ threshold) and clone the
   winner (win → prior); overrides → calibration.
8. **One living source of truth, queried before every brief** — ledger, registry,
   voice contract, calibration library. The opposite of a write-only archive.

The system's job is not to make humans or models reliable — they aren't. It is to make
unreliability **visible, logged, and harder to repeat.**

---

## Building this: staged, not a monolith

The model above is a spec for a subsystem, not a weekend. It sequences into ~5 slices,
cheapest-and-highest-value first (most of it *extends* existing substrate — the claim
registry, `extracted_quality_gate` packs, brand-voice profiles — so little is greenfield):

1. **Free reframes + cheap enums** — LLM-as-evidence wording, failure taxonomy, risk
   tier field, exception state + log, the four-part gate split. (config/enum-scale)
2. **Triage gate + experiment contract** — two schemas + human decisions. (small)
3. **Claims map** — claim extraction + registry mapping + status. (its own slice)
4. **Content-PR + coverage matrix** — the review interface; the anti-drift core. (epic)
5. **Calibration library + adversarial pass** — curated set + second prompt. (ongoing)

Model-disagreement orchestration (§4) is explicitly **out of scope** until 1–5 prove out.
