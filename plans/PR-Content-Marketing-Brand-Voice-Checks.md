# PR-Content-Marketing-Brand-Voice-Checks

## Why this slice exists

A stashed prototype (`atlas_brain/services/brand_voice_validator.py` +
`atlas_brain/skills/brand/brand_voice.yml` + a `.github/workflows/marketing_content_check.yml`
workflow) tries to give content marketers a deterministic "checks and balances"
lane mirroring Atlas's quality gate: marketers edit the YAML (forbidden words +
tone/content regex rules) and CI fails the PR on a brand-voice violation. The
intent is sound and the audience adaptation (YAML-externalized rules so
non-coders edit config, not Python) is the right call. But the prototype shipped
against the AGENTS.md contract in three load-bearing ways and was partly
non-functional, and the directory it gates did not exist.

This slice both hardens the checker and stands up the content tree it guards, so
the lane lands working, tested, and active in one PR:

1. The checker was a *checker with zero tests* (AGENTS.md 3i headline gap). Two
   latent defects (the inverted `content_rules` default, the dead
   `vocabulary.use` config) shipped precisely because no test exercised them.
2. The CI gate was broken and fail-open: a shallow checkout left no base SHA so
   the diff errored red on every PR, and even when reached it validated only the
   first changed file and silently passed the unrecognized-type branch.
3. The validator crashed on hand-edited config (KeyError on a missing rule key,
   TypeError on a commented-out section, AttributeError on an empty file) and
   substring-matched forbidden words, flagging clean prose ("non-disruptive",
   "Transformer architecture", "deleverage").
4. The gate watched `marketing/**`, which did not exist -- so it was dormant.

## Scope

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Robust testing

1. Add `tests/test_brand_voice_validator.py` -- the AGENTS.md 3i
   failure-detection suite: one negative fixture per detection branch asserting
   the specific violation fires, plus allowed near-miss fixtures that still pass,
   plus malformed-config guards. Mutation-verified (each fix proven to bite).
2. Enroll that test in CI with a dedicated workflow,
   `.github/workflows/atlas_brand_voice_checks.yml` (pull_request + push path filters
   over the validator, config, and test; a pytest run step).
3. Harden `atlas_brain/services/brand_voice_validator.py`: whole-word vocabulary
   matching, load-time rule-shape validation (clear error, not an opaque
   KeyError mid-gate), None/empty-config and non-string-text guards,
   `content_rules` `fail_on_match` default True (was silently inverting), and
   ASCII CLI output (was emoji).
4. Fix the CI gate `.github/workflows/marketing_content_check.yml`: full fetch so
   the base SHA exists, validate all changed marketing files (not the first
   only), cover four content types, surface (never silently skip) an
   unrecognized path, and drop the deprecated set-output API.
5. Relabel the dead `vocabulary.use` block as FUTURE in
   `atlas_brain/skills/brand/brand_voice.yml`.
6. Stand up the real `marketing/` content tree with on-brand seed copy per
   content type plus `marketing/README.md`, so the gate is active and marketers
   have a worked example. Every seed file validates clean against the shipped
   brand voice.

### Files touched

- `.github/workflows/atlas_brand_voice_checks.yml`
- `.github/workflows/marketing_content_check.yml`
- `atlas_brain/services/brand_voice_validator.py`
- `atlas_brain/skills/brand/brand_voice.yml`
- `marketing/README.md`
- `marketing/blog_posts/why-deterministic-checks.md`
- `marketing/landing_pages/atlas-platform.md`
- `marketing/release_notes/2026-06-release.md`
- `marketing/tweets/launch-brand-voice-checks.md`
- `plans/PR-Content-Marketing-Brand-Voice-Checks.md`
- `tests/test_brand_voice_validator.py`

### Review Contract

Acceptance criteria (reviewer checks one-by-one):

- Every detection branch has a negative fixture proving it fires with the
  expected violation text: a forbidden word; a tone pattern; a landing_page
  missing extensibility; a release_notes with future tense; and a content rule
  that omits `fail_on_match` (must ban, not invert).
- Every denylist/pattern change has an allowed near-miss that still passes:
  "non-disruptive" / "deleverage" / "Transformer architecture" do not trip the
  vocab list; a single exclamation mark; an on-brand landing_page.
- Malformed-config cases degrade safely, not fatally: None config, a
  commented-out (None) section, a rule missing a key, and a bad regex each
  produce a clear error -- never an opaque crash that disables the whole gate.
- CI: the new dedicated workflow runs the suite on changes to the validator,
  config, or test; the marketing gate validates all changed marketing files and
  does not silently skip an unrecognized marketing path.
- ASCII gate clean for the touched Python files; every seed content file is
  on-brand against the shipped brand voice.

Affected surfaces: the validator service + CLI, its YAML schema, the marketing
CI workflow, the new test workflow, and the seed content tree. Risk areas: regex
precision (over/under-match), CI shell semantics, and silent-pass paths.
Reviewer rule triggers: checker-without-test (3i), ASCII-only Python,
fail-closed CI / surface-don't-skip drift (3g).

## Mechanism

Pure, deterministic, no-I/O scanner (unchanged shape): `validate(text,
content_type)` returns a `list[str]` of violations; the CLI exits 1 on any
violation, 0 on clean. The fixes are surgical:

- **Tests.** `tests/test_brand_voice_validator.py` builds a `BrandVoiceValidator`
  from small inline YAML configs (and the real shipped config) and asserts exact
  violation strings per branch, plus near-miss configs that return an empty list.
- **CI enrollment.** `.github/workflows/atlas_brand_voice_checks.yml` runs the suite on
  pull_request and push when the validator, config, or test changes.
- **Config hardening.** The config and each section coerce None to empty; rule
  shape is validated at load and names the offending rule; non-string text raises
  a clear error.
- **`fail_on_match` default.** Read as `rule.get("fail_on_match", True)` on the
  `content_rules` path, matching `tone_rules`, so a forgotten key bans rather
  than silently inverting.
- **Word matching.** Forbidden vocabulary is matched whole-word, removing the
  prefix-glued false positives.
- **CI loop.** The marketing workflow fetches full history, iterates every
  changed marketing file, maps each directory to a content type, and exits
  non-zero if any file violates.
- **Content tree.** `marketing/` gains one on-brand seed file per content type
  plus a README that points marketers at the YAML as the editable source of
  truth.

## Intentional

- The YAML-externalized rule catalogue is the design (not folded into Python):
  the point of this lane is that marketers edit config, not code.
- Return type stays a flat `list[str]` with binary pass/fail this slice.
  Structured severity (BLOCKER/MAJOR/NIT) is a real adopt-from-our-lane
  improvement but is its own slice (see Deferred).
- Whole-word matching is the precision fix this slice; stem-aware patterns (to
  also catch inflections substring under-matched) are deferred to the severity
  slice that reworks the rule evaluator.
- A broken individual rule is surfaced with a clear load-time error, never
  silently no-op'd into a green pass.
- Seed copy is written to the brand persona in the YAML, not the operator's
  personal voice; one file per type, enough to activate the gate.

## Deferred

- Structured findings carrying a severity enum (BLOCKER/MAJOR/NIT) replacing the
  flat list -- adopt-from-our-lane slice.
- Suggested-fix output: implement `vocabulary.use` (preferred -> discouraged) so
  a discouraged word surfaces its replacement as an advisory.
- Stem-aware vocabulary patterns owned in the YAML (catch "synergies",
  "utilizing" that whole-word matching under-matches).
- Pure-module + thin entrypoint split and a CANONICAL.md note on the relationship
  to `extracted_content_pipeline/brand_voice.py`.
- A real ongoing content corpus -- this PR seeds, it is not the library.

Parked hardening: none.

## Verification

- The failure-detection suite at `tests/test_brand_voice_validator.py` runs green
  (41 passed) and is mutation-verified: reverting each fix turns its tests red.
- The validator at `atlas_brain/services/brand_voice_validator.py` and the test
  pass the ASCII-only Python policy (no non-ASCII bytes).
- CLI smoke against `atlas_brain/skills/brand/brand_voice.yml`: a clean
  landing-page fixture exits 0; a fixture with a forbidden word, double
  punctuation, and missing extensibility exits 1 with the expected lines; a
  known false-positive fixture ("non-disruptive", "Transformer architecture")
  exits 0.
- Each seed file under `marketing/` validates clean for its content type; a
  negative control with future tense, a forbidden word, and double punctuation
  exits 1.
- The new workflow `.github/workflows/atlas_brand_voice_checks.yml` enrolls the test
  with pull_request and push path filters and a pytest run step.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_brand_voice_checks.yml` | 38 |
| `.github/workflows/marketing_content_check.yml` | 72 |
| `atlas_brain/services/brand_voice_validator.py` | 165 |
| `atlas_brain/skills/brand/brand_voice.yml` | 59 |
| `marketing/README.md` | 29 |
| `marketing/blog_posts/why-deterministic-checks.md` | 12 |
| `marketing/landing_pages/atlas-platform.md` | 17 |
| `marketing/release_notes/2026-06-release.md` | 14 |
| `marketing/tweets/launch-brand-voice-checks.md` | 4 |
| `plans/PR-Content-Marketing-Brand-Voice-Checks.md` | 185 |
| `tests/test_brand_voice_validator.py` | 669 |
| **Total** | **~1264** |
