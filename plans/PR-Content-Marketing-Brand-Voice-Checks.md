# PR-Content-Marketing-Brand-Voice-Checks

## Why this slice exists

A stashed prototype (`atlas_brain/services/brand_voice_validator.py` +
`atlas_brain/skills/brand/brand_voice.yml` + a `marketing_content_check.yml`
workflow, all three currently untracked) tries to give content marketers a
deterministic "checks and balances" lane mirroring Atlas's quality gate:
marketers edit `brand_voice.yml` (forbidden words + tone/content regex rules)
and CI fails the PR on a brand-voice violation. The intent is sound and the
audience adaptation (YAML-externalized rules so non-coders edit config, not
Python) is the right call. But the prototype ships against the AGENTS.md
contract in three load-bearing ways and is partly non-functional:

1. It is a *checker* with **zero tests** (AGENTS.md 3i headline gap). Nothing
   proves the forbidden-word / tone / content-rule branches fire, and two
   latent defects (the inverted `content_rules` default, the dead
   `vocabulary.use` config) shipped precisely because no test exercised them.
2. The **CI gate is broken and fail-open**: it errors red on every normal PR
   (shallow checkout has no base SHA), and even when reached it validates only
   the first changed file and silently passes the `unknown`-type branch -- a
   green check that validated almost nothing.
3. The validator **crashes on hand-edited config** (KeyError on a missing rule
   key, TypeError on a commented-out section, AttributeError on an empty file)
   and **substring-matches forbidden words**, flagging clean ML/SaaS/finance
   prose ("non-disruptive", "Transformer architecture", "deleverage").

This slice is the **seed of a content-marketing checks lane** -- the Atlas
deterministic-review lane adapted for marketers, fighting brand-voice drift
across channels and over time. Because the prototype already exists, this is a
robust-testing-plus-fixes slice: add the missing failure-detection test layer
(the headline gap), then make the validator and its CI gate trustworthy. The
deeper "adopt from our lane" items (severity levels, committable suggested
fixes) are explicitly deferred so this slice stays under budget and lands a
working, tested gate first.

## Scope

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Robust testing

1. Add `tests/test_brand_voice_validator.py` -- the AGENTS.md 3i
   failure-detection suite: one negative fixture per detection branch asserting
   the SPECIFIC violation string fires, plus allowed near-miss fixtures that
   must still pass. Enroll it in CI execution (not just file presence).
2. Fix the ASCII-only convention violation: replace the two emoji in
   `brand_voice_validator.py` CLI output with ASCII markers ("FAIL:" / "PASS:").
3. Fix the CI gate end-to-end in `marketing_content_check.yml`: full-fetch so
   the base SHA exists (BLOCKER), loop over ALL changed marketing files instead
   of `head -n 1` (MAJOR), cover all four content types via directory mapping
   (MAJOR), turn the silent `unknown`-type skip into an explicit per-file
   continue / fail (MAJOR), `::set-output` -> `$GITHUB_OUTPUT`, and quote the
   file argument (NITs).
4. Harden the validator against malformed config and off-contract input:
   coerce `None` config/section to `{}`/`[]`, guard non-string `text`,
   `.get()`-with-fallback the `pattern`/`description` keys (skip a broken rule,
   do not crash the whole gate), fix the inverted `content_rules`
   `fail_on_match` default to match `tone_rules` (default True / fail-on-match),
   and catch regex-compile errors with a clear message naming the bad rule.
5. Switch forbidden-word matching from unanchored substring to per-word /
   stem-aware regex patterns sourced from the YAML, killing the prefix-glued
   false positives while keeping (and correctly catching) inflected forms.

### Files touched

- `.github/workflows/marketing_content_check.yml`
- `atlas_brain/services/brand_voice_validator.py`
- `atlas_brain/skills/brand/brand_voice.yml`
- `plans/PR-Content-Marketing-Brand-Voice-Checks.md`
- `tests/test_brand_voice_validator.py`

### Review Contract

Acceptance criteria (reviewer checks one-by-one):

- Every detection branch has a negative fixture proving it FIRES with the
  expected violation text: a forbidden word; a tone pattern ("!!", a casual
  phrase); a `landing_page` missing "extensibility"; a `release_notes` with
  "will be"; and a `content_rules` rule that omits `fail_on_match` (must ban,
  not invert).
- Every denylist/pattern change has an allowed near-miss that still PASSES:
  "non-disruptive" / "deleverage" / "Transformer architecture" do not trip the
  vocab list; a single "!"; an on-brand `landing_page` that mentions
  extensibility.
- Malformed-config tests prove the gate degrades safely, not fatally: `None`
  config, a commented-out (`None`) section, a rule missing `pattern` or
  `description`, and a bad regex each produce a clear error or a skipped rule --
  never an opaque KeyError/TypeError/AttributeError that disables the whole gate.
- CI: a fixture PR that violates brand voice fails the job; CI validates ALL
  changed marketing files (not just the first) and does not silently skip an
  unrecognized marketing path.
- ASCII gate is clean for the touched `.py` file.

Affected surfaces: the brand-voice validator service + CLI, its YAML schema,
and the marketing CI workflow. Risk areas: regex precision (over/under-match on
the vocab list), CI shell semantics (`set -e`/`pipefail` aborting on the diff),
and silent-pass paths. Reviewer rule triggers: checker-without-test (3i),
ASCII-only Python, fail-closed CI / surface-don't-skip drift (3g).

## Mechanism

Pure, deterministic, no-I/O scanner (unchanged shape): `validate(text,
content_type)` returns a `list[str]` of violations; the CLI exits 1 on any
violation, 0 on clean. The fixes are surgical:

- **Tests.** `tests/test_brand_voice_validator.py` constructs a
  `BrandVoiceValidator` from small inline YAML configs (via a tmp file or a
  config-dict seam) and asserts exact violation strings per branch, plus
  near-miss configs that must return `[]`. The suite runs under the repo's
  pytest in the marketing workflow's CI job (a `pytest tests/test_brand_voice_validator.py`
  step), so presence is backed by execution.
- **ASCII.** The two emoji literals in the CLI output (the cross-mark and
  check-mark) become `"FAIL:"` / `"PASS:"`.
- **Config hardening.** `self.config = yaml.safe_load(f) or {}` at load; each
  iteration uses `... or {}` / `... or []` so a present-but-`None` section
  defaults; `pattern = rule.get("pattern")` with `if not pattern: continue`,
  and `desc = rule.get("description", rule.get("id", "<unnamed>"))`; a missing
  `text` type raises a clear `TypeError`; patterns are `re.compile`-guarded with
  an error naming the offending rule id.
- **`fail_on_match` default.** Read once as `rule.get("fail_on_match", True)` on
  the `content_rules` path (matching the `tone_rules` path) so a forgotten key
  BANS rather than silently inverting to a must-contain check.
- **Word matching.** `vocabulary.avoid` becomes per-word regex patterns matched
  with `re.search(..., re.IGNORECASE)` (stem-aware entries, e.g.
  `revolutioniz(e|ed|ing)|revolutionary`, live in the YAML so the marketer owns
  over/under-match per word). This removes the prefix-glued FPs and catches
  inflections substring missed.
- **CI loop.** A single step replaces the Identify/Run pair: `git diff
  --name-only base head -- 'marketing/**'` (after `fetch-depth: 0`), a `case`
  per directory -> content type, validate each matched file, `continue` (with a
  log) on a non-marketing path, and `exit $status` so ANY violation fails the
  job.

## Intentional

- The YAML-externalized rule catalogue is kept as the design (not folded into
  Python like `safety_gate._PROHIBITED_PATTERNS`): the entire point of this lane
  is that marketers edit config, not code.
- Return type stays a flat `list[str]` with binary pass/fail this slice.
  Structured severity (BLOCKER/MAJOR/NIT) is a real adopt-from-our-lane
  improvement but is its own slice (see Deferred); landing a working, tested
  gate first is the priority.
- Word-boundary matching is NOT done by a blanket `\b` wrap. That is only a
  partial fix -- `\bdisrupt` still flags "non-disruptive" (the hyphen is a word
  boundary) and `\b<word>` still misses stem changes like "synergistic" /
  "utilizing". The robust fix is per-word / stem patterns owned in the YAML; the
  residual precision risk (a marketer writing an over-broad pattern) is locked
  by allowed-near-miss fixtures rather than hidden.
- A broken individual rule (missing key, bad regex) is surfaced (skip-with-log
  or clear load-time error), never silently no-op'd into a green pass --
  mirroring the "auditors surface, never skip" discipline (AGENTS.md 3g).
- The validator file stays at `atlas_brain/services/brand_voice_validator.py`
  for this slice; the pure-module + thin `scripts/check_*.py` split is deferred
  (tests can import `BrandVoiceValidator` directly without it, and keeping the
  file in place holds the diff budget).

## Deferred

- Service + thin-entrypoint split (pure importable validator under a module +
  `scripts/check_brand_voice.py` CLI) and a `CANONICAL.md` note clarifying the
  relationship to `extracted_content_pipeline.brand_voice` -- follow-up slice
  (placement/naming, AGENTS.md 3h).
- Structured findings carrying a `GateSeverity` enum (BLOCKER/MAJOR/NIT)
  replacing the flat `list[str]` -- adopt-from-our-lane slice.
- Committable suggested-fix output: implement `vocabulary.use`
  (preferred -> discouraged) so a discouraged word surfaces its replacement as
  an advisory, returned/tagged separately from violations -- adopt-from-our-lane
  slice. Until then the `vocabulary.use` block is relabeled FUTURE in the YAML
  so the comment stops implying it is active.
- `tone_rules` polarity symmetry (handling `fail_on_match: false` to REQUIRE a
  phrase) or an explicit load-time reject -- pairs naturally with the unified
  rule evaluator in the severity slice.
- Standing up a real `marketing/` content directory
  (landing_pages/blog_posts/release_notes/tweets) with seed copy, and aligning
  the CLI `--type` choices to the directories that exist -- content/scaffold
  slice that follows this enforcement slice.
- Scraping / deriving an established-voice profile to seed `brand_voice.yml`
  from existing on-brand copy -- separate research slice.
- Bumping pinned action versions (checkout@v4, setup-python@v5) beyond the
  minimum needed for the fetch-depth fix -- cosmetic, folded into a later touch.

Parked hardening: none.

## Verification

Commands the implementation will run locally before push (no counts asserted --
the suite does not exist yet):

- `pytest tests/test_brand_voice_validator.py -v` -- the new failure-detection
  suite (per-branch negative fixtures + allowed near-misses + malformed-config
  guards) all green.
- `bash scripts/check_ascii_python.sh` -- ASCII gate clean for the edited
  validator (and confirm a non-ASCII byte scan of the touched `.py` is empty).
- `python -c "from pathlib import Path; from atlas_brain.services.brand_voice_validator import BrandVoiceValidator"`
  -- import sanity.
- Manual CLI smoke against the shipped `brand_voice.yml`: a clean
  landing-page-style fixture exits 0; a fixture with "game-changer" / "!!" /
  missing "extensibility" exits 1 with the expected lines; a known-FP fixture
  ("non-disruptive migration", "Transformer architecture") exits 0.
- `actionlint .github/workflows/marketing_content_check.yml` (if available) plus
  a local replay of the diff/loop logic to confirm it iterates all changed files
  and fails on any violation.

## Estimated diff size

| Component | LOC |
|---|---:|
| `tests/test_brand_voice_validator.py` (new) | ~150 |
| `atlas_brain/services/brand_voice_validator.py` (hardening + word-boundary + ASCII) | ~55 |
| `.github/workflows/marketing_content_check.yml` (fetch-depth, per-file loop, type coverage, output API) | ~40 |
| `atlas_brain/skills/brand/brand_voice.yml` (stem patterns + relabel `use` as FUTURE) | ~25 |
| `plans/PR-Content-Marketing-Brand-Voice-Checks.md` | ~40 |
| **Total** | **~310** |
