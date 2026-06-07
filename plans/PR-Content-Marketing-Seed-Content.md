# PR-Content-Marketing-Seed-Content

## Why this slice exists

`PR-Content-Marketing-Brand-Voice-Checks` stood up a brand-voice checker and a
CI gate that watches `marketing/**`, but the `marketing/` directory does not
exist yet -- so the gate is dormant (it never has a file to validate). This
slice creates a real `marketing/` tree with on-brand seed copy for each content
type, which activates the gate and gives marketers a worked example to copy.

It is the content/scaffold follow-on the checker plan explicitly deferred.

## Scope

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Product polish

1. Create `marketing/{landing_pages,blog_posts,release_notes,tweets}/`, each
   with one on-brand seed file.
2. Add `marketing/README.md`: how the gate works, the type-to-directory map, how
   to edit the rules, and how to check a file locally.
3. Every seed file is validated PASS against the shipped `brand_voice.yml`; a
   negative control confirms the gate still bites a bad edit.

### Files touched

- `marketing/README.md`
- `marketing/landing_pages/atlas-platform.md`
- `marketing/blog_posts/why-deterministic-checks.md`
- `marketing/release_notes/2026-06-release.md`
- `marketing/tweets/launch-brand-voice-checks.md`
- `plans/PR-Content-Marketing-Seed-Content.md`

## Mechanism

Pure content (markdown) under the paths the `Marketing Content Voice Check`
workflow watches. No code change. Each file is written to satisfy its content
type's rules: the landing page mentions extensibility; the release note avoids
future tense; all of them avoid the forbidden vocabulary, double punctuation,
and casual phrases. The README points marketers at `brand_voice.yml` as the
single editable source of truth.

## Intentional

- Copy is written to the Atlas "The Guide" persona (plain, expert, no hype),
  which is the brand voice in `brand_voice.yml` -- not the operator's personal
  voice.
- One seed file per type: enough to activate the gate and demonstrate each rule,
  not a full content library.
- `README.md` lives at `marketing/README.md` (outside a typed subdir) so it is
  not itself type-validated, and a PR touching only it does not trip the gate.

## Deferred

- A real ongoing content corpus -- this is a seed, not the library.
- Aligning the validator `--type` choices with the directory set if new channels
  are added later.
- The adopt-from-our-lane items stay in `PR-Content-Marketing-Brand-Voice-Checks`
  (severity levels, suggested-fix output, stem-aware vocabulary).

Parked hardening: none.

## Verification

- `python atlas_brain/services/brand_voice_validator.py --file <each seed>
  --type <type>` -- all four exit 0 (PASS).
- Negative control: a release note containing "will be" / "game-changer" / "!!"
  exits 1 with the future-tense, forbidden-word, and tone rules.
- No `.py` touched (content only), so the ASCII gate is unaffected.

## Estimated diff size

| File | LOC |
|---|---:|
| `marketing/README.md` | 29 |
| `marketing/landing_pages/atlas-platform.md` | 17 |
| `marketing/blog_posts/why-deterministic-checks.md` | 12 |
| `marketing/release_notes/2026-06-release.md` | 14 |
| `marketing/tweets/launch-brand-voice-checks.md` | 4 |
| `plans/PR-Content-Marketing-Seed-Content.md` | 81 |
| **Total** | **~157** |
