# Marketing content

On-brand marketing copy, checked automatically on every pull request.

## How it works

Each file under `marketing/<type>/` is validated against the codified Atlas
brand voice in `atlas_brain/skills/brand/brand_voice.yml` by the
`Marketing Content Voice Check` workflow. A pull request that introduces a
`BLOCKER` or `MAJOR` brand-voice violation fails the check and names the
specific rule that fired. `NIT` findings print as advisory warnings with
suggested fixes, but do not fail the workflow.

| Directory | Type | Extra rule |
|---|---|---|
| `marketing/landing_pages/` | landing_page | Must mention extensibility |
| `marketing/blog_posts/` | blog_post | Vocabulary + tone only |
| `marketing/release_notes/` | release_notes | No future tense for shipped work |
| `marketing/tweets/` | tweet | Vocabulary + tone only |

## Editing the rules

Marketers own the brand voice. To change what is allowed -- forbidden words,
tone patterns, content rules -- edit `atlas_brain/skills/brand/brand_voice.yml`.
No code change is required.

## Checking locally

    python atlas_brain/brand/voice_validator.py \
      --file marketing/landing_pages/atlas-platform.md \
      --type landing_page

Use strict mode when advisory `NIT` findings should fail the local run:

    python atlas_brain/brand/voice_validator.py \
      --file marketing/blog_posts/why-deterministic-checks.md \
      --type blog_post \
      --strict
