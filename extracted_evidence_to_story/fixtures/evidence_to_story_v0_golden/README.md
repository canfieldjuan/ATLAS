# Evidence-to-Story v0 — Golden Fixture

Reference fixture for the v0 build contract at
[`docs/evidence_to_story_v0_build_contract.md`](../../docs/evidence_to_story_v0_build_contract.md).

> **Parked alongside the build contract.** The fixture is a scaffold
> until v0 work resumes. The case content (transcript + article + expected
> outputs) is deliberately left blank because picking the case is a
> human-judgment call that affects every downstream design decision.

## Structure

```
evidence_to_story_v0_golden/
├── README.md                  ← this file
├── inputs/
│   ├── manifest.json          ← Stage-1 input contract (skeleton)
│   ├── youtube_transcript.txt ← TBD — paste real transcript
│   └── news_article.txt       ← TBD — paste real article text
└── expected/
    ├── sources.json           ← Stage-1 expected output (placeholder)
    ├── claims.json            ← Stage-2
    ├── timeline.json          ← Stage-3
    ├── entities.json          ← Stage-4
    ├── angles.json            ← Stage-5
    ├── selected_outline.json  ← Stage-6
    ├── script.md              ← Stage-7
    ├── voice_direction.json   ← Stage-9
    └── validation_report.json ← Stage-8
```

## How to populate the fixture

When v0 work resumes:

1. **Pick the case.** Apply the selection criteria from §7 of the build
   contract. The case must have:
   - At least 1 long-form YouTube treatment
   - At least 1 reputable news article on the same case
   - A clear, factual timeline of events
   - A clear final reveal/outcome
   - No active legal contest
   - No minors involved
   - No top-of-mind notoriety (skip Wikipedia front-page cases)
   - Real source material (don't synthesize)

2. **Fill in `inputs/`.**
   - `inputs/manifest.json` — set `story_id`, source titles/URLs, point
     `text_path` at the two transcript/article files.
   - `inputs/youtube_transcript.txt` — paste the transcript verbatim.
     Preserve speaker turns and timestamps if present.
   - `inputs/news_article.txt` — paste the article body. Strip ads /
     navigation. Keep paragraph breaks.

3. **Run the v0 pipeline once authoritatively.** Manually inspect every
   stage output. Edit if needed. Save the approved outputs under
   `expected/` — they become the diff target for future runs.

4. **Lock the fixture.** Add a `FIXTURE_LOCKED` marker file (`touch
   FIXTURE_LOCKED`) once `expected/` has been reviewed end-to-end.

## How CI uses the fixture

v0 does not gate CI on full-text equality (LLM nondeterminism). CI
checks:

- Every file in `expected/` has a matching file in the test run output.
- JSON shapes match (same top-level keys, same per-record key set).
- Counts are within tolerance (e.g. `claims.json` has between
  `0.7×` and `1.3×` the expected number of claims of each type).
- Every acceptance test (§9 of the build contract) returns the same
  `pass | warn | fail` verdict on the fixture run as on the locked
  expected run.

A drift in any of those fails CI and surfaces a per-file diff.

## Why the fixture is in the repo (not a separate dataset)

The fixture is part of the build contract. Without it, the contract is
unverifiable. Storing it next to the spec doc keeps the round-trip
short: change the contract → update the fixture → catch breakage in
the same PR.

If the case turns out to involve sensitive material that shouldn't sit
in a public repo (private GH project; or this fixture moves to a
dedicated repo), revisit before locking.

## Resume condition

This fixture is dormant until the campaign-core spine is product-owned
per `remaining_productization_audit.md` "Next Concrete Slice".
