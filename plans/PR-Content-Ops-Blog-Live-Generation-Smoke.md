# PR: Content Ops Blog Live Generation Smoke

## Why this slice exists

PR #829 proved the live Content Ops landing-page path end to end: Atlas DB pool,
pipeline-routed OpenRouter/Claude provider, packaged skill prompt, executor,
quality gate, and persisted draft. Blog generation is wired through the same
host services bundle, but the live path cannot be proved against an empty
`blog_blueprints` table.

This slice adds the thinnest real blog proof: seed one tenant-scoped blueprint,
run `blog_post` through the same live executor path, and verify a saved
`blog_posts` draft. While scoping the slice, the blog prompt/gate contract also
showed a source mismatch: the prompt asked for 800-1500 words while the quality
gate blocks under 1500 words. The live smoke should prove quality gates on, so
this slice aligns the prompt with the gate instead of disabling quality checks.

## Scope (this PR)

Ownership lane: content-ops/blog-live-generation-smoke

1. Extend `scripts/smoke_content_ops_live_generation.py` with an
   `--output blog_post` mode.
2. Seed one scoped `blog_blueprints` row before blog execution, using the
   existing Postgres blueprint repository and a smoke-only topic-type filter so
   the run consumes the row it just seeded.
3. Keep the existing landing-page smoke behavior as the default.
4. Verify configured services, execution status, saved ids, and seeded blueprint
   ids in the smoke result.
5. Align the blog generation prompt's word-count instruction with the existing
   blog quality gate.
6. Add CI-safe tests with fake services and a fake blueprint seed function.
7. Document both live smoke modes in the README and host runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Blog-Live-Generation-Smoke.md` | Plan doc for this slice. |
| `scripts/smoke_content_ops_live_generation.py` | Add blog output mode, seeded blueprint setup, and output-aware validation. |
| `tests/test_smoke_content_ops_live_generation.py` | Add blog smoke coverage while preserving landing-page coverage. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Align blog prompt length rule with the quality gate. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Synced extracted skill copy. |
| `extracted_content_pipeline/README.md` | Document the blog live-generation smoke command. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Add the same operator runbook guidance. |

## Mechanism

The smoke command keeps `landing_page` as the default:

```bash
python scripts/smoke_content_ops_live_generation.py --account-id acct_123
```

Blog mode is explicit:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_123 \
  --env-file /path/to/Atlas/.env \
  --json
```

For `blog_post`, the command initializes the Atlas DB pool, builds the normal
`build_content_ops_execution_services(enable_db_services=True)` bundle, checks
that `blog_post` is configured, upserts one default blueprint through
`PostgresBlogBlueprintRepository.save_blueprints()`, and then calls
`execute_content_ops_from_mapping()` with:

```python
{
    "outputs": ["blog_post"],
    "target_mode": args.target_mode,
    "limit": 1,
    "require_quality_gates": True,
    "inputs": {"topic": "..."},
}
```

The default blueprint is scoped to the account id and uses a smoke-specific
slug so repeated runs update the same smoke row for that account instead of
creating a new unbounded queue. Blog mode also passes
`inputs.filters.topic_type=content_ops_live_smoke`, matching the seeded
blueprint's topic type, so newer unrelated unconsumed blueprints under the same
target mode cannot steal the smoke run. Because the generator only persists
drafts after passing the blog quality gate, a saved id with quality gates
enabled is the live quality proof.

## Intentional

- One shared smoke script. The landing-page and blog modes exercise the same
  host bundle and executor, so a second script would duplicate setup and
  failure handling.
- Blog seed is default for `--output blog_post`. The live database currently has
  no blueprints, and seeding one row makes the smoke self-contained for
  operators.
- The default seed does not include source quotes or chart requirements. Those
  are already covered in generator-level tests; the live smoke's job is to
  prove provider/DB/executor/persistence with quality gates on.
- The prompt length fix is included because the smoke would otherwise document
  an operator path where the model is asked to produce output the quality gate
  can reject.

## Deferred

- HTTP-level `/content-ops/execute` authenticated smoke remains a future PR
  because it needs a real B2B session and route auth setup.
- Custom operator-supplied blog blueprint JSON is deferred. The default seed is
  enough to prove live wiring; custom fixtures can be added after this path is
  stable.
- Parked hardening: existing `ATLAS-HARDENING.md` items are for the older
  Atlas blog/deep-dive content pipeline, not this Content Ops
  `blog_blueprints` smoke path. They remain parked.

## Verification

- `pytest tests/test_smoke_content_ops_live_generation.py -q` -> 5 passed.
- `python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py` -> passed.
- `pytest tests/test_smoke_content_ops_live_generation.py tests/test_extracted_blog_generation.py -q` -> 32 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -> passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -> passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed.
- `bash scripts/check_ascii_python.sh` -> passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` -> refreshed 43 mapped files; source and extracted blog skill copies verified identical.
- `python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_content_ops_blog_smoke --user-id codex-smoke --json` -> failed clearly before execution: `blog_post service is not configured`; this worktree has no provider key loaded.
- `python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_content_ops_blog_smoke --user-id codex-smoke --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --json` -> passed; configured all LLM-backed outputs, seeded blueprint id `0b04a34c-0876-4255-ba6b-bfd31b90b5d5`, filtered execution to `topic_type=content_ops_live_smoke`, generated 1 blog post, quality gate passed, saved draft id `989ef0f9-060b-47c3-bf49-847614833212`.
- Saved draft spot-check -> `blog_posts` row found for account `acct_content_ops_blog_smoke`, status `draft`, slug `content-ops-blog-live-smoke-acct-content-ops-blog-smoke`, target keyword `support ticket FAQ gaps`.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1826 passed, 1 skipped.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~130 |
| Smoke script | ~197 |
| Tests | ~68 |
| Prompt sync | ~8 |
| Docs | ~31 |
| **Total** | **~434** |

This may land slightly over the 400 LOC soft cap because the slice needs the
operator path, fake-backed CI coverage, docs, and the prompt/gate source fix to
make the live smoke meaningful with quality gates enabled.
