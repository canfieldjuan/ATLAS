# PR: Content Ops Generated Asset Reasoning Usage

## Goal

Bring the actual reasoning usage count added for email campaigns to the other
generated-asset services that already consume host reasoning context.

## Scope

- Add `reasoning_contexts_used` to blog post, report, landing page, and sales
  brief generation result dictionaries.
- Count only successful generated assets whose prompt payload carried
  `campaign_reasoning_context`.
- Update existing reasoning-provider tests to assert the counter.
- Update the frontend contract to document the shared generated-asset summary
  field.

## Non-Goals

- Do not change reasoning provider lookup or prompt payload construction.
- Do not expose raw reasoning context in execution results.
- Do not add new UI components; the existing generated-asset summary already
  renders `reasoning_contexts_used` when present.

## Verification

- `python -m py_compile` on the four edited generation modules.
- Focused generated-asset service test suites.
- `git diff --check`.
