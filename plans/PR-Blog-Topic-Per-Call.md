# PR: thread `topic` per-call kwarg through BlogPostGenerationService

## Why this slice exists

Closes the "topic for blog_post" item that was deferred across the
OptionA series. The plan layer emits ``step.config["topic"]`` from
``request.inputs.get("topic")``, but the blog service has no service-
side surface that consumes it -- the kwarg gets dropped at the
executor and the prompt never sees the operator's topic.

This PR:
1. Adds a ``{topic}`` placeholder to the blog skill prompt.
2. Adds ``topic: str | None = None`` to
   ``BlogPostGenerationService.generate()``.
3. Threads the resolved value through ``_generate_one`` into the
   prompt-template substitution.
4. Routes ``step.config["topic"]`` through the executor's blog
   dispatcher.

After this lands the operator's per-call topic actually steers the
LLM. Same per-field-kwarg shape as OptionA-1 through OptionA-5.

## Scope (this PR)

Per-call ``topic`` kwarg only. No new abstraction; no other blog
service surface changes. Skill prompt change is additive: a single
``{topic}`` line at the input-context section.

### Files touched

1. ``extracted_content_pipeline/skills/digest/blog_post_generation.md``
   -- add ``{topic}`` placeholder.
2. ``extracted_content_pipeline/blog_generation.py`` -- new kwarg
   on ``generate()``; resolved + threaded to ``_generate_one``;
   prompt template substitution at the existing
   ``{blueprint_json}`` site (or appended if no placeholder).
3. ``extracted_content_pipeline/content_ops_execution.py`` -- blog
   dispatcher gains ``topic=_step_config_text(step.config, "topic")``.
4. Test fakes in ``tests/test_extracted_content_ops_execution.py``
   (the ``_OpportunityService`` fake) gain ``topic`` as a named
   kwarg so the existing ``extras == {}`` assertions stay accurate.
5. New tests in ``tests/test_extracted_blog_generation.py`` and
   ``tests/test_extracted_content_ops_execution.py``.

## Mechanism

```python
# Service signature (added kwarg)
async def generate(
    self, *, scope, target_mode, limit=None, filters=None,
    # ... existing OptionA-2/3/4/5 kwargs ...
    topic: str | None = None,
) -> BlogPostGenerationResult:
    ...
    resolved_topic = (topic or "").strip()  # empty string when None
    ...
    parsed = await self._generate_one(
        prompt_template,
        blueprint=blueprint,
        target_mode=target_mode,
        # ... resolved kwargs ...
        topic=resolved_topic,
    )

# _generate_one (new param)
async def _generate_one(
    self, prompt_template, *, blueprint, target_mode, ..., topic: str,
) -> dict[str, Any] | None:
    system_prompt = prompt_template.replace("{topic}", topic)
    ...
```

Prompt placeholder: empty topic resolves to ``""`` (silent no-op when
the operator doesn't supply one). Hosts on the existing skill prompt
without ``{topic}`` are unaffected -- the ``replace()`` no-ops.

## Intentional (looks wrong but is deliberate)

- **Resolved topic is a stripped string, not Optional[str].** The
  service uses it for prompt substitution; ``None`` would propagate
  weird types. Empty string is the clean "no override" sentinel for
  prompt text.
- **No fallback to ``blueprint.get("topic")``.** The blueprint's
  ``topic`` field is an internal blog-blueprint concept (used for
  ``_blueprint_id`` only, see ``blog_generation.py`` L378).
  ``request.inputs["topic"]`` is operator-supplied per-call; mixing
  the two surfaces would conflate unrelated concerns.
- **Skill prompt change is additive.** Adding a ``{topic}``
  placeholder doesn't break hosts on the prior version of the
  prompt. If they pull updated skills, they get the new behavior;
  if they don't, ``replace("{topic}", topic)`` is a no-op on their
  unchanged prompt.
- **No `default_topic` config field on
  ``BlogPostGenerationConfig``.** Topic is per-call, not a
  configuration knob. Operators set it per generation, not per
  service-startup.

## Deferred (still on purpose)

- Reasoning context wiring on the new control-surface execute route
  -- separate slice (PR-Reasoning-Wiring-1).
- ``PR-Campaign-Config-V2`` (breaking change to remove the legacy
  ``channel`` field).
- Last few audit MINORs not yet touched.

## Verification

- ``pytest tests/test_extracted_blog_generation.py
  tests/test_extracted_content_ops_execution.py
  tests/test_extracted_content_generation_plan.py`` -> all passing
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``bash scripts/check_ascii_python.sh`` -> passed
