# PR-Audit-MINOR-Batch-2: parser strictness + slug length cap (blog_generation)

## Why this slice exists

Two related audit MINORs in ``blog_generation.py``. Several other
audit findings I considered for this batch turned out to be already
addressed by intermediate PRs (the API status-code split / provider
exception wrapping / scope sequence stripping all landed in
later commits). This batch closes the two genuinely-still-open
findings in the blog generator.

1. **MINOR — ``parse_blog_post_response`` rejects valid JSON missing
   title or content.** The parser requires BOTH ``title`` and
   ``content`` to be non-empty before returning a candidate. If the
   LLM emits ``{"title": "..."}`` but no ``content`` (or vice versa),
   the parser returns ``None`` and the executor reports
   ``unparseable_response``. A specific blocker would be more useful.
   The audit's prescribed fix matches the
   ``landing_page_generation.py`` pattern from PR-OptionA-1: parser
   identifies "is this a candidate" minimally; the quality pack
   judges the rest.
2. **MINOR — Slug has no length limit.** ``_slugify`` returns
   whatever the input title-or-fallback produces. A 2000-char title
   becomes a 2000-char slug. Most CMS / URL routing expects slugs
   ``<100`` chars.

## Scope (this PR)

Two small fixes in ``extracted_content_pipeline/blog_generation.py``
plus regression tests.

### Fix 1: relax parser to ``if title:``

```python
# Before
title = str(decoded.get("title") or "").strip()
content = str(decoded.get("content") or "").strip()
if title and content:
    return {**dict(decoded), "title": title, "content": content}

# After
title = str(decoded.get("title") or "").strip()
if title:
    return {**dict(decoded), "title": title}
```

Empty ``content`` now flows to the quality pack which already fires
``content_too_short`` when the body is empty (zero words is below
``min_words`` regardless of threshold). Operators see a precise
blocker instead of ``unparseable_response``.

The parser still requires ``title`` because that's the minimum
"is this a blog-post candidate" filter -- otherwise any well-formed
JSON object could be picked up.

### Fix 2: cap slug length at 100

```python
_MAX_SLUG_CHARS = 100  # CMS / URL convention; same shape as SEO meta limit

def _slugify(value: Any) -> str:
    text = str(value or "blog-post").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(slug) > _MAX_SLUG_CHARS:
        slug = slug[:_MAX_SLUG_CHARS].rstrip("-")
    return slug or "blog-post"
```

Trailing hyphen strip after truncation prevents ``"a-very-long-..."``
becoming ``"a-very-long--"``.

## Intentional (looks wrong but is deliberate)

- **Parser keeps ``title`` as the minimum filter.** Could relax
  further (any well-formed JSON object), but that risks picking up
  unrelated objects in the response (e.g., a metadata block before
  the actual blog draft). Title is the cheapest "this is a blog
  candidate" signal.
- **100-char slug cap is the standard convention.** Not configurable
  per call -- that would be plumbing without a real driver. If a
  host wants a different cap, they can post-process the slug
  before persistence.
- **Truncation rather than rejection.** A 200-char title is still
  a valid blog post; the slug just gets clipped. Rejection would
  drop drafts unnecessarily.

## Deferred (still on purpose)

- ``_accumulate_usage`` cumulative-usage assumption (audit MINOR).
- ``_landing_page_config_for_request`` discards request (audit
  MINOR).
- Host concurrency assumption documentation (audit MINOR).
- ``for_output`` if/elif chain (audit NIT).
- ``describe_control_surfaces`` calls execution-services per request
  (audit MINOR) -- needs separate caching design.
- ``topic`` for blog_post.
- ``PR-Campaign-Config-V2``.

## Verification

- ``pytest tests/test_extracted_blog_generation.py`` -> all passing
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``bash scripts/check_ascii_python.sh`` -> passed

## Sibling references

- Audit doc:
  ``docs/audits/ai_content_ops_post_merge_audit_2026-05.md``
- Same parser-strictness pattern fixed in PR-OptionA-1 (review
  finding on ``parse_landing_page_response``).
