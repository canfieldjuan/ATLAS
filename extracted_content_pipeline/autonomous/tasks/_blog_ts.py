"""Shared helpers for writing blog .ts files and updating index.ts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import markdown as _md

_md_converter = _md.Markdown(extensions=["tables", "fenced_code", "toc"])

logger = logging.getLogger(__name__)


def escape_js_single(text: str) -> str:
    """Escape a string for use inside JS single quotes."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def escape_template_literal(text: str) -> str:
    """Escape a string for use inside JS template literals (backtick strings)."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )


def slug_to_var_name(slug: str) -> str:
    """Convert a blog slug to a valid camelCase JS variable name.

    Examples:
        'my-blog-post'       -> 'myBlogPost'
        '2026-report'        -> 'post2026Report'
        'vendor.comparison'  -> 'vendorComparison'
    """
    var_name = re.sub(r"[^a-zA-Z0-9]", "_", slug).strip("_")
    parts = var_name.split("_")
    var_name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    # JS identifiers cannot start with a digit
    if var_name and var_name[0].isdigit():
        var_name = "post" + var_name[0].upper() + var_name[1:]
    return var_name


def update_blog_index(index_path: Path, slug: str, var_name: str) -> bool:
    """Add an import and POSTS entry to the blog index.ts file.

    Returns True if the index was updated, False if skipped (already present
    or file not found).
    """
    if not index_path.exists():
        logger.warning("index.ts not found: %s", index_path)
        return False

    index_text = index_path.read_text(encoding="utf-8")

    # Dynamic glob discovery -- posts are auto-discovered by filename pattern,
    # so no static import or array entry is needed.
    if "import.meta.glob" in index_text:
        logger.debug("index.ts uses dynamic glob; skipping static registration for %s", slug)
        return False

    import_line = f"import {var_name} from './{slug}'"

    # Already registered
    if slug in index_text:
        logger.debug("Post %s already in index.ts, skipping", slug)
        return False

    # Insert import after last existing import statement
    lines = index_text.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("import "):
            last_import_idx = i

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_line)
    else:
        lines.insert(0, import_line)

    # Insert into POSTS array -- handle both `].sort(` and plain `]` endings
    new_text = "\n".join(lines)
    if re.search(r"].sort\(", new_text):
        new_text = re.sub(
            r"(].sort\()",
            f"  {var_name},\n\\1",
            new_text,
            count=1,
        )
    else:
        # Plain array ending: `]\n` or `]\n\n` at end of POSTS
        new_text = re.sub(
            r"(?m)^(\])\s*$",
            f"  {var_name},\n\\1",
            new_text,
            count=1,
        )

    index_path.write_text(new_text, encoding="utf-8")
    logger.info("Updated index.ts with %s", slug)
    return True


def build_post_ts(
    slug: str,
    title: str,
    description: str,
    date_str: str,
    author: str,
    tags: list[str],
    topic_type: str,
    charts_json: list[dict],
    content: str,
    data_context: dict | None = None,
    seo_title: str = "",
    seo_description: str = "",
    target_keyword: str = "",
    secondary_keywords: list[str] | None = None,
    faq: list[dict] | None = None,
    related_slugs: list[str] | None = None,
    cta: dict | None = None,
) -> tuple[str, str]:
    """Build a complete .ts file for a blog post.

    Returns (var_name, ts_content).
    """
    var_name = slug_to_var_name(slug)
    charts_str = json.dumps(charts_json, indent=2, default=str)
    # Render markdown to HTML at deploy time so the frontend never parses markdown
    _md_converter.reset()
    html_content = _md_converter.convert(content)
    escaped_content = escape_template_literal(html_content)
    escaped_title = escape_js_single(title)
    escaped_desc = escape_js_single(description)

    # Only include affiliate-relevant fields in data_context to keep .ts files lean
    dc_output = {}
    if data_context:
        if data_context.get("affiliate_url"):
            dc_output["affiliate_url"] = data_context["affiliate_url"]
        if data_context.get("affiliate_partner"):
            dc_output["affiliate_partner"] = data_context["affiliate_partner"]
        if data_context.get("booking_url"):
            dc_output["booking_url"] = data_context["booking_url"]
    dc_str = json.dumps(dc_output, indent=2, default=str)

    # Build optional SEO fields
    seo_lines = ""
    if seo_title:
        seo_lines += f"  seo_title: '{escape_js_single(seo_title)}',\n"
    if seo_description:
        seo_lines += f"  seo_description: '{escape_js_single(seo_description)}',\n"
    if target_keyword:
        seo_lines += f"  target_keyword: '{escape_js_single(target_keyword)}',\n"
    if secondary_keywords:
        seo_lines += f"  secondary_keywords: {json.dumps(secondary_keywords)},\n"
    if faq:
        faq_str = json.dumps(faq, indent=2, default=str)
        seo_lines += f"  faq: {faq_str},\n"
    if related_slugs:
        seo_lines += f"  related_slugs: {json.dumps(related_slugs)},\n"
    if cta:
        seo_lines += f"  cta: {json.dumps(cta, indent=2, default=str)},\n"

    ts_content = f"""import type {{ BlogPost }} from './index'

const post: BlogPost = {{
  slug: '{slug}',
  title: '{escaped_title}',
  description: '{escaped_desc}',
  date: '{date_str}',
  author: '{escape_js_single(author)}',
  tags: {json.dumps(tags)},
  topic_type: '{topic_type}',
  charts: {charts_str},
  data_context: {dc_str},
{seo_lines}  content: `{escaped_content}`,
}}

export default post
"""
    return var_name, ts_content
