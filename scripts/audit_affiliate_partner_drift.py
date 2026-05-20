#!/usr/bin/env python3
"""Reconcile live affiliate_partners rows against migration-seeded definitions.

Closes the second half of affiliate-system-investigation item #2: migration
326 version-controlled the existing partners, but the /b2b/tenant/affiliates
API can still create partner rows directly in the DB, re-introducing
ungoverned data with no migration, no review, and no disaster-recovery path.

This audit detects that drift:

  * FAIL  -- a live partner whose product_name is not seeded by any migration
            (the core "every partner must be version-controlled" rule).
  * WARN  -- a partner that IS seeded, but whose migration definition diverges
            from the live row (URL / commission / aliases / category / notes
            edited in the DB after seeding). Surfaces post-seed API edits
            without blocking; the operator decides whether the DB or the
            migration is authoritative.
  * WARN  -- a migration seeds a product_name with no live row (seeded then
            deleted/disabled-and-removed). Informational.

`enabled` is intentionally excluded from divergence: toggling a partner on/off
is legitimate operational state, not a drift signal.

The parser and reconciler are pure (no DB) so they are unit-tested directly;
only `_run` touches the live database (the same :5433/atlas the app uses).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.database import close_database, get_db_pool, init_database

MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"

# Business columns reconciled for divergence. product_name is the key (not a
# value); enabled is operational state, not a definitional fact.
_DIVERGENCE_FIELDS = (
    "name",
    "category",
    "affiliate_url",
    "commission_type",
    "commission_value",
    "notes",
    "product_aliases",
)


# ---------------------------------------------------------------------------
# SQL parsing (pure -- no DB)
# ---------------------------------------------------------------------------


def _read_paren_group(s: str, open_idx: int) -> tuple[str, int]:
    """Return (inner_text, index_after_close) for the parenthesized group whose
    opening '(' is at open_idx. Respects single-quoted strings (with '' escape)
    and nested parens; bracket pairs ([]) do not affect paren depth."""
    if s[open_idx] != "(":
        raise ValueError("expected '(' at open_idx")
    depth = 0
    j = open_idx
    in_str = False
    while j < len(s):
        c = s[j]
        if in_str:
            if c == "'":
                if j + 1 < len(s) and s[j + 1] == "'":
                    j += 2
                    continue
                in_str = False
            j += 1
            continue
        if c == "'":
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : j], j + 1
        j += 1
    raise ValueError("unbalanced parentheses")


def _split_top_level(s: str) -> list[str]:
    """Split on commas at paren/bracket depth 0, respecting single-quoted
    strings (with '' escape)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_str:
            buf.append(c)
            if c == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    buf.append(s[i + 1])
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            buf.append(c)
        elif c in "([":
            depth += 1
            buf.append(c)
        elif c in ")]":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_pg_array(literal: str) -> list[str]:
    """Parse a Postgres array literal body like {a,"b c","d e"} into a list.
    `literal` includes the surrounding braces."""
    inner = literal[1:-1]
    if not inner.strip():
        return []
    items: list[str] = []
    buf: list[str] = []
    in_q = False
    quoted = False
    i = 0
    while i < len(inner):
        c = inner[i]
        if in_q:
            if c == "\\" and i + 1 < len(inner):
                buf.append(inner[i + 1])
                i += 2
                continue
            if c == '"':
                in_q = False
                i += 1
                continue
            buf.append(c)
            i += 1
            continue
        if c == '"':
            in_q = True
            quoted = True
            i += 1
            continue
        if c == ",":
            val = "".join(buf)
            items.append(val if quoted else val.strip())
            buf = []
            quoted = False
            i += 1
            continue
        buf.append(c)
        i += 1
    val = "".join(buf)
    items.append(val if quoted else val.strip())
    return items


def _parse_value(tok: str) -> Any:
    """Normalize a single SQL value token to a Python value."""
    tok = tok.strip()
    # Strip a trailing ::type or ::type[] cast.
    tok = re.sub(r"::\s*[a-zA-Z_]+(\s*\[\s*\])?\s*$", "", tok).strip()
    low = tok.lower()
    if low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if tok.startswith("'") and tok.endswith("'") and len(tok) >= 2:
        unescaped = tok[1:-1].replace("''", "'")
        if unescaped.startswith("{") and unescaped.endswith("}"):
            return _parse_pg_array(unescaped)
        return unescaped
    if tok.upper().startswith("ARRAY[") and tok.rstrip().endswith("]"):
        inner = tok[tok.index("[") + 1 : tok.rindex("]")]
        return [_parse_value(it) for it in _split_top_level(inner)]
    return tok


def parse_seeded_partners(sql: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract every affiliate_partners row seeded in a migration's SQL.

    Returns (rows, errors). Each row is a dict of column -> Python value.
    Handles multi-row inserts (`VALUES (...), (...), ...`) -- every tuple is
    read, not just the first. A column/value count mismatch is NOT silently
    skipped: it is recorded in `errors` so the caller can surface it as a
    failure rather than reconciling against silently-dropped definitions.
    """
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for m in re.finditer(
        r"INSERT\s+INTO\s+affiliate_partners\s*\(", sql, re.IGNORECASE
    ):
        cols_str, after_cols = _read_paren_group(sql, m.end() - 1)
        vm = re.compile(r"\s*VALUES\s*", re.IGNORECASE).match(sql, after_cols)
        if not vm:
            errors.append("INSERT INTO affiliate_partners with no VALUES clause")
            continue
        columns = [c.strip() for c in _split_top_level(cols_str)]
        # Read one-or-more parenthesized value tuples separated by commas.
        pos = vm.end()
        while pos < len(sql) and sql[pos].isspace():
            pos += 1
        if pos >= len(sql) or sql[pos] != "(":
            errors.append("VALUES clause with no value tuple")
            continue
        while pos < len(sql) and sql[pos] == "(":
            vals_str, pos = _read_paren_group(sql, pos)
            values = [_parse_value(v) for v in _split_top_level(vals_str)]
            if len(columns) != len(values):
                errors.append(
                    f"column/value count mismatch "
                    f"({len(columns)} columns vs {len(values)} values)"
                )
            else:
                rows.append(dict(zip(columns, values)))
            while pos < len(sql) and sql[pos].isspace():
                pos += 1
            if pos < len(sql) and sql[pos] == ",":
                pos += 1
                while pos < len(sql) and sql[pos].isspace():
                    pos += 1
                continue
            break
    return rows, errors


def find_partner_mutations(sql: str) -> list[str]:
    """Return descriptors for UPDATE/DELETE statements targeting
    affiliate_partners. The audit models only INSERT seeds, so a mutation
    means a live row's value (or absence) may be the intended result of a
    migration this parser does not apply -- the caller surfaces these so the
    blind spot is visible rather than silently skewing reconciliation."""
    found: list[str] = []
    if re.search(r"\bUPDATE\s+affiliate_partners\b", sql, re.IGNORECASE):
        found.append("UPDATE")
    if re.search(r"\bDELETE\s+FROM\s+affiliate_partners\b", sql, re.IGNORECASE):
        found.append("DELETE")
    return found


def parse_seeded_partners_dir(
    migrations_dir: pathlib.Path,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Parse every migration file, keyed by lower(product_name). Later
    migrations override earlier ones for the same product_name (mirrors the
    apply order; the last definition is the current intended seed). Returns
    (seeded, errors, mutations); errors and mutations are prefixed with the
    originating filename."""
    seeded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    mutations: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        text = path.read_text()
        rows, errs = parse_seeded_partners(text)
        errors.extend(f"{path.name}: {e}" for e in errs)
        mutations.extend(
            f"{path.name}: {kind} affiliate_partners" for kind in find_partner_mutations(text)
        )
        for row in rows:
            product = row.get("product_name")
            if not product:
                continue
            row["__migration__"] = path.name
            seeded[str(product).lower()] = row
    return seeded, errors, mutations


# ---------------------------------------------------------------------------
# Reconciliation (pure -- no DB)
# ---------------------------------------------------------------------------


def _norm_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    return value


def _norm_aliases(value: Any) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(str(a) for a in value)


def _field_equal(field: str, migration_val: Any, db_val: Any) -> bool:
    if field == "product_aliases":
        return _norm_aliases(migration_val) == _norm_aliases(db_val)
    return _norm_scalar(migration_val) == _norm_scalar(db_val)


def reconcile(
    db_partners: list[dict[str, Any]],
    seeded: dict[str, dict[str, Any]],
    parse_errors: list[str] | tuple[str, ...] = (),
    mutations: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Compare live partners against migration-seeded definitions and return a
    list of status checks (same shape as the rollout-readiness audit).

    `parse_errors` (migration INSERTs the parser could not map) are surfaced
    as a hard FAIL: if seed parsing regressed, the reconciliation below is
    comparing against an incomplete seed set and its pass/warn results cannot
    be trusted, so the audit must not report success.

    `mutations` (UPDATE/DELETE on affiliate_partners) are surfaced as a WARN,
    not a FAIL: a legitimate migration may intentionally edit a partner, so its
    presence is not an error -- but the audit only models INSERT seeds, so a
    value-divergence or orphan result for an affected partner may be explained
    by a mutation this parser does not apply. The warning points the operator
    at the responsible migration rather than letting that case look like clean
    drift."""
    unversioned: list[dict[str, Any]] = []
    divergent: list[dict[str, Any]] = []

    db_keys = set()
    for partner in db_partners:
        product = str(partner.get("product_name") or "")
        key = product.lower()
        db_keys.add(key)
        seed = seeded.get(key)
        if seed is None:
            unversioned.append(
                {"product_name": product, "enabled": partner.get("enabled")}
            )
            continue
        diffs = {}
        for field in _DIVERGENCE_FIELDS:
            if not _field_equal(field, seed.get(field), partner.get(field)):
                diffs[field] = {
                    "migration": seed.get(field),
                    "live": partner.get(field),
                }
        if diffs:
            divergent.append(
                {
                    "product_name": product,
                    "migration": seed.get("__migration__"),
                    "fields": diffs,
                }
            )

    seeded_without_live = sorted(
        seed["__migration__"] + ": " + str(seed.get("product_name"))
        for key, seed in seeded.items()
        if key not in db_keys
    )

    return [
        {
            "name": "migration_seeds_parseable",
            "status": "pass" if not parse_errors else "fail",
            "required": True,
            "detail": {"parse_errors": list(parse_errors)},
        },
        {
            "name": "partner_mutations_modeled",
            "status": "pass" if not mutations else "warn",
            "required": False,
            "detail": {"unmodeled_mutations": list(mutations)},
        },
        {
            "name": "all_live_partners_versioned",
            "status": "pass" if not unversioned else "fail",
            "required": True,
            "detail": {"unversioned_partners": unversioned},
        },
        {
            "name": "no_seed_value_divergence",
            "status": "pass" if not divergent else "warn",
            "required": False,
            "detail": {"divergent_partners": divergent},
        },
        {
            "name": "no_orphan_seeds",
            "status": "pass" if not seeded_without_live else "warn",
            "required": False,
            "detail": {"seeds_without_live_row": seeded_without_live},
        },
    ]


def _exit_code(checks: list[dict[str, Any]]) -> int:
    return 1 if any(item["status"] == "fail" for item in checks) else 0


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


async def _run() -> dict[str, Any]:
    seeded, parse_errors, mutations = parse_seeded_partners_dir(MIGRATIONS_DIR)
    await init_database()
    pool = get_db_pool()
    rows = await pool.fetch(
        """
        SELECT name, product_name, product_aliases, category, affiliate_url,
               commission_type, commission_value, notes, enabled
        FROM affiliate_partners
        ORDER BY product_name
        """
    )
    db_partners = [
        {
            "name": r["name"],
            "product_name": r["product_name"],
            "product_aliases": list(r["product_aliases"]) if r["product_aliases"] else [],
            "category": r["category"],
            "affiliate_url": r["affiliate_url"],
            "commission_type": r["commission_type"],
            "commission_value": r["commission_value"],
            "notes": r["notes"],
            "enabled": r["enabled"],
        }
        for r in rows
    ]
    checks = reconcile(db_partners, seeded, parse_errors, mutations)
    return {
        "checks": checks,
        "summary": {
            "seeded_partners": len(seeded),
            "live_partners": len(db_partners),
            "pass": sum(1 for c in checks if c["status"] == "pass"),
            "warn": sum(1 for c in checks if c["status"] == "warn"),
            "fail": sum(1 for c in checks if c["status"] == "fail"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    async def _main() -> int:
        try:
            result = await _run()
        finally:
            await close_database()
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return _exit_code(result["checks"])

    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
