#!/usr/bin/env python3
"""
check_contact_write_boundary.py - fail CI when a new SQL write to the
``contacts`` table appears outside the approved provider module.

Why a static gate is a real guarantee here, not a wish:

1. Atlas has no ORM. Every contact write is literal SQL in a string literal,
   so a static reader can see all of them. There is no lazy-loaded mapper
   emitting an INSERT somewhere this script cannot look.
2. The approved surface is tiny. Production INSERTs into ``contacts`` live at
   exactly two call sites, both inside ``atlas_brain/services/crm_provider.py``.
   An allow-list of one module is exact rather than aspirational.
3. The blast radius of a miss is a split-brain CRM: a writer that bypasses the
   provider also bypasses tenant stamping, provenance, normalization, and the
   lifecycle audit ledger.

What this does NOT prove: nothing here stops a process holding Atlas database
credentials from connecting and inserting directly. This gate governs code in
this repository. Credential-level restriction is tracked separately.

SQL is read out of Python string literals via ``ast``, so comments and
identifiers never produce a finding, and a commented-out INSERT is correctly
ignored.

stdlib only, Python 3.9+.

Usage:
    python scripts/check_contact_write_boundary.py
    python scripts/check_contact_write_boundary.py --baseline <path>
    python scripts/check_contact_write_boundary.py --baseline <path> --update-baseline
    python scripts/check_contact_write_boundary.py --json
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Modules permitted to contain SQL writes against `contacts`.
# Keep this list as short as the codebase allows; every entry is a place a
# future bypass can hide legitimately.
PROVIDER_MODULE = "atlas_brain/services/crm_provider.py"
INSERT_ALLOWED = (PROVIDER_MODULE,)

# UPDATE/DELETE are recorded but not blocking in this slice: legitimate
# operator scripts still carry their own guarded updates, and converging them
# is a separate, sequenced piece of work. They are baselined so that *new*
# ones are visible in review even while they do not fail the build.
MUTATION_ALLOWED = (
    "atlas_brain/services/crm_provider.py",
    "scripts/backfill_business_context.py",
    "scripts/import_eom_customers_live.py",
    "scripts/sync_eom_portal_customers.py",
)

# Directories that never contain production write paths.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# Tests legitimately contain SQL fixtures and assertions about SQL.
#
# Exemption is by AUTHORITATIVE ROOT, and this rule has been wrong twice.
# First it keyed on basename, which exempted 13 real production modules merely
# named `test_*.py` (`scripts/test_adapter_live.py`,
# `atlas_brain/test_token_tracking.py`, several `scripts/debug/test_*.py`).
# Replacing that with "any ancestor directory named tests" then exempted any
# nested `tests/` dir, so `atlas_brain/services/tests/evil.py` passed.
#
# The repo has exactly one test root -- `tests/` -- so the rule names it.
# Adding a second root is a deliberate, reviewable edit here, not something a
# new directory grants itself.
TEST_ROOTS = ("tests/",)


def _is_test_path(rel: str) -> bool:
    return rel.startswith(TEST_ROOTS)


# The table name may be schema-qualified and may be a quoted identifier:
# `INSERT INTO "contacts" (...)` is valid SQL and must not evade the gate.
TABLE = r"(?:(?:public|\"public\")\s*\.\s*)?(?:contacts\b|\"contacts\")"

# A fragment that opens a write statement but whose table name is not a literal
# in the same expression — `"INSERT INTO " + table` or `f"INSERT INTO {tbl}"`.
# The gate cannot prove such a statement does not target `contacts`, and a guard
# must not treat "cannot prove" as "safe".
#
# Deliberately case-SENSITIVE and restricted to the two-word forms. A
# case-insensitive `\bupdate\s*$` matches ordinary English -- it flagged
# "DRY RUN: Would update", "would update", and ", update" across nine files on
# the first run. SQL keywords are uppercase throughout this codebase, and prose
# is not, so requiring uppercase separates them without inventing a parser.
# Bare `UPDATE` is excluded entirely: too short to be distinctive even uppercase.

# A runtime hole left by constant-folding. Chosen because it cannot occur in
# source text, so it can never be confused with real SQL whitespace -- a plain
# space placeholder made the hole vanish and the rule stopped firing at all.
HOLE = "\x00"

DYNAMIC_TARGET = re.compile(
    r"(?:INSERT\s+INTO|DELETE\s+FROM|MERGE\s+INTO|UPDATE|COPY)\s*" + HOLE
    + r"|(?:INSERT\s+INTO|DELETE\s+FROM|MERGE\s+INTO)\s*$"
)

# Runtime-built table names only matter where a `contacts` write is plausible.
# The repo has 18 pre-existing `INSERT INTO {table}` sites in unrelated
# subsystems (podcast/campaign/reddit importers, a generic migration runner)
# that parameterize their own tables by design. Blocking those would fail the
# build on code this boundary has no claim over, which is how a gate earns a
# reputation for noise and gets switched off. Outside this scope a dynamic
# target is reported, not blocking.
DYNAMIC_SCOPE = (
    "atlas_brain/services/",
    "atlas_brain/api/",
    "atlas_brain/mcp/",
    "atlas_brain/comms/",
    "atlas_brain/eom_api/",
    "atlas_brain/autonomous/",
    "atlas_brain/tools/",
    "scripts/",
)


def _in_dynamic_scope(rel: str) -> bool:
    return rel.startswith(DYNAMIC_SCOPE)

# Each pattern requires the *syntactic shape* of the statement, not just the
# keyword next to the table name. Prose matches otherwise: a docstring reading
# "Create/update contacts in the Atlas CRM" is not a write, and neither is this
# script's own diagnostic text. Requiring the trailing clause keeps findings to
# things that could actually execute.
PATTERNS = {
    "INSERT": re.compile(
        r"\binsert\s+into\s+" + TABLE
        # Optional alias: `INSERT INTO contacts AS c (...)` is valid and was a
        # live bypass. `AS` is required for the alias form so that a bare
        # following keyword cannot be mistaken for one.
        + r"(?:\s+as\s+[a-z_][a-z0-9_]*)?"
        + r"\s*(?:\(|select\b|values\b|default\b|overriding\b)",
        re.IGNORECASE | re.DOTALL,
    ),
    # INSERT is not the only way a row reaches the table. MERGE can insert,
    # COPY bulk-loads, and SELECT ... INTO creates and fills. All three write
    # rows while skipping the provider, so all three are treated as creates.
    "MERGE": re.compile(r"\bmerge\s+into\s+" + TABLE, re.IGNORECASE | re.DOTALL),
    "COPY": re.compile(
        r"\bcopy\s+" + TABLE + r"(?:\s*\([^)]*\))?\s+from\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "SELECT_INTO": re.compile(
        # The optional qualifiers are PostgreSQL SELECT INTO syntax. The inner
        # guard stops this matching the INTO of a following INSERT: without it,
        # "SELECT 1; INSERT INTO contacts (...)" reported a phantom SELECT_INTO
        # alongside the real INSERT.
        r"\bselect\b(?:(?!\binsert\b|;).)*?\binto\s+"
        r"(?:temp\s+|temporary\s+|unlogged\s+)?" + TABLE,
        re.IGNORECASE | re.DOTALL,
    ),
    # TRUNCATE deletes every row without touching the provider or the lifecycle
    # ledger. Classified with the mutations rather than the creates, so it
    # surfaces in the inventory under this slice's deferred UPDATE/DELETE policy
    # instead of reading as a clean tree.
    "TRUNCATE": re.compile(
        r"\btruncate\s+(?:table\s+)?(?:only\s+)?" + TABLE,
        re.IGNORECASE | re.DOTALL,
    ),
    "UPDATE": re.compile(
        r"\bupdate\s+(?:only\s+)?" + TABLE + r"(?:\s+(?:as\s+)?[a-z_][a-z0-9_]*)?\s+set\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "DELETE": re.compile(
        r"\bdelete\s+from\s+(?:only\s+)?" + TABLE,
        re.IGNORECASE | re.DOTALL,
    ),
}


_MERGE_INSERT_BRANCH = re.compile(
    r"when\s+not\s+matched(?:(?!;).)*?\binsert\b", re.IGNORECASE | re.DOTALL
)


def _refine_operation(operation: str, haystack: str, start: int) -> str:
    """Narrow an operation using the statement body following the match.

    A `MERGE INTO contacts ... WHEN MATCHED THEN UPDATE` with no insert branch
    creates no rows, so classifying it as a create would red the build for a
    mutation this slice deliberately leaves non-blocking.
    """
    if operation != "MERGE":
        return operation
    end = haystack.find(";", start)
    body = haystack[start: end if end != -1 else len(haystack)]
    return "MERGE" if _MERGE_INSERT_BRANCH.search(body) else "MERGE_UPDATE"


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    operation: str
    snippet: str

    def key(self) -> str:
        """Baseline identity. Deliberately excludes the line number so that
        unrelated edits above a known write do not churn the baseline."""
        return f"{self.path}::{self.operation}::{self.snippet}"


SELF_PATH = "scripts/check_contact_write_boundary.py"


def _iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        # The detector describes the very patterns it looks for, so scanning
        # itself would always report a violation it caused.
        if rel.as_posix() == SELF_PATH:
            continue
        yield path


def _folded_value(node: ast.AST):
    """Constant-fold a string expression where the value is statically known.

    Scanning each `ast.Constant` in isolation misses SQL assembled across
    literals: `"INSERT INTO " + "contacts (...)"` and
    `f"INSERT INTO {schema}.contacts (...)"` are both single statements at
    runtime but several constants in the AST. Interpolations and unfoldable
    operands fold to HOLE, so a match cannot be manufactured across a runtime
    value while adjacent literal text still joins up.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append(HOLE)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _folded_value(node.left)
        right = _folded_value(node.right)
        if left is None and right is None:
            return None
        # An unfoldable side becomes a hole: it may be a table name at runtime,
        # which DYNAMIC_TARGET is responsible for catching. A plain space would
        # be indistinguishable from real whitespace and the hole would vanish.
        return (left if left is not None else HOLE) + (
            right if right is not None else HOLE
        )
    return None


# asyncpg (pinned in requirements.txt) can write rows without a SQL statement.
# `conn.copy_records_to_table("contacts", records=rows)` inserts directly and
# leaves nothing for a text scanner to find.
DRIVER_WRITE_METHODS = frozenset({
    "copy_records_to_table",
    "copy_to_table",
})


def _driver_write_findings(tree: ast.AST, rel: str) -> list:
    """Find driver-level table writes whose target is a literal `contacts`."""
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in DRIVER_WRITE_METHODS:
            continue
        targets = list(node.args) + [kw.value for kw in node.keywords if kw.arg in (None, "table_name")]
        for target in targets:
            if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
                continue
            if re.fullmatch(r'(?:public\.)?"?contacts"?', target.value.strip(), re.IGNORECASE):
                findings.append(
                    Finding(
                        path=rel,
                        line=getattr(node, "lineno", 0),
                        operation="DRIVER_WRITE",
                        snippet=_normalize(f"{name}({target.value!r}, ...)"),
                    )
                )
                break
    return findings


def _docstring_nodes(tree: ast.AST) -> set:
    """Ids of Constant nodes that are docstrings.

    A docstring is prose about code, never a statement the database executes.
    Scanning them produced the `import_calendar_contacts.py` false positive and
    would keep producing more as the pattern set widens.
    """
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = next(iter(getattr(node, "body", None) or []), None)
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _string_literals(tree: ast.AST):
    """Yield (value, lineno) for each *outermost* string expression.

    Constants inside a folded expression are consumed by their parent rather
    than yielded separately. Without that, `"INSERT INTO " + "contacts (...)"`
    reports twice — once as the resolved INSERT and once as a DYNAMIC finding
    for the dangling left fragment, even though the fold already proved the
    target. Reporting a resolved statement as unresolvable is the kind of
    self-contradiction that makes reviewers stop reading gate output.
    """
    consumed = set(_docstring_nodes(tree))
    folded_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.JoinedStr, ast.BinOp)):
            folded = _folded_value(node)
            if folded is None or not folded.replace(HOLE, "").strip():
                continue
            children = [
                child for child in ast.walk(node)
                if isinstance(child, ast.Constant) and child is not node
            ]
            # A nested BinOp is walked separately; only the outermost fold wins.
            if id(node) in consumed:
                continue
            for child in children:
                consumed.add(id(child))
            for child in ast.walk(node):
                if isinstance(child, (ast.JoinedStr, ast.BinOp)) and child is not node:
                    consumed.add(id(child))
            folded_nodes.append((folded, node.lineno))

    for folded, lineno in folded_nodes:
        yield folded, lineno

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in consumed
        ):
            yield node.value, node.lineno


def _normalize(sql_fragment: str) -> str:
    readable = sql_fragment.replace(HOLE, "{runtime value}")
    return " ".join(readable.split())[:120]


def scan_file(path: Path, root: Path) -> tuple:
    """Return (findings, unanalyzable_reason).

    A file this gate cannot read or parse must never be silently treated as
    clean: that would let a bypass hide behind a syntax error or a decoding
    failure, and the gate would still report OK. The caller surfaces the reason
    instead. A file that does not parse cannot execute, so it is not a live
    write path — but it is also not evidence of absence, which is the claim a
    silent skip would be making.
    """
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], f"{rel}: unreadable ({type(exc).__name__})"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], f"{rel}: unparsable (line {exc.lineno})"

    findings = []
    seen = set()
    for raw_value, lineno in _string_literals(tree):
        # Python-embedded SQL needs the same lexing as a .sql file: PostgreSQL
        # treats `INSERT /* note */ INTO contacts` as a write, and treats
        # `'... INSERT INTO contacts ...'` as data. Matching the raw literal
        # missed the first and falsely blocked the second.
        value = _blank_sql_noise(raw_value)
        for operation, pattern in PATTERNS.items():
            # finditer, not search: one literal can hold several statements, and
            # reporting only the first under-counts the inventory. A second
            # write added inside an existing literal in an allow-listed module
            # would then never surface as drift -- the same hole the multiset
            # comparison closed for identical keys. scan_sql_file already did
            # this; the Python path did not.
            for match in pattern.finditer(value):
                refined = _refine_operation(operation, value, match.start())
                start = max(0, match.start() - 10)
                finding = Finding(
                    path=rel,
                    line=lineno,
                    operation=refined,
                    snippet=_normalize(value[start:match.end() + 60]),
                )
            # Dedup on (operation, line, snippet). key() omits the line, which
            # collapsed two distinct statements sharing a normalized snippet and
            # dropped one of the provider's nine UPDATEs. (operation, line)
            # alone then collapsed two distinct writes on ONE line. The triple
            # keeps genuinely different writes while still folding the duplicate
            # a folded expression and its child constant would otherwise emit.
                dedup = (finding.operation, finding.line, finding.snippet)
                if dedup not in seen:
                    seen.add(dedup)
                    findings.append(finding)

        # A write whose target is built at runtime cannot be cleared by reading
        # the literal. Report it so a reviewer resolves it, rather than letting
        # `"INSERT INTO " + table` pass as clean.
        if DYNAMIC_TARGET.search(value.rstrip()):
            finding = Finding(
                path=rel,
                line=lineno,
                operation="DYNAMIC",
                snippet=_normalize(value[-90:]),
            )
            dedup = (finding.operation, finding.line, finding.snippet)
            if dedup not in seen:
                seen.add(dedup)
                findings.append(finding)
    # Driver-level writes carry no SQL text for the patterns above to see.
    findings.extend(_driver_write_findings(tree, rel))
    return findings, None


def scan(root: Path) -> list:
    """Findings only. Callers needing analyzability use scan_tree()."""
    findings, _unanalyzable = scan_tree(root)
    return findings


# EXECUTE immediately preceding a literal, allowing INTO/USING-free simple form.
_EXECUTE_BEFORE = re.compile(r"\bexecute\s*$", re.IGNORECASE)


def _blank_sql_noise(sql: str) -> str:
    """Blank comments AND string-literal bodies, preserving byte offsets.

    Two failure directions, both reproduced before this was written:

    * A comment stripper that does not understand literals erases real code.
      ``SELECT E'it\\'s -- still literal'; INSERT INTO contacts (...)`` ended
      with the INSERT treated as comment text, and the gate exited 0.
    * A scanner that reads literal *contents* invents statements that never
      execute. ``SELECT 'Example only: INSERT INTO contacts (...)'`` was
      reported as a blocking write.

    Both are fixed by the same pass: literal bodies become spaces, so SQL
    quoted as data is inert and SQL outside quotes is still visible. Everything
    is replaced in place (newlines preserved) so reported line numbers stay
    correct.

    Handles PostgreSQL line and nested block comments, single-quoted strings
    with '' escapes, ``E''`` escape strings where a backslash escapes the
    delimiter, and tagged dollar quotes. Double-quoted identifiers are kept
    intact, because ``INSERT INTO "contacts"`` is a real target rather than data.
    """
    out: list[str] = []
    i = 0
    n = len(sql)

    def blank(text: str) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in text)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(blank(sql[i:j]))
            i = j
            continue

        if ch == "/" and nxt == "*":
            depth, j = 1, i + 2
            while j < n and depth:
                if sql.startswith("/*", j):
                    depth += 1
                    j += 2
                elif sql.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(blank(sql[i:j]))
            i = j
            continue

        # E'...' escape string: a backslash escapes the closing quote.
        if ch in "eE" and nxt == "'" and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")):
            j = i + 2
            while j < n:
                if sql[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if sql[j] == "'":
                    if sql[j + 1:j + 2] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:i + 1] + "'" + blank(sql[i + 2:max(i + 2, j - 1)]) + "'")
            i = j
            continue

        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if sql[j + 1:j + 2] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            body = sql[i + 1:max(i + 1, j - 1)]
            # `EXECUTE 'INSERT INTO contacts ...'` runs the literal. Treating
            # every literal as inert data hid exactly that, so a literal in
            # EXECUTE position is lexed as code instead. Doubled quotes inside
            # it are unescaped first, since that is how the SQL was written.
            if _EXECUTE_BEFORE.search("".join(out)):
                out.append("'" + _blank_sql_noise(body.replace("''", " '")) + "'")
            else:
                out.append("'" + blank(body) + "'")
            i = j
            continue

        if ch == "$":
            match = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if match:
                tag = match.group(0)
                close = sql.find(tag, i + len(tag))
                j = (close + len(tag)) if close != -1 else n
                body = sql[i + len(tag): max(i + len(tag), j - len(tag))]
                # A dollar-quoted body is NOT reliably inert. PostgreSQL runs
                # `DO $$ BEGIN INSERT INTO contacts ...; END $$;` and function
                # bodies the same way, and the migration runner submits the
                # whole file. Blanking these hid a real writer and exited 0, so
                # the body is lexed recursively and left visible: fail closed.
                # The cost is that SQL quoted as documentation inside a dollar
                # body reads as executable, which is the safe direction for a
                # guard.
                out.append(blank(tag) + _blank_sql_noise(body) + blank(tag))
                i = j
                continue

        if ch == '"':
            j = i + 1
            while j < n and sql[j] != '"':
                j += 1
            j = min(j + 1, n)
            out.append(sql[i:j])
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def scan_sql_file(path: Path, root: Path) -> tuple:
    """Scan a .sql file. Migrations are executable SQL, not commentary.

    Python is not the only way a statement reaches the database: a migration or
    data-fix script writes rows directly. Comments are stripped with a
    string-aware tokenizer first, so a documented rollback recipe is ignored
    while a `--` inside a literal cannot swallow a real statement.
    """
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], f"{rel}: unreadable ({type(exc).__name__})"

    stripped = _blank_sql_noise(source)

    findings, seen = [], set()
    for operation, pattern in PATTERNS.items():
        for match in pattern.finditer(stripped):
            refined = _refine_operation(operation, stripped, match.start())
            line = stripped.count("\n", 0, match.start()) + 1
            start = max(0, match.start() - 10)
            finding = Finding(
                path=rel,
                line=line,
                operation=refined,
                snippet=_normalize(stripped[start:match.end() + 60]),
            )
            dedup = (finding.operation, finding.line, finding.snippet)
            if dedup not in seen:
                seen.add(dedup)
                findings.append(finding)
    return findings, None


def _iter_sql_files(root: Path):
    for path in sorted(root.rglob("*.sql")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield path


def scan_tree(root: Path) -> tuple:
    """Return (findings, unanalyzable) for the whole tree."""
    findings, unanalyzable = [], []
    for path in _iter_python_files(root):
        file_findings, reason = scan_file(path, root)
        findings.extend(file_findings)
        if reason:
            unanalyzable.append(reason)
    for path in _iter_sql_files(root):
        file_findings, reason = scan_sql_file(path, root)
        findings.extend(file_findings)
        if reason:
            unanalyzable.append(reason)
    return sorted(findings), sorted(unanalyzable)


# Every statement form that can put a row into `contacts`. These share INSERT's
# stricter allow-list; UPDATE/DELETE remain the softer, still-converging set.
CREATE_OPERATIONS = (
    "INSERT", "MERGE", "COPY", "SELECT_INTO", "DYNAMIC", "DRIVER_WRITE",
)


def is_allowed(finding: Finding) -> bool:
    if finding.operation in CREATE_OPERATIONS:
        return finding.path in INSERT_ALLOWED
    return finding.path in MUTATION_ALLOWED


def classify(findings: list, baseline: dict) -> tuple:
    """Split findings into blocking violations and non-blocking new mutations."""
    known = set(baseline.get("known_writes", []))
    blocking, new_mutations = [], []
    for finding in findings:
        if _is_test_path(finding.path):
            continue
        if is_allowed(finding):
            continue
        if finding.operation in ("INSERT", "MERGE", "COPY", "SELECT_INTO", "DRIVER_WRITE"):
            blocking.append(finding)
        elif finding.operation == "DYNAMIC":
            if _in_dynamic_scope(finding.path) and finding.key() not in known:
                blocking.append(finding)
            elif finding.key() not in known:
                new_mutations.append(finding)
        elif finding.key() not in known:
            new_mutations.append(finding)
    return blocking, new_mutations


def build_baseline(findings: list, unanalyzable: list | None = None) -> dict:
    """Record the full production writer inventory, not just the exceptions.

    ``known_writes`` alone would be an empty list today, because every current
    writer sits in an allow-list. A baseline that is empty makes its own drift
    test vacuous: empty-in, empty-out, passing forever while the tree changes
    underneath it. ``writer_inventory`` records every production write site so
    that adding, moving, or deleting one shows up as a reviewable diff.
    """
    production = [f for f in findings if not _is_test_path(f.path)]
    return {
        "_comment": (
            "Every SQL write to `contacts` in production code. INSERTs outside "
            "atlas_brain/services/crm_provider.py always fail the build and are "
            "never silenced by this file. `writer_inventory` is the drift record: "
            "a diff here means a contact write site was added, moved, or removed. "
            "`known_writes` holds non-blocking UPDATE/DELETE sites tolerated "
            "outside the allow-listed modules while legacy writers are converged."
        ),
        "insert_allowed": list(INSERT_ALLOWED),
        "mutation_allowed": list(MUTATION_ALLOWED),
        "writer_inventory": sorted(f.key() for f in production),
        "known_writes": sorted(
            f.key() for f in production if not is_allowed(f)
        ),
        "unanalyzable": sorted(unanalyzable or []),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail when a contacts-table write appears outside the approved provider."
    )
    ap.add_argument("--root", default=str(ROOT), help="Repository root to scan.")
    ap.add_argument("--baseline", default=None, help="Path to the baseline JSON file.")
    ap.add_argument(
        "--inventory-baseline",
        default=None,
        help=(
            "Baseline whose writer_inventory is compared against the scanned "
            "tree. Defaults to <root>/tests/contact_write_boundary/baseline.json. "
            "Kept separate from --baseline so trusted-base runs can take POLICY "
            "from the base revision while still requiring the scanned tree to "
            "keep its own inventory honest."
        ),
    )
    ap.add_argument("--update-baseline", action="store_true",
                    help="Rewrite the baseline from the current tree.")
    ap.add_argument("--json", action="store_true", help="Emit findings as JSON.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    findings, unanalyzable = scan_tree(root)

    if args.update_baseline:
        if not args.baseline:
            raise SystemExit("--update-baseline requires --baseline")
        Path(args.baseline).write_text(
            json.dumps(build_baseline(findings, unanalyzable), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {args.baseline}")
        return 0

    baseline = {}
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    # Inventory drift. classify() only reads known_writes, so without this a PR
    # could add a third INSERT inside the allow-listed provider module, or
    # delete an inventoried writer, and still exit 0 -- the drift record the
    # plan promises would never actually be checked.
    inventory_path = Path(
        args.inventory_baseline
        or (root / "tests" / "contact_write_boundary" / "baseline.json")
    )
    inventory_drift: dict = {"added": [], "removed": [], "missing_baseline": False}
    if inventory_path.exists():
        committed = json.loads(inventory_path.read_text(encoding="utf-8"))
        # Multisets, not sets. Two byte-for-byte identical writes on different
        # lines share a key, so set arithmetic reports no drift when one is
        # added or removed -- the third and last variant of "a collapse lost
        # the information the comparison needed".
        recorded = Counter(committed.get("writer_inventory", []))
        current = Counter(build_baseline(findings, unanalyzable)["writer_inventory"])
        inventory_drift["added"] = sorted((current - recorded).elements())
        inventory_drift["removed"] = sorted((recorded - current).elements())
    elif not args.update_baseline:
        # Fail closed. Skipping the comparison when the file is absent made
        # deleting the baseline a one-line way to switch inventory enforcement
        # off, in the same diff as the writer it would have surfaced.
        inventory_drift["missing_baseline"] = True

    blocking, new_mutations = classify(findings, baseline)
    known_unanalyzable = set(baseline.get("unanalyzable", []))
    new_unanalyzable = [u for u in unanalyzable if u not in known_unanalyzable]

    if args.json:
        print(json.dumps(
            {
                "blocking": [asdict(f) for f in blocking],
                "new_mutations": [asdict(f) for f in new_mutations],
                "unanalyzable": new_unanalyzable,
                "inventory_drift": inventory_drift,
            },
            indent=2,
        ))
        return 1 if (blocking or new_unanalyzable or any(inventory_drift.values())) else 0

    print("contact write-boundary check")
    print("-" * 60)

    if blocking:
        print("BLOCKING: INSERT INTO contacts outside the approved provider module.\n")
        for f in blocking:
            print(f"  {f.path}:{f.line}")
            print(f"    {f.snippet}")
        print(
            "\nEvery contact insert must go through "
            f"{PROVIDER_MODULE}, so that tenant stamping, provenance,\n"
            "normalization, and the lifecycle audit ledger cannot be skipped.\n"
            "Route the write through the provider (or the EOM ingress/funnel service\n"
            "layered on top of it) rather than adding a second insert site."
        )

    if new_mutations:
        print("\nNEW (non-blocking) contact mutation outside the approved modules:\n")
        for f in new_mutations:
            print(f"  {f.path}:{f.line}  [{f.operation}]")
            print(f"    {f.snippet}")
        print(
            "\nThese do not fail the build yet. If intentional, refresh the baseline:\n"
            f"  python scripts/check_contact_write_boundary.py --baseline <path> --update-baseline"
        )

    if new_unanalyzable:
        print("\nBLOCKING: file(s) this gate could not analyze.\n")
        for reason in new_unanalyzable:
            print(f"  {reason}")
        print(
            "\nAn unreadable or unparsable file is not evidence that it contains no\n"
            "contact write. Fix the file, or record it in the baseline's\n"
            "`unanalyzable` list with a reason if it is a deliberate fixture."
        )

    if inventory_drift.get("missing_baseline"):
        print(
            f"\nBLOCKING: no writer inventory at {inventory_path}.\n\n"
            "  The inventory is the record of every contact write site. A tree\n"
            "  without one cannot be checked for drift, so its absence is treated\n"
            "  as a failure rather than as nothing to do.\n"
        )
    elif any(inventory_drift.values()):
        print("\nBLOCKING: the committed writer inventory does not match the tree.\n")
        for entry in inventory_drift["added"]:
            print(f"  + {entry}")
        for entry in inventory_drift["removed"]:
            print(f"  - {entry}")
        print(
            "\nEvery contact write site is recorded so that adding, moving, or\n"
            "removing one is a reviewable diff rather than a silent change.\n"
            "Refresh it and review the delta as part of the change:\n"
            f"  python {SELF_PATH} --baseline {inventory_path} --update-baseline"
        )

    if not blocking and not new_mutations and not new_unanalyzable and not any(inventory_drift.values()):
        total = len([f for f in findings if not _is_test_path(f.path)])
        print(f"OK - {total} contact write(s), all inside approved modules or baselined.")

    return 1 if (blocking or new_unanalyzable or any(inventory_drift.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
