#!/usr/bin/env python3
"""Produce one fail-closed Atlas PR watcher snapshot.

The systemd timer invokes an installed copy of this file. The producer reads
GitHub and local worktree state, writes a versioned readiness proof, and never
mutates or merges a pull request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence
from urllib.parse import quote


HOME = Path.home()
DEFAULT_CONFIG_DIR = HOME / ".config" / "atlas-pr-watchers"
DEFAULT_STATE_DIR = HOME / ".local" / "state" / "atlas-pr-watchers"
MARKER_START = "<!-- atlas-pr-watch:start -->"
MARKER_END = "<!-- atlas-pr-watch:end -->"
SAFE_WATCHER_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
VALID_CHECK_EXIT_CODES = {0, 1, 8}
MAX_THREAD_PAGES = 50
MAX_REVIEW_PAGES = 50
COMMAND_TIMEOUT_SECONDS = 60
CODEX_CONNECTOR_LOGINS = frozenset(
    {
        "chatgpt-codex-connector",
        "chatgpt-codex-connector[bot]",
    }
)
RECONCILIATION_LIB_DIR = "atlas-pr-watch-lib"
RECONCILIATION_CHECKER_NAME = "check_ai_reconciliation_live.py"
TRUSTED_RECONCILIATION_CHECKER = (
    Path(__file__).resolve().parent / RECONCILIATION_LIB_DIR / RECONCILIATION_CHECKER_NAME
)

THREADS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviewThreads(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          id
          isResolved
          isOutdated
          path
          line
          comments(first:1){ nodes{ author{ login } } }
        }
      }
    }
  }
}
"""

REVIEWS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviews(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{ author{ login } commit{ oid } state }
      }
    }
  }
}
"""


def _valid_watcher_id(value: str) -> bool:
    return bool(SAFE_WATCHER_ID_RE.fullmatch(value)) and ".." not in value and not value.startswith(".")


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line in {path}: {raw_line!r}")
        key, raw_value = line.split("=", 1)
        parts = shlex.split(raw_value.strip()) if raw_value.strip() else []
        if len(parts) > 1:
            raise ValueError(f"config value must be quoted in {path}: {key.strip()}")
        values[key.strip()] = next(iter(parts), "")
    return values


def _run(command: Sequence[str], *, cwd: Path) -> tuple[int, str, str]:
    program = next(iter(command), "")
    if not program:
        return 127, "", "cannot run an empty command"
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out after {COMMAND_TIMEOUT_SECONDS}s: {program}"
    except OSError as exc:
        return 127, "", f"could not run {program}: {exc}"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _run_json(
    command: Sequence[str],
    *,
    cwd: Path,
    allowed_codes: set[int],
    expected_type: type,
) -> tuple[Any | None, str | None]:
    program = next(iter(command), "command")
    code, stdout, stderr = _run(command, cwd=cwd)
    if code not in allowed_codes:
        detail = stderr or stdout or f"exit {code}"
        return None, f"{program} command failed ({code}): {detail}"
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"{program} returned invalid JSON: {exc}"
    if not isinstance(value, expected_type):
        return None, f"{program} JSON must be {expected_type.__name__}"
    return value, None


def _pr_view_command(pr: str, repo: str) -> list[str]:
    return [
        "gh",
        "pr",
        "view",
        pr,
        "--repo",
        repo,
        "--json",
        "number,title,url,baseRefName,headRefName,headRefOid,mergeStateStatus,reviewDecision,isDraft,state",
    ]


def _validate_pr_metadata(value: dict[str, Any], *, label: str) -> str | None:
    number = value.get("number")
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        return f"{label} PR metadata has no positive number"
    for key in (
        "title",
        "url",
        "baseRefName",
        "headRefName",
        "headRefOid",
        "mergeStateStatus",
        "state",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            return f"{label} PR metadata has no {key}"
    if not isinstance(value.get("isDraft"), bool):
        return f"{label} PR metadata has no boolean isDraft"
    review_decision = value.get("reviewDecision")
    if "reviewDecision" not in value or (
        review_decision is not None and not isinstance(review_decision, str)
    ):
        return f"{label} PR metadata has malformed reviewDecision"
    return None


def _checks_command(pr: str, repo: str, *, required: bool) -> list[str]:
    command = [
        "gh",
        "pr",
        "checks",
        pr,
        "--repo",
        repo,
        "--json",
        "name,state,bucket,link,workflow,startedAt,completedAt,description,event",
    ]
    if required:
        command.append("--required")
    return command


def _required_policy_command(repo: str, base_ref: str) -> list[str]:
    encoded_ref = quote(base_ref, safe="")
    return [
        "gh",
        "api",
        f"repos/{repo}/branches/{encoded_ref}/protection/required_status_checks",
    ]


def _required_contexts(payload: dict[str, Any]) -> tuple[set[str], str | None]:
    raw_contexts = payload.get("contexts")
    raw_checks = payload.get("checks")
    if not isinstance(raw_contexts, list) or not isinstance(raw_checks, list):
        return set(), "required-status policy contexts/checks are malformed"
    contexts: set[str] = set()
    for index, item in enumerate(raw_contexts):
        if not isinstance(item, str) or not item:
            return set(), f"required-status context {index} is malformed"
        contexts.add(item)
    for index, item in enumerate(raw_checks):
        if not isinstance(item, dict):
            return set(), f"required-status check {index} is not an object"
        context = item.get("context")
        if not isinstance(context, str) or not context:
            return set(), f"required-status check {index} has no context"
        contexts.add(context)
    if not contexts:
        return set(), "required-status policy has no contexts/checks"
    return contexts, None


def _check_summary(checks: list[Any], *, required: bool) -> tuple[list[str], list[str], str | None]:
    failures: list[str] = []
    pending: list[str] = []
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            return [], [], f"check result {index} is not an object"
        name = item.get("name")
        bucket = item.get("bucket")
        if not isinstance(name, str) or not name:
            return [], [], f"check result {index} has no name"
        if not isinstance(bucket, str) or not bucket:
            return [], [], f"check {name!r} has no bucket"
        normalized = bucket.lower()
        if normalized == "pass":
            continue
        if normalized == "pending":
            pending.append(name)
            continue
        if not required and normalized == "skipping":
            continue
        failures.append(f"{name} ({normalized})")
    return failures, pending, None


def _repo_parts(repo: str) -> tuple[str, str]:
    owner, separator, name = repo.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise ValueError("REPO must be an owner/name slug")
    return owner, name


def _is_codex_login(value: Any) -> bool:
    return isinstance(value, str) and value.lower() in CODEX_CONNECTOR_LOGINS


def _first_thread_author_login(node: dict[str, Any], *, index: int) -> tuple[str, str | None]:
    comments = node.get("comments")
    if not isinstance(comments, dict):
        return "", f"review thread {index} comments envelope is missing"
    comment_nodes = comments.get("nodes")
    if not isinstance(comment_nodes, list):
        return "", f"review thread {index} comments nodes are malformed"
    if not comment_nodes:
        return "", f"review thread {index} has no first comment"
    first = comment_nodes[0]
    if not isinstance(first, dict):
        return "", f"review thread {index} first comment is not an object"
    author = first.get("author")
    if author is None:
        return "", ""
    if not isinstance(author, dict):
        return "", f"review thread {index} author is malformed"
    login = author.get("login")
    if login is None:
        return "", ""
    if not isinstance(login, str):
        return "", f"review thread {index} author login is malformed"
    return login, None


def _fetch_threads(pr: int, repo: str, *, cwd: Path) -> tuple[list[dict[str, Any]], int, bool, str | None]:
    owner, name = _repo_parts(repo)
    unresolved: list[dict[str, Any]] = []
    cursor: str | None = None
    pages = 0
    for _ in range(MAX_THREAD_PAGES):
        command = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={THREADS_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"pr={pr}",
        ]
        if cursor is not None:
            command.extend(["-F", f"cursor={cursor}"])
        payload, error = _run_json(
            command,
            cwd=cwd,
            allowed_codes={0},
            expected_type=dict,
        )
        if error:
            return unresolved, pages, False, error
        if not isinstance(payload, dict):
            return unresolved, pages, False, "GraphQL response must be an object"
        graphql_errors = payload.get("errors")
        if graphql_errors:
            return unresolved, pages, False, "GraphQL response contains errors"
        data = payload.get("data")
        repository = data.get("repository") if isinstance(data, dict) else None
        pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
        threads = pull_request.get("reviewThreads") if isinstance(pull_request, dict) else None
        if threads is None:
            return unresolved, pages, False, "GraphQL reviewThreads envelope is missing"
        if not isinstance(threads, dict):
            return unresolved, pages, False, "GraphQL reviewThreads must be an object"
        nodes = threads.get("nodes")
        page_info = threads.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            return unresolved, pages, False, "GraphQL thread nodes/pageInfo are malformed"
        pages += 1
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                return unresolved, pages, False, f"review thread {index} is not an object"
            is_resolved = node.get("isResolved")
            is_outdated = node.get("isOutdated")
            if not isinstance(is_resolved, bool) or not isinstance(is_outdated, bool):
                return unresolved, pages, False, f"review thread {index} has malformed resolution flags"
            thread_id = node.get("id")
            path = node.get("path")
            line = node.get("line")
            if not isinstance(thread_id, str) or not thread_id:
                return unresolved, pages, False, f"review thread {index} has no id"
            if path is not None and not isinstance(path, str):
                return unresolved, pages, False, f"review thread {index} has malformed path"
            if line is not None and (not isinstance(line, int) or isinstance(line, bool)):
                return unresolved, pages, False, f"review thread {index} has malformed line"
            if is_resolved:
                continue
            author_login, author_error = _first_thread_author_login(node, index=index)
            if author_error:
                return unresolved, pages, False, author_error
            if not _is_codex_login(author_login):
                continue
            unresolved.append(
                {
                    "id": thread_id,
                    "is_outdated": is_outdated,
                    "path": path,
                    "line": line,
                }
            )
        has_next = page_info.get("hasNextPage")
        if not isinstance(has_next, bool):
            return unresolved, pages, False, "GraphQL hasNextPage must be boolean"
        if not has_next:
            return unresolved, pages, True, None
        next_cursor = page_info.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return unresolved, pages, False, "GraphQL pagination cursor is missing"
        cursor = next_cursor
    return unresolved, pages, False, f"review-thread pagination exceeded {MAX_THREAD_PAGES} pages"


def _fetch_codex_head_reviews(
    pr: int,
    repo: str,
    *,
    head_sha: str,
    cwd: Path,
) -> tuple[int, int, bool, str | None]:
    owner, name = _repo_parts(repo)
    cursor: str | None = None
    pages = 0
    matches = 0
    for _ in range(MAX_REVIEW_PAGES):
        command = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={REVIEWS_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"pr={pr}",
        ]
        if cursor is not None:
            command.extend(["-F", f"cursor={cursor}"])
        payload, error = _run_json(
            command,
            cwd=cwd,
            allowed_codes={0},
            expected_type=dict,
        )
        if error:
            return matches, pages, False, error
        graphql_errors = payload.get("errors")
        if graphql_errors:
            return matches, pages, False, "GraphQL response contains errors"
        data = payload.get("data")
        repository = data.get("repository") if isinstance(data, dict) else None
        pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
        reviews = pull_request.get("reviews") if isinstance(pull_request, dict) else None
        if reviews is None:
            return matches, pages, False, "GraphQL reviews envelope is missing"
        if not isinstance(reviews, dict):
            return matches, pages, False, "GraphQL reviews must be an object"
        nodes = reviews.get("nodes")
        page_info = reviews.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            return matches, pages, False, "GraphQL review nodes/pageInfo are malformed"
        pages += 1
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                return matches, pages, False, f"review {index} is not an object"
            author = node.get("author")
            commit = node.get("commit")
            state = node.get("state")
            if author is None or commit is None:
                continue
            if not isinstance(author, dict) or not isinstance(commit, dict):
                return matches, pages, False, f"review {index} author/commit is malformed"
            login = author.get("login")
            oid = commit.get("oid")
            if login is not None and not isinstance(login, str):
                return matches, pages, False, f"review {index} author login is malformed"
            if oid is not None and not isinstance(oid, str):
                return matches, pages, False, f"review {index} commit oid is malformed"
            if state is not None and not isinstance(state, str):
                return matches, pages, False, f"review {index} state is malformed"
            if (
                _is_codex_login(login)
                and oid == head_sha
                and state in {"COMMENTED", "APPROVED"}
            ):
                matches += 1
        has_next = page_info.get("hasNextPage")
        if not isinstance(has_next, bool):
            return matches, pages, False, "GraphQL review hasNextPage must be boolean"
        if not has_next:
            return matches, pages, True, None
        next_cursor = page_info.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return matches, pages, False, "GraphQL review pagination cursor is missing"
        cursor = next_cursor
    return matches, pages, False, f"review pagination exceeded {MAX_REVIEW_PAGES} pages"


def _read_previous(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read previous watcher state: {exc}"
    if not isinstance(value, dict):
        return {}, "previous watcher state must be an object"
    return value, None


def _stored_count(previous: dict[str, Any], key: str) -> tuple[int, str | None]:
    value = previous.get(key, 0)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value, None
    return 0, f"previous watcher state has invalid {key}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _one_line(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text.replace(MARKER_START, "[watcher-marker]").replace(MARKER_END, "[watcher-marker]")


def _write_session_state(path: Path, block: str) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    marked = f"{MARKER_START}\n{block}\n{MARKER_END}"
    if MARKER_START in text and MARKER_END in text:
        before, rest = text.split(MARKER_START, 1)
        _, after = rest.split(MARKER_END, 1)
        updated = f"{before}{marked}{after}"
    else:
        updated = text.rstrip() + "\n\n" + marked + "\n"
    _atomic_write(path, updated)


def _notify(title: str, body: str, enabled: bool) -> None:
    if enabled and shutil.which("notify-send") is not None:
        _run(["notify-send", title, body], cwd=HOME)


def _disable_timer(watcher_id: str) -> None:
    if shutil.which("systemctl") is not None:
        _run(["systemctl", "--user", "disable", "--now", f"atlas-pr-watch@{watcher_id}.timer"], cwd=HOME)


def _classify(
    *,
    pr: dict[str, Any],
    errors: Sequence[str | None],
    unsafe_auto_merge: bool,
    head_mismatch: bool,
    worktree_dirty: bool,
    failures: list[str],
    required_failures: list[str],
    pending: list[str],
    required_pending: list[str],
    required_count: int,
    threads_complete: bool,
    unresolved_threads: list[dict[str, Any]],
    reviews_complete: bool,
    codex_head_review_count: int,
    reconciliation_code: int,
    review_changed: bool,
) -> str:
    if unsafe_auto_merge:
        return "attention"
    if pr.get("state") in {"MERGED", "CLOSED"}:
        return "closed"
    if errors or head_mismatch or worktree_dirty:
        return "attention"
    if pr.get("state") != "OPEN":
        return "attention"
    if failures or required_failures or required_count < 1 or reconciliation_code != 0:
        return "attention"
    if (
        not threads_complete
        or unresolved_threads
        or not reviews_complete
        or codex_head_review_count < 1
        or pr.get("isDraft") is not False
    ):
        return "attention"
    if str(pr.get("reviewDecision") or "").upper() == "CHANGES_REQUESTED":
        return "attention"
    if review_changed:
        return "review_changed"
    if pending or required_pending:
        return "pending"
    if pr.get("mergeStateStatus") != "CLEAN":
        return "attention"
    return "ready_for_human_merge"


def produce(watcher_id: str, *, config_dir: Path, state_dir: Path) -> tuple[int, dict[str, Any], str]:
    config_path = config_dir / f"{watcher_id}.env"
    if not config_path.exists():
        raise ValueError(f"watcher config not found: {config_path}")
    config = _load_env(config_path)
    repo_dir = Path(config.get("REPO_DIR", "")).expanduser()
    if not repo_dir.is_dir():
        raise ValueError(f"invalid REPO_DIR: {repo_dir}")
    pr_text = config.get("PR", "")
    if not pr_text.isdigit() or int(pr_text) < 1:
        raise ValueError("PR must be a positive integer")
    repo = config.get("REPO", "")
    _repo_parts(repo)

    status_path = state_dir / f"{watcher_id}.json"
    previous, previous_error = _read_previous(status_path)
    pr_initial, initial_error = _run_json(
        _pr_view_command(pr_text, repo), cwd=repo_dir, allowed_codes={0}, expected_type=dict
    )
    all_checks, all_checks_error = _run_json(
        _checks_command(pr_text, repo, required=False),
        cwd=repo_dir,
        allowed_codes=VALID_CHECK_EXIT_CODES,
        expected_type=list,
    )
    required_checks, required_error = _run_json(
        _checks_command(pr_text, repo, required=True),
        cwd=repo_dir,
        allowed_codes=VALID_CHECK_EXIT_CODES,
        expected_type=list,
    )
    base_ref = pr_initial.get("baseRefName") if isinstance(pr_initial, dict) else None
    if isinstance(base_ref, str) and base_ref:
        required_policy, required_policy_error = _run_json(
            _required_policy_command(repo, base_ref),
            cwd=repo_dir,
            allowed_codes={0},
            expected_type=dict,
        )
    else:
        required_policy = None
        required_policy_error = "initial PR metadata has no baseRefName for required-check discovery"
    review_data, reviews_error = _run_json(
        ["gh", "pr", "view", pr_text, "--repo", repo, "--comments", "--json", "comments,reviews"],
        cwd=repo_dir,
        allowed_codes={0},
        expected_type=dict,
    )
    reconciliation_command = [
        sys.executable,
        str(TRUSTED_RECONCILIATION_CHECKER),
        "--pr",
        pr_text,
        "--repo",
        repo,
    ]
    pr_final, final_error = _run_json(
        _pr_view_command(pr_text, repo), cwd=repo_dir, allowed_codes={0}, expected_type=dict
    )

    pr_initial = pr_initial if isinstance(pr_initial, dict) else {}
    pr = pr_final if isinstance(pr_final, dict) else pr_initial
    initial_shape_error = _validate_pr_metadata(pr_initial, label="initial") if pr_initial else None
    final_shape_error = _validate_pr_metadata(pr, label="final") if pr else None
    all_checks = all_checks if isinstance(all_checks, list) else []
    required_checks = required_checks if isinstance(required_checks, list) else []
    required_policy = required_policy if isinstance(required_policy, dict) else {}
    failures, pending, all_shape_error = _check_summary(all_checks, required=False)
    required_failures, required_pending, required_shape_error = _check_summary(required_checks, required=True)
    expected_required, required_policy_shape_error = _required_contexts(required_policy)
    reported_required = {
        item.get("name")
        for item in required_checks
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
    }
    missing_required = sorted(expected_required - reported_required)
    required_pending.extend(f"{name} (not reported)" for name in missing_required)
    required_contexts = expected_required | reported_required
    checks_errors = [
        item
        for item in (
            all_checks_error,
            required_error,
            required_policy_error,
            all_shape_error,
            required_shape_error,
            required_policy_shape_error,
        )
        if item
    ]
    checks_error = "; ".join(checks_errors) if checks_errors else None

    comments = review_data.get("comments") if isinstance(review_data, dict) else None
    reviews = review_data.get("reviews") if isinstance(review_data, dict) else None
    if not isinstance(comments, list) or not isinstance(reviews, list):
        comments = []
        reviews = []
        reviews_error = reviews_error or "review comments/reviews envelope is malformed"
    comment_count = len(comments)
    review_count = len(reviews)
    previous_comments, previous_comments_error = _stored_count(previous, "comment_count")
    previous_reviews, previous_reviews_error = _stored_count(previous, "review_count")
    previous_errors = [
        item
        for item in (previous_error, previous_comments_error, previous_reviews_error)
        if item
    ]
    previous_error = "; ".join(previous_errors) if previous_errors else None
    review_changed = comment_count > previous_comments or review_count > previous_reviews

    dirty_code, dirty_out, dirty_err = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    worktree_dirty = dirty_code != 0 or bool(dirty_out) or bool(dirty_err)
    expected_head = config.get("HEAD_SHA", "")
    initial_head = str(pr_initial.get("headRefOid") or "")
    final_head = str(pr.get("headRefOid") or "")
    codex_head_review_count, review_pages, reviews_complete, codex_reviews_error = (
        _fetch_codex_head_reviews(int(pr_text), repo, head_sha=final_head, cwd=repo_dir)
        if final_head
        else (0, 0, False, "PR head SHA missing before Codex review pagination")
    )
    unresolved_threads, thread_pages, threads_complete, threads_error = _fetch_threads(
        int(pr_text), repo, cwd=repo_dir
    )
    reconciliation_code, reconciliation_out, reconciliation_err = _run(reconciliation_command, cwd=repo_dir)
    pr_after_reviews, post_review_error = _run_json(
        _pr_view_command(pr_text, repo), cwd=repo_dir, allowed_codes={0}, expected_type=dict
    )
    pr_after_reviews = pr_after_reviews if isinstance(pr_after_reviews, dict) else {}
    post_review_shape_error = (
        _validate_pr_metadata(pr_after_reviews, label="post-review")
        if pr_after_reviews
        else None
    )
    post_review_head = str(pr_after_reviews.get("headRefOid") or "")
    initial_base = str(pr_initial.get("baseRefName") or "")
    final_base = str(pr.get("baseRefName") or "")
    post_review_base = str(pr_after_reviews.get("baseRefName") or "")
    observed_pr = pr_after_reviews if pr_after_reviews and post_review_head == final_head else pr
    head_mismatch = (
        not expected_head
        or not initial_head
        or initial_head != expected_head
        or final_head != initial_head
        or not post_review_head
        or post_review_head != final_head
    )
    base_mismatch_error = (
        "base branch changed during watcher observation"
        if (
            not initial_base
            or final_base != initial_base
            or not post_review_base
            or post_review_base != final_base
        )
        else None
    )
    unsafe_auto_merge = config.get("AUTO_MERGE", "0").lower() not in {"0", "false", "no", "off", ""}
    merge_error = "unsafe auto-merge config ignored; watcher cannot merge" if unsafe_auto_merge else ""
    errors = [
        item
        for item in (
            previous_error,
            initial_error,
            final_error,
            post_review_error,
            initial_shape_error,
            final_shape_error,
            post_review_shape_error,
            base_mismatch_error,
            checks_error,
            reviews_error,
            threads_error,
            codex_reviews_error,
        )
        if item
    ]
    state = _classify(
        pr=observed_pr,
        errors=errors,
        unsafe_auto_merge=unsafe_auto_merge,
        head_mismatch=head_mismatch,
        worktree_dirty=worktree_dirty,
        failures=failures,
        required_failures=required_failures,
        pending=pending,
        required_pending=required_pending,
        required_count=len(required_contexts),
        threads_complete=threads_complete,
        unresolved_threads=unresolved_threads,
        reviews_complete=reviews_complete,
        codex_head_review_count=codex_head_review_count,
        reconciliation_code=reconciliation_code,
        review_changed=review_changed,
    )

    now = dt.datetime.now().astimezone()
    try:
        poll_minutes = int(config.get("POLL_MINUTES", "30"))
    except ValueError as exc:
        raise ValueError("POLL_MINUTES must be an integer") from exc
    if not 1 <= poll_minutes <= 1440:
        raise ValueError("POLL_MINUTES must be between 1 and 1440")
    next_poll = now + dt.timedelta(minutes=poll_minutes)
    reconciliation_text = reconciliation_out or reconciliation_err
    status: dict[str, Any] = {
        "watcher_id": watcher_id,
        "label": config.get("LABEL", f"PR #{pr_text}"),
        "observed_at": now.isoformat(timespec="seconds"),
        "next_poll_at": next_poll.isoformat(timespec="seconds"),
        "state": state,
        "pr": observed_pr,
        "check_failures": failures,
        "check_pending": pending,
        "comment_count": comment_count,
        "review_count": review_count,
        "review_changed": review_changed,
        "head_mismatch": head_mismatch,
        "worktree_dirty": worktree_dirty,
        "merge_output": "",
        "merge_error": merge_error,
        "reconciliation_exit_code": reconciliation_code,
        "reconciliation_summary": "\n".join(reconciliation_text.splitlines()[-8:]),
        "view_error": "; ".join(
            item
            for item in (
                initial_error,
                final_error,
                post_review_error,
                initial_shape_error,
                final_shape_error,
                post_review_shape_error,
                base_mismatch_error,
            )
            if item
        ),
        "checks_error": checks_error or "",
        "reviews_error": reviews_error or "",
        "review_threads_error": threads_error or "",
        "codex_reviews_error": codex_reviews_error or "",
        "previous_state_error": previous_error or "",
        "worktree_status_error": dirty_err if dirty_code else "",
        "readiness": {
            "version": 1,
            "evaluated_head_sha": initial_head,
            "required_check_count": len(required_contexts),
            "required_checks_complete": (
                not required_error
                and not required_policy_error
                and not required_shape_error
                and not required_policy_shape_error
                and len(required_contexts) > 0
                and not required_failures
                and not required_pending
            ),
            "required_check_failures": required_failures,
            "required_check_pending": required_pending,
            "review_threads_complete": threads_complete,
            "review_thread_pages_fetched": thread_pages,
            "unresolved_review_threads": unresolved_threads,
            "codex_reviews_complete": reviews_complete,
            "codex_review_pages_fetched": review_pages,
            "codex_head_review_count": codex_head_review_count,
            "review_decision": observed_pr.get("reviewDecision"),
            "merge_state_status": observed_pr.get("mergeStateStatus"),
        },
    }
    _atomic_write(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")

    label = _one_line(status["label"])
    block = "\n".join(
        [
            f"Watcher: {_one_line(watcher_id)}",
            f"Observed: {now.strftime('%Y-%m-%d %H:%M %Z')}",
            f"Next poll: {next_poll.strftime('%Y-%m-%d %H:%M %Z')}",
            f"State: {state}",
            f"PR: #{_one_line(observed_pr.get('number', pr_text))} {_one_line(observed_pr.get('title', ''))}",
            f"URL: {_one_line(observed_pr.get('url', ''))}",
            f"Head: {_one_line(observed_pr.get('headRefName', ''))} @ {_one_line(post_review_head or final_head)}",
            f"Expected head: {_one_line(expected_head or 'missing')}",
            f"Merge state: {_one_line(observed_pr.get('mergeStateStatus', ''))}",
            f"Review decision: {_one_line(observed_pr.get('reviewDecision') or 'none')}",
            f"Worktree dirty: {'yes' if worktree_dirty else 'no'}",
            f"Failing checks: {_one_line(', '.join(failures) or 'none')}",
            f"Pending checks: {_one_line(', '.join(pending) or 'none')}",
            (
                f"Required checks: {len(required_contexts)} total, "
                f"{len(required_failures)} failed, {len(required_pending)} pending"
            ),
            f"Unresolved review threads: {len(unresolved_threads)} across {thread_pages} fetched page(s)",
            f"Current-head Codex reviews: {codex_head_review_count} across {review_pages} fetched page(s)",
            f"Reviews/comments: {review_count} reviews, {comment_count} comments",
            f"Review changed since last poll: {'yes' if review_changed else 'no'}",
            f"AI reconciliation: {'pass' if reconciliation_code == 0 else 'fail'}",
            "Watcher merge authority: forbidden",
        ]
    )
    session_state_value = config.get("SESSION_STATE", "")
    receipt_errors: list[str] = []
    if session_state_value:
        try:
            _write_session_state(Path(session_state_value).expanduser(), block)
        except OSError as exc:
            receipt_errors.append(f"session-state receipt failed: {exc}")
    log_path = state_dir / f"{watcher_id}.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n\n")
    except OSError as exc:
        receipt_errors.append(f"watcher log receipt failed: {exc}")
    if receipt_errors:
        block += "\nReceipt errors: " + _one_line("; ".join(receipt_errors))

    notify = config.get("NOTIFY", "1").lower() not in {"0", "false", "no", "off"}
    if state == "attention":
        _notify(
            f"Atlas PR watcher: {label} needs attention",
            _one_line(
                ", ".join(errors + failures + required_failures)
                or "review/readiness blocker"
            ),
            notify,
        )
    elif state == "ready_for_human_merge":
        _notify(
            f"Atlas PR watcher: {label} is green",
            "Ready for active-agent guards; watcher merge is forbidden.",
            notify,
        )
    elif state == "review_changed":
        _notify(f"Atlas PR watcher: {label} has new review activity", "Inspect before any merge decision.", notify)
    elif state == "closed":
        _disable_timer(watcher_id)
    return 0, status, block


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watcher_id")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _valid_watcher_id(args.watcher_id):
        print(f"invalid watcher id: {args.watcher_id!r}", file=sys.stderr)
        return 2
    try:
        code, _status, block = produce(
            args.watcher_id,
            config_dir=args.config_dir.expanduser(),
            state_dir=args.state_dir.expanduser(),
        )
    except (OSError, ValueError) as exc:
        print(f"PR watcher: {exc}", file=sys.stderr)
        return 2
    print(block)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
