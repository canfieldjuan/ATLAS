"""Auto-deploy blog posts: git commit + push + Vercel deploy hook."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


async def auto_deploy_blog(
    ui_path: str,
    slug: str,
    *,
    enabled: bool = False,
    branch: str = "dev",
    hook_url: str = "",
) -> dict[str, str | bool | int]:
    """Git-add the blog .ts files, commit, push, and fire deploy hook.

    Parameters are explicit so callers pass the correct config
    (consumer ExternalDataConfig or B2BChurnConfig).

    Returns a status dict. Never raises -- all steps are wrapped in try/except.
    """
    result: dict[str, str | bool | int] = {"deployed": False}

    if not enabled:
        result["skipped"] = "auto_deploy disabled"
        return result

    # Resolve repo root (walk up from ui_path to find .git/)
    repo_root = _find_repo_root(ui_path)
    if not repo_root:
        result["error"] = f"No .git/ found above {ui_path}"
        logger.warning(result["error"])
        return result

    blog_dir = Path(ui_path)
    ts_file = blog_dir / f"{slug}.ts"
    index_file = blog_dir / "index.ts"

    # Stage only the specific blog files
    files_to_add = [str(f) for f in (ts_file, index_file) if f.exists()]
    if not files_to_add:
        result["skipped"] = "no files to stage"
        return result

    try:
        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=str(repo_root), check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:
        result["error"] = f"git add failed: {exc}"
        logger.warning(result["error"])
        return result

    # Check if there are staged changes
    try:
        cp = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(repo_root), capture_output=True, timeout=10,
        )
        if cp.returncode == 0:
            result["skipped"] = "no staged changes"
            return result
    except Exception as exc:
        result["error"] = f"git diff check failed: {exc}"
        logger.warning(result["error"])
        return result

    # Commit
    try:
        subprocess.run(
            ["git", "commit", "-m", f"blog: auto-publish {slug}"],
            cwd=str(repo_root), check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:
        result["error"] = f"git commit failed: {exc}"
        logger.warning(result["error"])
        return result

    # Push current HEAD to the target branch (works even if local branch != target)
    try:
        subprocess.run(
            ["git", "push", "origin", f"HEAD:{branch}"],
            cwd=str(repo_root), check=True, capture_output=True, timeout=60,
        )
        result["pushed"] = True
    except Exception as exc:
        result["push_error"] = f"git push failed: {exc}"
        logger.warning(result["push_error"])
        # Continue -- deploy hook can still trigger a build from latest remote

    # Fire Vercel deploy hook regardless of push outcome.
    # Vercel may block git-triggered builds (unverified commits) but the
    # hook always works and rebuilds from whatever is on the branch.
    if hook_url:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(hook_url)
                result["hook_status"] = resp.status_code
        except Exception as exc:
            result["hook_error"] = str(exc)
            logger.warning("Deploy hook failed: %s", exc)

    result["deployed"] = bool(result.get("pushed") or result.get("hook_status"))
    logger.info("Auto-deployed blog post: %s (deployed=%s)", slug, result["deployed"])
    return result


def _find_repo_root(start_path: str) -> Path | None:
    """Walk up from start_path to find a directory containing .git/."""
    p = Path(start_path).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return None
