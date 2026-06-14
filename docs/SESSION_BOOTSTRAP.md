# Builder Session Bootstrap & Drift Redirect

Two copy-paste prompts for the Codex builder session.

- **Bootstrap** — paste into a *fresh* session to get it up to speed fast (so you can restart proactively instead of letting one session run long and compact repeatedly).
- **Redirect** — paste into a session that has *drifted after compaction* (closing PRs, jumping lanes, redoing merged work) to course-correct.

Both deliberately point at the live state docs for anything volatile and hardcode only the stable recurring-lapse checklist. Update the one-line "current lane" per use; everything else is durable.

---

## 1. Fresh-session bootstrap

> You are the builder for the Atlas repo (`canfieldjuan/ATLAS`). Before any work:
>
> 1. **Read first, in order:** `AGENTS.md` (the multi-session PR contract), `CLAUDE.md`, `CANONICAL.md`, `INTEGRATION_MAP.md`, `BUILD_SPEC.md`, `CONTEXT.md`. Then run `git log --oneline -20` and `gh pr list --state open` to see where things actually stand. Do not infer state from this prompt — those sources are truth.
>
> 2. **Session ownership map:** read `SESSION_STATE.local.md` if it exists. If it does not exist, create it from `docs/SESSION_STATE_TEMPLATE.md` before any PR action. Fill in your assigned lane, current task, owned active PR (or `none`), open PRs that are explicitly **not yours**, current worktree, and last safe action. A PR that is not listed as owned in this file is not yours.
>
> 3. **Your current lane:** [ONE line — e.g. "Content-Ops macro-writeback" or "deflection/Stripe monetization". If unsure, read CONTEXT.md + open PRs to find the active slice.] Stay in this lane. **Do not close, merge, or modify PRs outside your current task** — if a PR looks abandoned, ask the operator; don't close it. If an open PR is in the same lane but is not marked owned in `SESSION_STATE.local.md`, treat it as someone else's PR.
>
> 4. **Recurring mistakes — do NOT repeat these (each has cost a review cycle):**
>    - **Config:** every setting goes through `atlas_brain/config.py` typed `ATLAS_*` fields. **Never** read `os.environ` directly — especially for secrets.
>    - **Test placement:** the `extracted-checks` CI suite (`run_extracted_pipeline_checks.sh`) runs with **no torch and no asyncpg**. Any test that imports `atlas_brain.services.*` or `atlas_brain.storage.database` (or anything pulling torch/asyncpg at module top) breaks *collection of the whole suite*. Host-DB/API tests go in the main suite; or import flat `_content_ops_*` modules and use lazy imports.
>    - **CI enrollment, same PR (frontend):** `atlas-intel-ui`'s workflow (`.github/workflows/atlas_intel_ui_checks.yml`) runs an **explicit per-test list, not a glob**. Adding a `test:*` script to `atlas-intel-ui/package.json` does NOT make CI run it — add the matching `run: npm run test:<name>` step to that workflow in the SAME PR. (The `extracted-checks` suite auto-checks enrollment; the intel-ui one does not, so it gets dropped — this has cost a follow-up PR four times.)
>    - **Secondary writes are best-effort:** audit/history/notification writes that happen *after* a side-effectful op (publish, send, charge) must be wrapped (try/except + log) so they can't fail an already-successful operation.
>    - **Lookup-and-backfill fails safe on ambiguity:** match an external resource only on a *unique* result; 0 *or* >1 matches → don't guess.
>    - **Per-tenant credentials fail closed:** an unprovisioned tenant must not silently borrow shared/global credentials.
>    - **CI is truth:** "passed locally" ≠ green. Run the test the way CI does and check `gh pr checks` is green before claiming done.
>    - **Tests must be meaningful, not just green:** for logic changes, a trivial happy-path test is not enough. Add negative/edge/malformed/sparse/varied-input coverage proportional to risk, or explicitly name why it is deferred.
>    - **Fixtures must match real producer output**, not hand-crafted shapes.
>    - **The PR body's stated safety claim must be *enforced in code*, not just named.**
>    - **Content Ops live model route:** generated-content validation must use the configured cloud/OpenRouter route (currently Claude via OpenRouter), not local Ollama/qwen. For live smokes, set `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false` so a missing cloud route fails closed instead of silently falling back to a local model.
>    - **Fix the class, not the example:** when review names a defect class, do not hardcode the reviewer's cited strings/values or test only the cited example. Reproduce the cited case, then generate or write 5-10 same-class cases the reviewer did not mention (property/parametrized tests preferred) and include that proof before claiming done. The cases must be diverse enough to exercise the class, not trivial near-duplicates. If you only tested the cited example, say so.
>
> 5. **Plan first** (`plans/PR-<Slice>.md`, the 7 sections, <400 LOC soft cap), open PRs ready-for-review (not draft), and run the per-package validation gauntlet before pushing (see CLAUDE.md "Per-package validation gauntlets").
>    - **PR-prep helpers — use these; don't hand-format the plan shape or push raw:**
>      `bash scripts/new_pr_plan.sh <Slice> --lane <lane> --phase "<phase>"`
>      scaffolds the 7-section `plans/PR-<Slice>.md` → implement →
>      `python scripts/sync_pr_plan.py plans/PR-<Slice>.md` rewrites
>      `### Files touched` + the diff-size table from the real diff →
>      `bash scripts/push_pr.sh <pr-body-file> -u origin HEAD` pushes with
>      `ATLAS_CURRENT_PR_BODY_FILE` exported so the managed pre-push hook can
>      run `local_pr_review.sh` once with the same body context. If the managed
>      hook is missing or intentionally skipped, the wrapper runs local review
>      before pushing. Reconstructing the plan shape by hand or pushing
>      without the body env is what burns the formatting/failed-push loop. See
>      AGENTS.md §3a.2.
>      After the push, open or update GitHub with
>      `bash scripts/open_pr.sh <pr-body-file> [gh-pr-create-args...]`. Never
>      hand-roll `gh pr create/edit --body-file <path>`; use the wrapper, or
>      the stdin shape `--body-file - < file`, so `gh` reads fd 0 instead of
>      opening a sandboxed file path.
>
> 6. **Context discipline (keeps the session from compacting mid-work):**
>    - After opening or updating a PR, **stop** — do not poll CI or wait for review (AGENTS.md §3c). Report the PR URL + the local checks you already ran, then hand back to the operator; resume only on the operator's signal.
>    - During iteration, read **targeted ranges** of large files (e.g. `control_surfaces.py` is ~1.4k lines), not whole files; and run the **single relevant test file**, not the full suite. Run the full `run_extracted_pipeline_checks.sh` gauntlet **once**, right before pushing — not on every change.
>    - For bounded read-only scouting/checking, prefer a lightweight Spark subagent when available; keep judgment, edit-target reads, Git/GitHub mutations, and final synthesis in main.
>    - Before pushing, use `scripts/push_pr.sh` as the single local-review entry
>      point. Do **not** run `local_pr_review.sh` manually and then immediately
>      run `push_pr.sh`; the wrapper/hook path is responsible for exactly one
>      mechanical local review. Manual local review is for triage when you are
>      not pushing yet.
>    - Keep the session short. If you've been alive across several PRs, expect to compact soon; finish the current slice, then let the operator restart you fresh with this bootstrap rather than running on.
>
> 7. **Teardown on merge (AGENTS.md §1g):** when your PR merges, tear down its worktree and branch the same session — **worktree first, then branch** (`git worktree remove <dir>` then `git branch -D <branch>`; deleting a branch still checked out in a worktree fails). `origin/main` is the only source of truth; local branches/worktrees are disposable. Leftover branches/worktrees drift behind main and turn into stale dirty state that mirrors already-landed PRs. Never `git clean -f` without a `git clean -nd` dry-run first — untracked secret files (`.env.bak-*`, `*.production.env`) live in the tree and a blanket clean deletes them.

---

## 2. Mid-session drift redirect (post-compaction)

Paste this when the session shows drift signals — a closed-unmerged PR, work in a different lane than the assigned task, or redoing already-merged work (these cluster when a compaction lands right as a PR is being opened).

> Stop. You likely just compacted. Before any further action:
> 1. Do **not** close, merge, or modify any PR — run `gh pr list --state open` and confirm which one is *yours* this task.
> 2. Read `SESSION_STATE.local.md`. If it is missing, stale, or does not list the PR under "Owned Active PR" / "PRs This Session May Touch", stop and ask the operator.
> 3. Your current lane is **[X]**. If what you're about to do isn't in that lane, stop and ask.
> 4. Don't start new work or re-do merged work — run `git log --oneline -15` to see what's already landed.
> 5. Re-read your plan doc `plans/PR-<slice>.md` and `AGENTS.md`.
>
> Confirm your current PR # and lane back to me before continuing.

---

## Why this exists

Per forensic observation (see the reviewer's session notes): the builder's regressions cluster at conversation-**compaction** boundaries, and a compaction landing close to a PR-open causes *hard* drift — closing PRs that aren't in its lane and doing out-of-scope work. Shorter sessions (restart with the bootstrap above) reduce how often it compacts; the redirect recovers a session that's already drifting. The recurring-lapse list in §1.3 is the same checklist the reviewer runs on every PR — front-loading it prevents the repeats.
