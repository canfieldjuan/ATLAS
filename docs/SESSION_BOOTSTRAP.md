# Builder Session Bootstrap & Drift Redirect

Two copy-paste prompts for the Codex builder session.

- **Bootstrap** — paste into a *fresh* session to get it up to speed fast (so you can restart proactively instead of letting one session run long and compact repeatedly).
- **Redirect** — paste into a session that has *drifted after compaction* (closing PRs, jumping lanes, redoing merged work) to course-correct.

Both deliberately point at the live state docs for anything volatile and hardcode only the stable recurring-lapse checklist. Update the one-line "current lane" per use; everything else is durable.

---

## 1. Fresh-session bootstrap

> You are the builder for the Atlas repo (`canfieldjuan/ATLAS`). Before any work:
>
> 1. **Read first, in order:** `AGENTS.md` (the multi-session PR contract), `CLAUDE.md`, `docs/CURRENT_PRODUCT_DISCIPLINE.md`, `CANONICAL.md`, `INTEGRATION_MAP.md`, `CONTEXT.md`. `BUILD_SPEC.md` is deprecated historical context, not the current product roadmap or DoD. `CONTEXT.md` is debt/session-note context only; do not use it as current roadmap, priority, or product state without live verification. Then run `git log --oneline -20` and `gh pr list --state open` to see where things actually stand. Do not infer state from this prompt — those sources are truth.
>
> 2. **Session ownership map:** pick a session-scoped state file before any PR action. Use `SESSION_STATE.<session-id>.local.md` at the repo/worktree root (for example `SESSION_STATE.codex-macro-1979.local.md`); use legacy `SESSION_STATE.local.md` only when one active session owns that worktree. Export `ATLAS_SESSION_STATE_FILE=<that file>`, then read it if it exists or create it from `docs/SESSION_STATE_TEMPLATE.md`. Fill in its canonical `Current lane:`, current task, Spark/subagent routing used or considered, owned active PR (or `none`), open PRs that are explicitly **not yours**, current worktree, and last safe action. The plan scaffold resolves an explicit `--state-file`, then `ATLAS_SESSION_STATE_FILE`, then legacy state. A PR that is not listed as owned in this file is not yours.
>
> 3. **Your current lane:** [ONE line — e.g. "Content-Ops macro-writeback" or "deflection/Stripe monetization". If unsure, read CONTEXT.md + open PRs to find the active slice.] Stay in this lane. **Do not close, merge, or modify PRs outside your current task** — if a PR looks abandoned, ask the operator; don't close it. If an open PR is in the same lane but is not marked owned in your session state file, treat it as someone else's PR.
>
> 4. **Recurring mistakes — do NOT repeat these (each has cost a review cycle):**
>    - **Config:** every setting goes through `atlas_brain/config.py` typed `ATLAS_*` fields. **Never** read `os.environ` directly — especially for secrets.
>    - **Test placement:** the `extracted-checks` CI suite (`run_extracted_pipeline_checks.sh`) runs with **no torch and no asyncpg**. Any test that imports `atlas_brain.services.*` or `atlas_brain.storage.database` (or anything pulling torch/asyncpg at module top) breaks *collection of the whole suite*. Host-DB/API tests go in the main suite; or import flat `_content_ops_*` modules and use lazy imports.
>    - **CI enrollment, same PR (frontend):** `atlas-intel-ui`'s workflow (`.github/workflows/atlas_intel_ui_checks.yml`) runs an **explicit per-test list, not a glob**. Adding a `test:*` script to `atlas-intel-ui/package.json` does NOT make CI run it — add the matching `run: npm run test:<name>` step to that workflow in the SAME PR. (The `extracted-checks` suite auto-checks enrollment; the intel-ui one does not, so it gets dropped — this has cost a follow-up PR four times.)
>    - **Secondary writes are best-effort:** audit/history/notification writes that happen *after* a side-effectful op (publish, send, charge) must be wrapped (try/except + log) so they can't fail an already-successful operation.
>    - **Lookup-and-backfill fails safe on ambiguity:** match an external resource only on a *unique* result; 0 *or* >1 matches → don't guess.
>    - **Per-tenant credentials fail closed:** an unprovisioned tenant must not silently borrow shared/global credentials.
>    - **CI is truth:** "passed locally" ≠ green. Before you push, run the relevant checks the way CI does and report only what actually ran. After the PR is pushed/opened/updated, confirming merge readiness is the watcher/operator's job after handoff, not an in-session polling loop.
>    - **Tests must be meaningful, not just green:** for logic changes, a trivial happy-path test is not enough. Add negative/edge/malformed/sparse/varied-input coverage proportional to risk, or explicitly name why it is deferred.
>    - **New surfaces need reachability proof:** if a slice adds a runtime, workflow, UI, report, billing, delivery, or public contract surface, exercise the real entrypoint and assert an observable result. A unit-tested helper is not enough to prove the surface is wired.
>    - **Vertical product proof first:** default to the smallest end-to-end buyer-visible or operator-visible path. Defer hardening, harness polish, workflow tooling, and process codification unless it blocks that proof, fixes a real safety/security/privacy/money risk, or is justified by a recent product run that failed because the infrastructure was missing.
>    - **No product-shape changes without operator consent:** do not change report/snapshot/email/PDF structure, landing-page claims/copy, pricing/checkout/subscription surfaces, buyer-visible tables/cards/sections/labels, customer-facing promises, or output semantics unless the operator explicitly approved that product shape in this session or the accepted issue/plan.
>    - **Fixtures must match real producer output**, not hand-crafted shapes.
>    - **Identity guards need canonical producer/consumer parity:** enumerate every identity channel the canonical producer exposes, trace each through the consumer query/matcher, and drive at least one behavioral test through the real producer with only the external adapter faked. A consumer-authored key fixture cannot prove a channel was not omitted.
>    - **The PR body's stated safety claim must be *enforced in code*, not just named.**
>    - **A later guard does not make an operation fail-closed:** for any rejected mutation flow, trace the first possible write across providers/helpers and require the rejection predicate to guard that first mutation. Prove the rejected path leaves a zero-write mutation set.
>    - **Problem-derived contract before code:** before editing, write the root cause, what the correct fix must touch/change, and what must not change in the plan. Build only to that contract. Before pushing, reconstruct your own diff cold with `file:line` citations and put the result in the PR body under `## Cold diff reconstruction`; lead with gaps and do not call the slice done while any gap stands.
>    - **Content Ops live model route:** generated-content validation must use the configured cloud/OpenRouter route (currently Claude via OpenRouter), not local Ollama/qwen. For live smokes, set `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false` so a missing cloud route fails closed instead of silently falling back to a local model.
>    - **Fix the class, not the example:** when review names a defect class, do not hardcode the reviewer's cited strings/values or test only the cited example. Reproduce the cited case, then generate or write 5-10 same-class cases the reviewer did not mention (property/parametrized tests preferred) and include that proof before claiming done. The cases must be diverse enough to exercise the class, not trivial near-duplicates. If you only tested the cited example, say so. For **open-input guards** (privacy/safety classifiers, sanitizers, parser-admission rules over free text or nested structures) the stronger bar in `AGENTS.md` section 3k.1 / `docs/GUARD_CLASS_CLOSURE.md` applies (a fail-closed / evidence-gated choke point plus a property test in the bar's form -- including the open-category evidence-gated substitute when no list closes the recognizer; see that canonical doc for the full bar, do not restate it here), not just 5-10 cases.
>
> 5. **Plan first** (`plans/PR-<Slice>.md`, the 7 top-level sections plus nested Problem-derived and Review Contracts, <400 LOC soft cap), open PRs ready-for-review (not draft), and run the per-package validation gauntlet before pushing (see CLAUDE.md "Per-package validation gauntlets"). A non-empty human diff made only of regular Git blobs with `.md` as their sole suffix may omit the plan only with a PR body beginning `Docs-only: true`; every other human diff adds exactly one plan.
>    - **PR-prep helpers — use these; don't hand-format the plan shape or push raw:**
>      `bash scripts/new_pr_plan.sh <Slice> --lane <lane> --phase "<phase>"`
>      scaffolds the 7-top-level-section `plans/PR-<Slice>.md` plus its nested contracts → implement →
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
>    - The Atlas handoff baton is your session state file plus watcher status output; do not add a second marker, timer, or poller just to wait for CI.
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
> 2. Read the session state file named by `ATLAS_SESSION_STATE_FILE`. If that variable is unset, pick the intended session file before continuing. If the file is missing, stale, or does not list the PR under "Owned Active PR" / "PRs This Session May Touch", stop and ask the operator.
> 3. Your current lane is **[X]**. If what you're about to do isn't in that lane, stop and ask.
> 4. Don't start new work or re-do merged work — run `git log --oneline -15` to see what's already landed.
> 5. Re-read your plan doc `plans/PR-<slice>.md` and `AGENTS.md`.
>
> Confirm your current PR # and lane back to me before continuing.

---

## Why this exists

Per forensic observation (see the reviewer's session notes): the builder's regressions cluster at conversation-**compaction** boundaries, and a compaction landing close to a PR-open causes *hard* drift — closing PRs that aren't in its lane and doing out-of-scope work. Shorter sessions (restart with the bootstrap above) reduce how often it compacts; the redirect recovers a session that's already drifting. The recurring-lapse list in §1.3 is the same checklist the reviewer runs on every PR — front-loading it prevents the repeats.
