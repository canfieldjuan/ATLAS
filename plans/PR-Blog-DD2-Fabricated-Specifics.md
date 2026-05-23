# PR-Blog-DD2-Fabricated-Specifics

Ownership lane: `content-ops/blog-dd2-fabricated-specifics`

## Why this slice exists

Deep-pass DD2: fabricated / garbled / incoherent specific claims. Two posts carry
clearly-wrong specifics around the same per-seat-pricing datum:

- **asana** quotes a competitor as **"Ira" at $9.05/user** (FAQ + body). "Ira" is
  not a project-management tool; it's a garble of **Jira** -- the same $9.05/seat
  datum is attributed to "Jira" in three other posts (best-project-management,
  switch-to-asana). Fix the garbled name.
- **best-project-management-for-201-1000** frames per-seat pricing with an
  incoherent **"Nx multiplier"**: "92x at $9.05 for Jira, 78x at $9 for Monday,
  62x at $10.99 for Asana" -- the multipliers are inverted versus the prices
  (cheaper Jira shown with a *higher* multiplier) and have no defined base. Drop
  the multiplier framing; keep the real per-seat prices ($9.05 / $9 / $10.99) and
  the coherent scale math ($9 x 200 = $1800/mo).

## Scope (this PR)

- `asana-deep-dive-2026-04.ts` L137 (FAQ) + L193 (body): "Ira" -> "Jira".
- `best-project-management-for-201-1000-2026-04.ts` L157 (FAQ), L178, L284, L356-360
  (list), L393: remove the "Nx multiplier" / "the multiplier matters" framing; keep
  the per-seat prices and the $1800/$2198-per-200-seat math.

### Files touched

- `plans/PR-Blog-DD2-Fabricated-Specifics.md`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`

## Mechanism

Garbled name corrected to the real, cross-post-consistent competitor (Jira); the
incoherent multiplier is removed while the genuinely-grounded numbers (per-seat
prices, the $265 renewal charge, the linear seat math) are preserved. No new
claims invented.

## Intentional

- **Jira, not a guess.** $9.05/seat is attributed to "Jira" in three sibling posts;
  "Ira" is the outlier garble.
- **Kept the real math.** best-pm's "$9/seat = $1800/mo for 200 seats" is correct
  and stays; only the undefined "Nx multiplier" is dropped.

## Deferred

- **DD3 (prose-vs-chart inversion)**: the widened `detectProseVsChartMetric` now
  catches "X dominates" phrasing and flags asana ("Pricing dominates" vs pain-radar
  top `data_migration`=6.8), linode, and mailchimp (microsoft-defender + workday are
  already fixed in the open #831). DD3 is NOT in this PR -- asana's case is entangled
  with the parked chart-provenance question (data_migration=6.8 is a suspicious
  outlier), and a clean fix needs the frequency-vs-urgency framing resolved per post.
  asana therefore still shows `prose_vs_chart_metric` (and `pain_as_strength`, fixed
  by the open #844 -- this branch is off origin/main).

## Verification

- Grep confirms no residual "Ira" / "multiplier" / "Nx" garbles in either post.
- best-project-management re-audited (`--slug=`): clean. asana shows only the
  deferred DD3 `prose_vs_chart_metric` and the #844 `pain_as_strength` (both
  pre-existing relative to this DD2 change; not introduced here).
- asana file-overlaps the open #844 on different lines (DD2 L137/L193 vs DD1 chart
  + strengths/weakness prose) -- auto-merge.

## Estimated diff size

| Area | LOC |
|---|---:|
| asana ("Ira" -> "Jira") | ~4 |
| best-project-management (multiplier removals) | ~16 |
| Plan doc | ~75 |
| **Total** | **~95** |
