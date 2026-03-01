# Atlas Technical Debt Baseline Audit

Date: 2026-03-01
Scope: `/home/juan-canfield/Desktop/Atlas`
Audit type: Automated repository scan

## Method

1. Count tracked source files by language and test area.
2. Read `.env.local` key inventory.
3. Run deterministic debt rule checks against known hotspots.
4. Run `pytest --collect-only -q` for collectability signal.
5. Rank with `priority_score = (impact * frequency * risk) / effort`.

## Verified Baseline Facts

- Python files: 579
- TS/TSX files: 139
- Test files: 74
- `.env.local` path verified: `/home/juan-canfield/Desktop/Atlas/.env.local`
- `.env.local` keys currently present:
  - `VLLM_PYTHON`
  - `VLLM_MODEL`
  - `VLLM_HOST`
  - `VLLM_PORT`
  - `VLLM_QUANTIZATION`
  - `VLLM_DTYPE`
  - `VLLM_GPU_MEMORY_UTILIZATION`
  - `VLLM_MAX_NUM_BATCHED_TOKENS`
  - `VLLM_MAX_NUM_SEQS`
  - `VLLM_ENABLE_PREFIX_CACHING`
- Test collection signal:
  - `1317 tests collected`
  - `0 collection errors`
  - `pytest return code: 0`

## Prioritized Top Debt Items

| Rank | ID | Score | Category | Finding | Verified Evidence |
|---|---|---:|---|---|---|
| - | - | - | - | No active debt findings detected by current rules. | - |
