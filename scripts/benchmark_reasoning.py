#!/usr/bin/env python3
"""
Benchmark reasoning models on OpenRouter against Claude Opus 4.6.

Runs the competitive_intelligence skill prompt through multiple models,
captures quality/cost/latency, then has Opus grade each response.

Usage:
    python scripts/benchmark_reasoning.py [--judge-only]

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Models to benchmark (OpenRouter model IDs)
# ---------------------------------------------------------------------------

MODELS = [
    # Round 2 baseline (beat Opus in round 2)
    {
        "id": "openai/o4-mini",
        "label": "o4-mini",
        "tier": "baseline",
        "input_cost": 1.10,
        "output_cost": 4.40,
    },
    # Round 3 challengers
    {
        "id": "minimax/minimax-m2.5",
        "label": "MiniMax M2.5",
        "tier": "value",
        "input_cost": 0.27,
        "output_cost": 0.95,
    },
    {
        "id": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "tier": "value",
        "input_cost": 0.45,
        "output_cost": 2.20,
    },
    {
        "id": "google/gemini-3.1-pro-preview",
        "label": "Gemini 3.1 Pro",
        "tier": "frontier",
        "input_cost": 2.00,
        "output_cost": 12.00,
    },
    {
        "id": "z-ai/glm-5",
        "label": "GLM-5",
        "tier": "value",
        "input_cost": 0.72,
        "output_cost": 2.30,
    },
    {
        "id": "openrouter/hunter-alpha",
        "label": "Hunter Alpha",
        "tier": "free",
        "input_cost": 0.00,
        "output_cost": 0.00,
    },
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"

# ---------------------------------------------------------------------------
# Grading rubric (sent to Opus as the judge)
# ---------------------------------------------------------------------------

GRADING_PROMPT = """\
You are evaluating LLM outputs for a competitive intelligence analysis task.
The task asks the model to analyze complaint/review data and produce a structured
JSON report with competitive flows, feature gaps, buyer personas, brand vulnerability
scores, insights, and recommendations.

Grade the response on these 6 dimensions (0-10 each):

1. **JSON Validity** (0-10): Is the output valid JSON? Does it parse without error?
   10 = perfect JSON. 5 = minor issues (trailing comma, wrapped in markdown).
   0 = not JSON at all.

2. **Schema Compliance** (0-10): Does it include all required top-level fields
   (analysis_text, competitive_flows, feature_gaps, buyer_personas,
   brand_vulnerability, insights, recommendations)? Are nested fields present?
   10 = all fields with correct types. 5 = most fields. 0 = missing structure.

3. **Analytical Depth** (0-10): Quality of reasoning. Does it identify real
   patterns in the data? Are vulnerability scores computed correctly using the
   formula? Are competitive flows mapped with meaningful reasons?
   10 = deep, multi-dimensional analysis. 5 = surface-level. 0 = generic filler.

4. **Data Fidelity** (0-10): Does it only reference data that was in the input?
   Are numbers accurate? Does it avoid hallucinating brands or statistics?
   10 = perfectly grounded. 5 = mostly accurate. 0 = fabricates freely.

5. **Temporal Anchoring** (0-10): Does every statistic include a timeframe?
   The prompt requires anchoring claims with dates from data_context.
   10 = every claim anchored. 5 = some anchored. 0 = no timeframes.

6. **Actionability** (0-10): Are insights specific and actionable? Do
   recommendations have clear urgency and reasoning? Is the analysis_text
   suitable for a push notification (under 600 words, leads with critical finding)?
   10 = immediately actionable. 5 = vague but directional. 0 = generic advice.

Respond with ONLY a JSON object:
{
  "json_validity": <0-10>,
  "schema_compliance": <0-10>,
  "analytical_depth": <0-10>,
  "data_fidelity": <0-10>,
  "temporal_anchoring": <0-10>,
  "actionability": <0-10>,
  "total": <sum of all scores, max 60>,
  "percentage": <total/60 * 100, rounded to 1 decimal>,
  "notes": "Brief explanation of strengths and weaknesses"
}
"""


# ---------------------------------------------------------------------------
# Data fetching (real DB data, same as competitive_intelligence task)
# ---------------------------------------------------------------------------


async def fetch_benchmark_payload() -> dict:
    """Fetch real data from the DB to use as the benchmark payload."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()

    # Import the actual fetchers from competitive_intelligence
    from atlas_brain.autonomous.tasks.competitive_intelligence import (
        _fetch_brand_health,
        _fetch_competitive_flows,
        _fetch_feature_gaps,
        _fetch_buyer_personas,
        _fetch_safety_signals,
        _fetch_loyalty_churn,
        _fetch_data_context,
    )

    logger.info("Fetching benchmark data from DB...")
    (
        brand_health,
        competitive_flows,
        feature_gaps,
        buyer_personas,
        safety_signals,
        loyalty_churn,
        data_context,
    ) = await asyncio.gather(
        _fetch_brand_health(pool),
        _fetch_competitive_flows(pool),
        _fetch_feature_gaps(pool),
        _fetch_buyer_personas(pool),
        _fetch_safety_signals(pool),
        _fetch_loyalty_churn(pool),
        _fetch_data_context(pool),
        return_exceptions=True,
    )

    # Convert exceptions to empty
    for name, val in [
        ("brand_health", brand_health),
        ("competitive_flows", competitive_flows),
        ("feature_gaps", feature_gaps),
        ("buyer_personas", buyer_personas),
        ("safety_signals", safety_signals),
        ("loyalty_churn", loyalty_churn),
    ]:
        if isinstance(val, Exception):
            logger.warning("%s fetch failed: %s", name, val)

    if isinstance(data_context, Exception):
        data_context = {}

    from datetime import date

    payload = {
        "date": str(date.today()),
        "data_context": data_context if not isinstance(data_context, Exception) else {},
        "total_brands": len(brand_health) if not isinstance(brand_health, Exception) else 0,
        "brand_health": (brand_health if not isinstance(brand_health, Exception) else [])[:8],
        "competitive_flows": (competitive_flows if not isinstance(competitive_flows, Exception) else [])[:10],
        "feature_gaps": (feature_gaps if not isinstance(feature_gaps, Exception) else [])[:8],
        "buyer_personas": (buyer_personas if not isinstance(buyer_personas, Exception) else [])[:6],
        "safety_signals": (safety_signals if not isinstance(safety_signals, Exception) else [])[:6],
        "loyalty_churn": (loyalty_churn if not isinstance(loyalty_churn, Exception) else [])[:8],
    }

    await close_database()
    logger.info(
        "Payload ready: %d brands, %d flows, %d gaps",
        len(payload["brand_health"]),
        len(payload["competitive_flows"]),
        len(payload["feature_gaps"]),
    )
    return payload


# ---------------------------------------------------------------------------
# OpenRouter API calls
# ---------------------------------------------------------------------------


async def call_openrouter(
    client: httpx.AsyncClient,
    model_id: str,
    system_prompt: str,
    user_content: str,
    api_key: str,
    max_tokens: int = 4096,
    temperature: float = 0.4,
) -> dict:
    """Call a model via OpenRouter and return result dict."""
    t0 = time.monotonic()

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # JSON mode for models that support it
    if not model_id.startswith("openai/o"):  # o3/o4-mini use different format
        body["response_format"] = {"type": "json_object"}

    try:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://atlas-brain.local",
                "X-Title": "Atlas Reasoning Benchmark",
            },
            json=body,
            timeout=300,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})

        return {
            "success": True,
            "content": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": round(elapsed_ms),
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "success": False,
            "content": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round(elapsed_ms),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(judge_only: bool = False):
    """Run the full benchmark."""
    # Load env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"reasoning_benchmark_{timestamp}.json"
    responses_dir = RESULTS_DIR / f"responses_{timestamp}"
    responses_dir.mkdir(exist_ok=True)

    # Load skill prompt
    skill_path = (
        Path(__file__).resolve().parent.parent
        / "atlas_brain"
        / "skills"
        / "digest"
        / "competitive_intelligence.md"
    )
    skill_content = skill_path.read_text()
    # Strip YAML frontmatter
    if skill_content.startswith("---"):
        _, _, skill_content = skill_content.split("---", 2)
    system_prompt = skill_content.strip()

    if not judge_only:
        # Fetch real payload
        payload = await fetch_benchmark_payload()
        user_content = json.dumps(payload, default=str)

        # Save payload for reproducibility
        (RESULTS_DIR / f"payload_{timestamp}.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )

        logger.info("=" * 70)
        logger.info("REASONING MODEL BENCHMARK")
        logger.info("Payload: %d chars, %d brands", len(user_content), len(payload["brand_health"]))
        logger.info("Models: %d", len(MODELS))
        logger.info("=" * 70)

        results = []
        async with httpx.AsyncClient() as client:
            for model in MODELS:
                logger.info("")
                logger.info("--- %s (%s) ---", model["label"], model["id"])

                result = await call_openrouter(
                    client,
                    model["id"],
                    system_prompt,
                    user_content,
                    api_key,
                    max_tokens=16384,
                    temperature=0.4,
                )

                # Compute cost
                input_cost = (result["input_tokens"] / 1_000_000) * model["input_cost"]
                output_cost = (result["output_tokens"] / 1_000_000) * model["output_cost"]
                total_cost = input_cost + output_cost

                entry = {
                    **model,
                    **result,
                    "input_cost_usd": round(input_cost, 6),
                    "output_cost_usd": round(output_cost, 6),
                    "total_cost_usd": round(total_cost, 6),
                }
                results.append(entry)

                # Save individual response
                resp_file = responses_dir / f"{model['label'].replace(' ', '_').lower()}.txt"
                resp_file.write_text(result["content"] or result.get("error") or "empty")

                if result["success"]:
                    # Quick JSON validity check
                    try:
                        json.loads(result["content"])
                        json_valid = True
                    except (json.JSONDecodeError, TypeError):
                        json_valid = False

                    logger.info(
                        "  OK | %d in / %d out tokens | $%.4f | %dms | JSON: %s",
                        result["input_tokens"],
                        result["output_tokens"],
                        total_cost,
                        result["latency_ms"],
                        "valid" if json_valid else "INVALID",
                    )
                else:
                    logger.error("  FAILED: %s (%dms)", result["error"], result["latency_ms"])

        # Save raw results
        results_file.write_text(json.dumps(results, indent=2, default=str))
        logger.info("")
        logger.info("Raw results saved to %s", results_file)
    else:
        # Load most recent results
        result_files = sorted(RESULTS_DIR.glob("reasoning_benchmark_*.json"))
        if not result_files:
            logger.error("No benchmark results found. Run without --judge-only first.")
            return
        results_file = result_files[-1]
        results = json.loads(results_file.read_text())
        responses_dir = RESULTS_DIR / results_file.stem.replace("reasoning_benchmark_", "responses_")
        logger.info("Loading results from %s", results_file)

    # -----------------------------------------------------------------------
    # Phase 2: Judge each response with Opus
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("JUDGING RESPONSES WITH OPUS")
    logger.info("=" * 70)

    async with httpx.AsyncClient() as client:
        for entry in results:
            if not entry.get("success"):
                entry["grade"] = {
                    "json_validity": 0, "schema_compliance": 0,
                    "analytical_depth": 0, "data_fidelity": 0,
                    "temporal_anchoring": 0, "actionability": 0,
                    "total": 0, "percentage": 0.0,
                    "notes": f"Model failed: {entry.get('error', 'unknown')}",
                }
                continue

            label = entry["label"]
            logger.info("Judging: %s ...", label)

            judge_input = (
                f"## Model Output to Grade\n\n"
                f"```\n{entry['content'][:12000]}\n```"
            )

            grade_result = await call_openrouter(
                client,
                "anthropic/claude-opus-4-6",
                GRADING_PROMPT,
                judge_input,
                api_key,
                max_tokens=1024,
                temperature=0.1,
            )

            if grade_result["success"]:
                try:
                    # Strip thinking tags / markdown
                    grade_text = grade_result["content"]
                    import re
                    grade_text = re.sub(r"<think>[\s\S]*?</think>", "", grade_text).strip()
                    grade_text = re.sub(r"^```(?:json)?\s*\n?", "", grade_text)
                    grade_text = re.sub(r"\n?```\s*$", "", grade_text).strip()

                    grade = json.loads(grade_text)
                    entry["grade"] = grade
                    logger.info(
                        "  %s: %s/60 (%.1f%%) - %s",
                        label, grade["total"], grade["percentage"],
                        grade.get("notes", "")[:80],
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("  Failed to parse grade for %s: %s", label, e)
                    entry["grade"] = {"error": str(e), "raw": grade_result["content"][:500]}
            else:
                logger.warning("  Judge call failed for %s: %s", label, grade_result["error"])
                entry["grade"] = {"error": grade_result["error"]}

            # Judge cost tracking
            entry["judge_tokens"] = grade_result["input_tokens"] + grade_result["output_tokens"]

    # Save final results with grades
    final_file = results_file.with_name(results_file.stem + "_graded.json")
    final_file.write_text(json.dumps(results, indent=2, default=str))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info("")

    # Sort by grade percentage descending
    graded = [r for r in results if isinstance(r.get("grade"), dict) and "percentage" in r["grade"]]
    graded.sort(key=lambda r: r["grade"]["percentage"], reverse=True)

    baseline_pct = None
    for r in graded:
        if r["id"] == "openai/o4-mini":
            baseline_pct = r["grade"]["percentage"]
            break

    header = f"{'Model':<25} {'Tier':<10} {'Score':>6} {'%':>7} {'vs Opus':>8} {'Cost':>8} {'Latency':>8} {'Out Tok':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in graded:
        pct = r["grade"]["percentage"]
        vs_opus = f"{pct / baseline_pct * 100:.0f}%" if baseline_pct else "N/A"
        logger.info(
            "%-25s %-10s %5d/60 %6.1f%% %7s $%6.4f %6dms %7d",
            r["label"],
            r["tier"],
            r["grade"]["total"],
            pct,
            vs_opus,
            r.get("total_cost_usd", 0),
            r.get("latency_ms", 0),
            r.get("output_tokens", 0),
        )

    logger.info("")
    logger.info("Results saved to: %s", final_file)

    # Cost efficiency ranking
    logger.info("")
    logger.info("COST EFFICIENCY (quality per dollar):")
    for r in graded:
        cost = r.get("total_cost_usd", 0)
        if cost > 0:
            efficiency = r["grade"]["percentage"] / cost
            logger.info("  %-25s %.1f quality/$", r["label"], efficiency)


def main():
    judge_only = "--judge-only" in sys.argv
    asyncio.run(run_benchmark(judge_only=judge_only))


if __name__ == "__main__":
    main()
