#!/usr/bin/env python3
"""
Benchmark reasoning models on OpenRouter against the production path.

Runs the competitive_intelligence skill prompt through multiple models,
captures parseability, schema coverage, cost, and latency using the same
payload construction and parse recovery path as production.

Usage:
    python scripts/benchmark_reasoning.py

Requires: OPENROUTER_API_KEY in .env or environment.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, datetime
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
    # Baseline
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
# Data fetching (real DB data, same as competitive_intelligence task)
# ---------------------------------------------------------------------------


async def fetch_benchmark_payload() -> dict:
    """Fetch the exact production payload used by competitive_intelligence."""
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()

    from atlas_brain.autonomous.tasks.competitive_intelligence import (
        _build_llm_payload,
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

    payload = _build_llm_payload(
        date.today(),
        brand_health if not isinstance(brand_health, Exception) else [],
        competitive_flows if not isinstance(competitive_flows, Exception) else [],
        feature_gaps if not isinstance(feature_gaps, Exception) else [],
        buyer_personas if not isinstance(buyer_personas, Exception) else [],
        safety_signals if not isinstance(safety_signals, Exception) else [],
        loyalty_churn if not isinstance(loyalty_churn, Exception) else [],
        data_context if not isinstance(data_context, Exception) else {},
    )

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


def parse_pipeline_response(raw_text: str) -> dict:
    """Parse model output with the same recovery path production uses."""
    from atlas_brain.pipelines.llm import parse_json_response

    return parse_json_response(raw_text or "", recover_truncated=True)


def score_pipeline_output(parsed_output: dict) -> dict:
    """Deterministic structural score for competitive-intelligence output."""
    required_fields = {
        "analysis_text": str,
        "competitive_flows": list,
        "feature_gaps": list,
        "buyer_personas": list,
        "brand_vulnerability": list,
        "insights": list,
        "recommendations": list,
    }
    present = 0
    typed = 0
    for field, expected_type in required_fields.items():
        if field in parsed_output:
            present += 1
        if isinstance(parsed_output.get(field), expected_type):
            typed += 1

    analysis_text = parsed_output.get("analysis_text")
    analysis_words = len(analysis_text.split()) if isinstance(analysis_text, str) else 0
    return {
        "required_fields_present": present,
        "required_fields_expected": len(required_fields),
        "required_fields_typed": typed,
        "analysis_word_count": analysis_words,
        "parse_fallback": bool(parsed_output.get("_parse_fallback")),
        "flow_count": len(parsed_output.get("competitive_flows", [])) if isinstance(parsed_output.get("competitive_flows"), list) else 0,
        "gap_count": len(parsed_output.get("feature_gaps", [])) if isinstance(parsed_output.get("feature_gaps"), list) else 0,
        "persona_count": len(parsed_output.get("buyer_personas", [])) if isinstance(parsed_output.get("buyer_personas"), list) else 0,
        "brand_count": len(parsed_output.get("brand_vulnerability", [])) if isinstance(parsed_output.get("brand_vulnerability"), list) else 0,
        "insight_count": len(parsed_output.get("insights", [])) if isinstance(parsed_output.get("insights"), list) else 0,
        "recommendation_count": len(parsed_output.get("recommendations", [])) if isinstance(parsed_output.get("recommendations"), list) else 0,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark():
    """Run the benchmark."""
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

    from atlas_brain.skills import get_skill_registry

    skill = get_skill_registry().get("digest/competitive_intelligence")
    if not skill:
        logger.error("Skill digest/competitive_intelligence not found")
        return
    system_prompt = skill.content

    payload = await fetch_benchmark_payload()
    user_content = json.dumps(payload, separators=(",", ":"), default=str)

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

            input_cost = (result["input_tokens"] / 1_000_000) * model["input_cost"]
            output_cost = (result["output_tokens"] / 1_000_000) * model["output_cost"]
            total_cost = input_cost + output_cost
            raw_json_valid = False
            if result["success"] and result["content"]:
                try:
                    json.loads(result["content"])
                    raw_json_valid = True
                except (json.JSONDecodeError, TypeError):
                    raw_json_valid = False

            parsed_output = parse_pipeline_response(result["content"]) if result["success"] else {}
            parse_fallback = bool(isinstance(parsed_output, dict) and parsed_output.get("_parse_fallback"))
            pipeline_score = score_pipeline_output(parsed_output) if isinstance(parsed_output, dict) else {}

            entry = {
                **model,
                **result,
                "raw_json_valid": raw_json_valid,
                "parsed_output": parsed_output,
                "parse_fallback": parse_fallback,
                "pipeline_score": pipeline_score,
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_cost_usd": round(total_cost, 6),
            }
            results.append(entry)

            resp_file = responses_dir / f"{model['label'].replace(' ', '_').lower()}.txt"
            resp_file.write_text(result["content"] or result.get("error") or "empty")

            if result["success"]:
                logger.info(
                    "  OK | %d in / %d out tokens | $%.4f | %dms | raw JSON: %s | parsed fallback: %s | schema typed: %d/%d",
                    result["input_tokens"],
                    result["output_tokens"],
                    total_cost,
                    result["latency_ms"],
                    "valid" if raw_json_valid else "INVALID",
                    "yes" if parse_fallback else "no",
                    pipeline_score.get("required_fields_typed", 0),
                    pipeline_score.get("required_fields_expected", 0),
                )
            else:
                logger.error("  FAILED: %s (%dms)", result["error"], result["latency_ms"])

    results_file.write_text(json.dumps(results, indent=2, default=str))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info("")

    ranked = sorted(
        results,
        key=lambda r: (
            r.get("raw_json_valid", False),
            not r.get("parse_fallback", True),
            (r.get("pipeline_score") or {}).get("required_fields_typed", 0),
            (r.get("pipeline_score") or {}).get("brand_count", 0),
            -(r.get("total_cost_usd", 0) or 0),
        ),
        reverse=True,
    )

    baseline_typed = None
    for result in ranked:
        if result["id"] == "openai/o4-mini":
            baseline_typed = (result.get("pipeline_score") or {}).get("required_fields_typed", 0)
            break

    header = f"{'Model':<25} {'Tier':<10} {'Typed':>7} {'vs Base':>8} {'RawJSON':>8} {'Fallback':>9} {'Cost':>8} {'Latency':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for result in ranked:
        typed = (result.get("pipeline_score") or {}).get("required_fields_typed", 0)
        vs_base = f"{typed / baseline_typed * 100:.0f}%" if baseline_typed else "N/A"
        logger.info(
            "%-25s %-10s %6d/%-1d %7s %7s %8s $%6.4f %6dms",
            result["label"],
            result["tier"],
            typed,
            (result.get("pipeline_score") or {}).get("required_fields_expected", 0),
            vs_base,
            "yes" if result.get("raw_json_valid") else "no",
            "yes" if result.get("parse_fallback") else "no",
            result.get("total_cost_usd", 0),
            result.get("latency_ms", 0),
        )

    logger.info("")
    logger.info("Results saved to: %s", results_file)

    logger.info("")
    logger.info("PIPELINE PARSE SUMMARY:")
    for result in ranked:
        score = result.get("pipeline_score") or {}
        logger.info(
            "  %-25s raw_json=%-7s parse_fallback=%-5s typed=%d/%d brands=%d flows=%d insights=%d recs=%d",
            result["label"],
            "yes" if result.get("raw_json_valid") else "no",
            "yes" if result.get("parse_fallback") else "no",
            score.get("required_fields_typed", 0),
            score.get("required_fields_expected", 0),
            score.get("brand_count", 0),
            score.get("flow_count", 0),
            score.get("insight_count", 0),
            score.get("recommendation_count", 0),
        )

    # Cost efficiency ranking
    logger.info("")
    logger.info("COST EFFICIENCY (typed required fields per dollar):")
    for result in ranked:
        cost = result.get("total_cost_usd", 0)
        if cost > 0:
            efficiency = ((result.get("pipeline_score") or {}).get("required_fields_typed", 0)) / cost
            logger.info("  %-25s %.1f typed-fields/$", result["label"], efficiency)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark competitive-intelligence reasoning models against the production path"
    )
    parser.parse_args()
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
