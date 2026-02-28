#!/usr/bin/env python3
"""
Benchmark deep enrichment extraction throughput and schema quality.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from atlas_brain.config import settings
from atlas_brain.pipelines.llm import clean_llm_output
from atlas_brain.services import llm_registry
from atlas_brain.services.protocols import Message
from atlas_brain.skills import get_skill_registry
from atlas_brain.storage.database import get_db_pool


REQUIRED_KEYS = {
    "sentiment_aspects": list,
    "feature_requests": list,
    "failure_details": (dict, type(None)),
    "product_comparisons": list,
    "product_name_mentioned": str,
    "buyer_context": dict,
    "quotable_phrases": list,
    "would_repurchase": (bool, type(None)),
    "external_references": list,
    "positive_aspects": list,
    "expertise_level": str,
    "frustration_threshold": str,
    "discovery_channel": str,
    "consideration_set": list,
    "buyer_household": str,
    "profession_hint": (str, type(None)),
    "budget_type": str,
    "use_intensity": str,
    "research_depth": str,
    "community_mentions": list,
    "consequence_severity": str,
    "replacement_behavior": str,
    "brand_loyalty_depth": str,
    "ecosystem_lock_in": dict,
    "safety_flag": dict,
    "bulk_purchase_signal": dict,
    "review_delay_signal": str,
    "sentiment_trajectory": str,
    "occasion_context": str,
    "switching_barrier": dict,
    "amplification_intent": dict,
    "review_sentiment_openness": dict,
}

VALID_EXPERTISE = {"novice", "intermediate", "expert", "professional"}
VALID_FRUSTRATION = {"low", "medium", "high"}
VALID_DISCOVERY = {"amazon_organic", "youtube", "reddit", "friend", "amazon_choice", "unknown"}
VALID_HOUSEHOLD = {"single", "family", "professional", "gift", "bulk"}
VALID_BUDGET = {"budget_constrained", "value_seeker", "premium_willing", "unknown"}
VALID_INTENSITY = {"light", "moderate", "heavy"}
VALID_RESEARCH = {"impulse", "light", "moderate", "deep"}
VALID_CONSEQUENCE = {"inconvenience", "workflow_impact", "financial_loss", "safety_concern"}
VALID_REPLACEMENT = {"returned", "replaced_same", "switched_brand", "kept_broken", "unknown"}
VALID_LOYALTY = {"first_time", "occasional", "loyal", "long_term_loyal"}
VALID_DELAY = {"immediate", "days", "weeks", "months", "unknown"}
VALID_TRAJECTORY = {"always_bad", "degraded", "mixed_then_bad", "initially_positive", "unknown"}
VALID_OCCASION = {"none", "gift", "replacement", "upgrade", "first_in_category", "seasonal"}

# Section A sub-object enums
VALID_SENTIMENT = {"positive", "negative", "mixed"}
VALID_DIRECTION = {"switched_to", "switched_from", "considered", "compared"}
VALID_PRICE_SENTIMENT = {"expensive", "fair", "cheap", "not_mentioned"}

# Section C sub-object enums
VALID_ECOSYSTEM_LEVEL = {"free", "partially", "fully"}
VALID_BARRIER_LEVEL = {"none", "low", "medium", "high"}
VALID_AMPLIFICATION = {"quiet", "private", "social"}
VALID_BULK_TYPE = {"single", "multi"}


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)
    load_dotenv(ROOT / ".env.local", override=True)


def parse_models(raw: str) -> list[str]:
    items = [m.strip() for m in raw.split(",") if m.strip()]
    if not items:
        raise ValueError("No models provided")
    return items


def resolve_vllm_url() -> str:
    explicit = os.getenv("VLLM_BASE_URL")
    if explicit:
        return explicit
    host = os.getenv("VLLM_HOST", "localhost")
    port = os.getenv("VLLM_PORT", "8080")
    return f"http://{host}:{port}"


def resolve_path(value: str | None, env_key: str) -> str:
    resolved = value or os.getenv(env_key)
    if not resolved:
        raise ValueError(f"Missing required value: {env_key}")
    return resolved


def validate_required_keys(data: dict) -> bool:
    for key, expected in REQUIRED_KEYS.items():
        if key not in data:
            return False
        val = data[key]
        if isinstance(expected, tuple):
            if not isinstance(val, expected):
                return False
        else:
            if not isinstance(val, expected):
                return False
    return True


def validate_section_a_objects(data: dict) -> bool:
    bc = data.get("buyer_context")
    if not isinstance(bc, dict):
        return False
    for field in ("use_case", "buyer_type", "price_sentiment"):
        if field not in bc:
            return False
    ps = bc.get("price_sentiment")
    if ps is not None and ps not in VALID_PRICE_SENTIMENT:
        return False
    aspects = data.get("sentiment_aspects")
    if isinstance(aspects, list):
        for a in aspects:
            if isinstance(a, dict) and a.get("sentiment") not in VALID_SENTIMENT:
                return False
    comparisons = data.get("product_comparisons")
    if isinstance(comparisons, list):
        for c in comparisons:
            if isinstance(c, dict) and c.get("direction") not in VALID_DIRECTION:
                return False
    return True


def validate_section_b(data: dict) -> bool:
    return (
        data.get("expertise_level") in VALID_EXPERTISE
        and data.get("frustration_threshold") in VALID_FRUSTRATION
        and data.get("discovery_channel") in VALID_DISCOVERY
        and data.get("buyer_household") in VALID_HOUSEHOLD
        and data.get("budget_type") in VALID_BUDGET
        and data.get("use_intensity") in VALID_INTENSITY
        and data.get("research_depth") in VALID_RESEARCH
        and data.get("consequence_severity") in VALID_CONSEQUENCE
        and data.get("replacement_behavior") in VALID_REPLACEMENT
    )


def validate_section_c_enums(data: dict) -> bool:
    return (
        data.get("brand_loyalty_depth") in VALID_LOYALTY
        and data.get("review_delay_signal") in VALID_DELAY
        and data.get("sentiment_trajectory") in VALID_TRAJECTORY
        and data.get("occasion_context") in VALID_OCCASION
    )


def validate_section_c_objects(data: dict) -> bool:
    eco = data.get("ecosystem_lock_in", {})
    if not isinstance(eco, dict) or "level" not in eco or "ecosystem" not in eco:
        return False
    if eco["level"] not in VALID_ECOSYSTEM_LEVEL:
        return False
    safety = data.get("safety_flag", {})
    if not isinstance(safety, dict) or "flagged" not in safety or "description" not in safety:
        return False
    bulk = data.get("bulk_purchase_signal", {})
    if not isinstance(bulk, dict) or "type" not in bulk or "estimated_qty" not in bulk:
        return False
    if bulk["type"] not in VALID_BULK_TYPE:
        return False
    barrier = data.get("switching_barrier", {})
    if not isinstance(barrier, dict) or "level" not in barrier or "reason" not in barrier:
        return False
    if barrier["level"] not in VALID_BARRIER_LEVEL:
        return False
    amp = data.get("amplification_intent", {})
    if not isinstance(amp, dict) or "intent" not in amp or "context" not in amp:
        return False
    if amp["intent"] not in VALID_AMPLIFICATION:
        return False
    openness = data.get("review_sentiment_openness", {})
    if not isinstance(openness, dict) or "open" not in openness or "condition" not in openness:
        return False
    return True


def validate_extraction(data: dict) -> bool:
    return (
        validate_required_keys(data)
        and validate_section_a_objects(data)
        and validate_section_b(data)
        and validate_section_c_enums(data)
        and validate_section_c_objects(data)
    )


def build_payload(row: dict[str, Any]) -> str:
    payload = {
        "review_text": (row.get("review_text") or "")[:3000],
        "summary": row.get("summary") or "",
        "rating": float(row.get("rating") or 0),
        "product_name": row.get("product_title") or row.get("asin"),
        "brand": row.get("product_brand") or "",
        "root_cause": row.get("root_cause") or "",
        "severity": row.get("severity") or "",
        "pain_score": float(row.get("pain_score") or 0.0),
    }
    return json.dumps(payload)


def call_llm(llm, skill_content: str, row: dict[str, Any], max_tokens: int) -> str:
    payload = build_payload(row)
    result = llm.chat(
        messages=[
            Message(role="system", content=skill_content),
            Message(role="user", content=payload),
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return clean_llm_output(result.get("response", ""))


def parse_extraction(text: str) -> tuple[dict | None, str]:
    if not text:
        return None, "empty"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, "json_error"
    if not isinstance(parsed, dict):
        return None, "not_dict"
    return parsed, "ok"


async def process_row(llm, skill_content: str, row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    start = time.perf_counter()
    error = "ok"
    try:
        text = await asyncio.to_thread(call_llm, llm, skill_content, row, max_tokens)
        parsed, error = parse_extraction(text)
        schema_ok = bool(parsed) and validate_extraction(parsed)
        if parsed and not schema_ok:
            error = "schema_invalid"
    except Exception:
        parsed = None
        schema_ok = False
        error = "exception"
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "id": row.get("id"),
        "json_ok": bool(parsed),
        "schema_ok": schema_ok,
        "error": error,
        "duration_ms": duration_ms,
    }


async def run_workers(rows: list[dict[str, Any]], workers: int, worker_fn) -> list[dict[str, Any]]:
    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            item = await worker_fn(row)
            async with lock:
                results.append(item)
            queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    await asyncio.gather(*tasks)
    return results


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[idx]


def summarize(results: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    total = len(results)
    json_ok = sum(1 for r in results if r["json_ok"])
    schema_ok = sum(1 for r in results if r["schema_ok"])
    durations = [r["duration_ms"] for r in results]
    rate = (total / elapsed_s * 3600) if elapsed_s > 0 else 0
    return {
        "total": total,
        "json_ok": json_ok,
        "schema_ok": schema_ok,
        "json_ok_rate": round(json_ok / total, 4) if total else 0,
        "schema_ok_rate": round(schema_ok / total, 4) if total else 0,
        "avg_ms": round(sum(durations) / total, 2) if total else 0,
        "p50_ms": round(percentile(durations, 50), 2),
        "p95_ms": round(percentile(durations, 95), 2),
        "reviews_per_hour": round(rate, 2),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tier_filter(tier: str) -> str:
    if tier == "all":
        return ""
    tiers = {"1": (50, None), "2": (20, 49), "3": (10, 19)}
    min_rev, max_rev = tiers[tier]
    having = f"HAVING count(*) >= {min_rev}"
    if max_rev is not None:
        having += f" AND count(*) <= {max_rev}"
    return (
        "AND pr.asin IN ("
        "SELECT asin FROM product_reviews "
        "WHERE enrichment_status = 'enriched' "
        f"GROUP BY asin {having})"
    )


async def prepare_sample(pool, size: int, tier: str) -> list[int]:
    rows = await pool.fetch(
        f"""
        SELECT pr.id
        FROM product_reviews pr
        WHERE pr.deep_enrichment_status = 'pending'
          AND pr.deep_enrichment_attempts < $1
          {tier_filter(tier)}
        ORDER BY pr.imported_at ASC
        LIMIT $2
        """,
        settings.external_data.deep_enrichment_max_attempts,
        size,
    )
    return [r["id"] for r in rows]


async def fetch_rows(pool, ids: list[int]) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT pr.id, pr.asin, pr.rating, pr.summary, pr.review_text,
               pr.root_cause, pr.severity, pr.pain_score,
               pm.title AS product_title, pm.brand AS product_brand
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.id = ANY($1)
        """,
        ids,
    )
    by_id = {row["id"]: dict(row) for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def compare_baseline(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_rows = {r["id"]: r for r in baseline.get("results", [])}
    cur_rows = {r["id"]: r for r in current.get("results", [])}
    regressions = 0
    for rid, base in base_rows.items():
        if not base.get("schema_ok"):
            continue
        if rid in cur_rows and not cur_rows[rid].get("schema_ok"):
            regressions += 1
    return {
        "baseline_schema_ok_rate": baseline.get("summary", {}).get("schema_ok_rate", 0),
        "current_schema_ok_rate": current.get("summary", {}).get("schema_ok_rate", 0),
        "regressions": regressions,
    }


def print_summary(output: dict[str, Any]) -> None:
    summary = output.get("summary", {})
    line = (
        f"model={output.get('model')} "
        f"schema_ok_rate={summary.get('schema_ok_rate')} "
        f"reviews_per_hour={summary.get('reviews_per_hour')}"
    )
    print(line)
    if "comparison" in output:
        comp = output["comparison"]
        print(
            f"baseline_schema_ok_rate={comp.get('baseline_schema_ok_rate')} "
            f"regressions={comp.get('regressions')}"
        )


async def run_model(pool, args, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    llm_registry.activate(
        args.provider,
        model=model,
        base_url=args.base_url,
        timeout=settings.llm.ollama_timeout,
    )
    llm = llm_registry.get_active()
    if not llm:
        raise RuntimeError("No LLM available")

    skill = get_skill_registry().get("digest/deep_extraction")
    if not skill:
        raise RuntimeError("Skill digest/deep_extraction not found")

    start = time.perf_counter()
    worker_fn = lambda row: process_row(llm, skill.content, row, args.max_tokens)
    results = await run_workers(rows, args.workers, worker_fn)
    elapsed = time.perf_counter() - start
    summary = summarize(results, elapsed)
    return {
        "model": model,
        "provider": args.provider,
        "base_url": args.base_url,
        "sample_file": args.sample_file,
        "sample_size": len(rows),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 2),
        "summary": summary,
        "results": results,
    }


async def run_prepare(pool, args, sample_path: Path) -> None:
    ids = await prepare_sample(pool, args.sample_size, args.tier)
    ensure_parent(sample_path)
    sample = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "ids": ids,
    }
    sample_path.write_text(json.dumps(sample, indent=2))
    print(f"Sample saved: {sample_path} ({len(ids)} reviews)")


async def run_benchmark(pool, args, sample_path: Path) -> None:
    sample_data = json.loads(sample_path.read_text())
    ids = sample_data.get("ids", [])
    if not ids:
        raise ValueError("Sample file has no ids")
    rows = await fetch_rows(pool, ids)
    if len(rows) != len(ids):
        raise RuntimeError("Sample rows could not be loaded from database")

    baseline = None
    if args.baseline_file:
        baseline = json.loads(Path(args.baseline_file).read_text())

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models)
    regressions_found = False

    for model in models:
        output = await run_model(pool, args, model, rows)
        if baseline:
            comparison = compare_baseline(baseline, output)
            output["comparison"] = comparison
            if not args.allow_regressions:
                if comparison["regressions"] > 0:
                    regressions_found = True
                if comparison["current_schema_ok_rate"] < comparison["baseline_schema_ok_rate"]:
                    regressions_found = True
        print_summary(output)
        out_path = results_dir / f"bench_{model.replace('/', '_')}.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote results: {out_path}")

    if regressions_found:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark deep enrichment extraction")
    parser.add_argument("--prepare-sample", action="store_true", help="Create a fixed sample set")
    parser.add_argument("--sample-size", type=int, help="Number of reviews in the sample")
    parser.add_argument("--sample-file", type=str, help="Path to sample file (json)")
    parser.add_argument("--tier", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--models", type=str, help="Comma-separated model list")
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "ollama"])
    parser.add_argument("--base-url", type=str, default=None, help="Override provider base URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent workers")
    parser.add_argument("--baseline-file", type=str, default=None, help="Baseline results file")
    parser.add_argument(
        "--allow-regressions",
        action="store_true",
        help="Do not fail when baseline regressions are detected",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Output dir for results")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> None:
    args.sample_file = resolve_path(args.sample_file, "BENCH_SAMPLE_FILE")
    if args.prepare_sample:
        if not args.sample_size or args.sample_size <= 0:
            raise ValueError("sample-size must be > 0 when preparing a sample")
        return
    if not args.models:
        raise ValueError("models is required for benchmarking")
    if not args.workers or args.workers <= 0:
        raise ValueError("workers must be > 0 for benchmarking")
    if not args.max_tokens:
        args.max_tokens = settings.external_data.deep_enrichment_max_tokens
    if not args.base_url:
        if args.provider == "vllm":
            args.base_url = resolve_vllm_url()
        else:
            args.base_url = settings.llm.ollama_url
    if not args.results_dir:
        args.results_dir = os.getenv("BENCH_RESULTS_DIR", "data/benchmarks/results")


async def main() -> None:
    load_env()
    args = parse_args()
    resolve_args(args)

    pool = get_db_pool()
    await pool.initialize()

    sample_path = Path(args.sample_file)
    if args.prepare_sample:
        await run_prepare(pool, args, sample_path)
        await pool.close()
        return

    await run_benchmark(pool, args, sample_path)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
