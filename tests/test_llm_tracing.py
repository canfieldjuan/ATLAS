from atlas_brain.config import ModelPricingConfig
from atlas_brain.services.tracing import FTLTracingClient


class _FakePool:
    def __init__(self):
        self.is_initialized = True
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return "INSERT 0 1"


async def test_store_local_persists_cache_breakdown(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: pool)
    tracer = FTLTracingClient()
    await tracer._store_local(
        {
            "span_name": "pipeline.demo",
            "operation_type": "llm_call",
            "model_name": "anthropic/claude-sonnet-4-6",
            "model_provider": "openrouter",
            "input_tokens": 12000,
            "output_tokens": 900,
            "total_tokens": 12900,
            "cost_usd": 0.0522,
            "duration_ms": 880,
            "ttft_ms": 140,
            "inference_time_ms": 650,
            "queue_time_ms": 20,
            "tokens_per_second": 31.4,
            "billable_input_tokens": 5000,
            "cached_tokens": 6000,
            "cache_write_tokens": 1000,
            "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "provider_request_id": "req_456",
            "status": "completed",
            "metadata": {"workflow": "demo"},
        }
    )
    _, args = pool.calls[0]
    assert args[13] == 5000
    assert args[14] == 6000
    assert args[15] == 1000
    assert args[16] == "https://openrouter.ai/api/v1/chat/completions"
    assert args[17] == "req_456"


def test_model_pricing_accounts_for_cache_reads_and_writes():
    pricing = ModelPricingConfig()
    cost = pricing.cost_usd(
        "openrouter",
        "anthropic/claude-sonnet-4-6",
        12000,
        900,
        cached_tokens=6000,
        cache_write_tokens=1000,
        billable_input_tokens=5000,
    )
    expected = (
        5000 * pricing.anthropic_sonnet_input
        + 6000 * pricing.anthropic_sonnet_cache_read_input
        + 1000 * pricing.anthropic_sonnet_cache_write_input
        + 900 * pricing.anthropic_sonnet_output
    ) / 1_000_000
    assert cost == expected
