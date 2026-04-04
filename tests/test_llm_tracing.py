from atlas_brain.config import ModelPricingConfig
from atlas_brain.pipelines.llm import trace_llm_call
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


async def test_store_local_promotes_business_attribution_fields(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", lambda: pool)
    tracer = FTLTracingClient()
    await tracer._store_local(
        {
            "span_name": "reasoning.process",
            "operation_type": "reasoning",
            "model_name": "anthropic/claude-haiku-4-5",
            "model_provider": "openrouter",
            "input_tokens": 400,
            "output_tokens": 200,
            "total_tokens": 600,
            "cost_usd": 0.0012,
            "duration_ms": 420,
            "status": "completed",
            "metadata": {
                "business": {
                    "vendor_name": "Slack",
                    "source_name": "crm_provider",
                    "event_type": "crm.interaction_logged",
                    "entity_type": "contact",
                    "entity_id": "contact-123",
                },
                "run_id": "run-abc",
            },
        }
    )
    _, args = pool.calls[0]
    assert args[20] == "Slack"
    assert args[21] == "run-abc"
    assert args[22] == "crm_provider"
    assert args[23] == "crm.interaction_logged"
    assert args[24] == "contact"
    assert args[25] == "contact-123"


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


def test_trace_llm_call_runs_even_when_remote_tracing_disabled(monkeypatch):
    tracer = FTLTracingClient()
    tracer._enabled = False
    original_start_span = tracer.start_span
    original_end_span = tracer.end_span
    started: dict[str, object] = {}
    ended: dict[str, object] = {}

    def _fake_start_span(
        span_name,
        operation_type,
        parent=None,
        model_name=None,
        model_provider=None,
        session_id=None,
        metadata=None,
    ):
        started.update({
            "span_name": span_name,
            "operation_type": operation_type,
            "model_name": model_name,
            "model_provider": model_provider,
            "metadata": metadata,
        })
        return original_start_span(
            span_name=span_name,
            operation_type=operation_type,
            parent=parent,
            model_name=model_name,
            model_provider=model_provider,
            session_id=session_id,
            metadata=metadata,
        )

    def _fake_end_span(ctx, **kwargs):
        ended.update(kwargs)
        return original_end_span(ctx, **kwargs)

    monkeypatch.setattr("atlas_brain.services.tracing.tracer", tracer)
    monkeypatch.setattr(tracer, "start_span", _fake_start_span)
    monkeypatch.setattr(tracer, "end_span", _fake_end_span)

    trace_llm_call(
        "task.b2b_reasoning_synthesis",
        input_tokens=1200,
        output_tokens=300,
        model="anthropic/claude-sonnet-4-5",
        provider="openrouter",
        metadata={"run_id": "run-123", "vendor_name": "Slack"},
        duration_ms=1500,
    )

    assert started["span_name"] == "task.b2b_reasoning_synthesis"
    assert started["operation_type"] == "llm_call"
    assert started["metadata"] == {"run_id": "run-123", "vendor_name": "Slack"}
    assert ended["input_tokens"] == 1200
    assert ended["output_tokens"] == 300
