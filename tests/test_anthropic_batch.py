import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from atlas_brain.services.b2b.anthropic_batch import (
    AnthropicBatchItem,
    reconcile_anthropic_message_batch,
    run_anthropic_message_batch,
    submit_anthropic_message_batch,
)
from atlas_brain.services.llm.anthropic import AnthropicLLM
from atlas_brain.services.protocols import Message


class _FakeBatchResults:
    def __init__(self, rows):
        self._rows = list(rows)

    def __aiter__(self):
        self._iter = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeBatchesAPI:
    def __init__(self, rows, *, retrieve_status="ended"):
        self._rows = rows
        self._retrieve_status = retrieve_status

    async def create(self, requests, timeout=None):
        return SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    async def retrieve(self, provider_batch_id, timeout=None):
        return SimpleNamespace(id=provider_batch_id, processing_status=self._retrieve_status)

    async def results(self, provider_batch_id, timeout=None):
        return _FakeBatchResults(self._rows)


class _FakeAsyncClient:
    def __init__(self, rows, *, retrieve_status="ended"):
        self.messages = SimpleNamespace(batches=_FakeBatchesAPI(rows, retrieve_status=retrieve_status))


def _fake_llm(rows, *, retrieve_status="ended"):
    llm = AnthropicLLM(model="claude-sonnet-test", api_key="test-key")
    llm._async_client = _FakeAsyncClient(rows, retrieve_status=retrieve_status)
    llm._loaded = True
    return llm


class _MemoryPool:
    def __init__(self):
        self.is_initialized = True
        self.jobs = {}
        self.items = {}

    async def execute(self, query, *args):
        if "INSERT INTO anthropic_message_batches" in query:
            batch_id = str(args[0])
            self.jobs[batch_id] = {
                "id": batch_id,
                "stage_id": args[1],
                "task_name": args[2],
                "run_id": args[3],
                "status": args[4],
                "total_items": args[5],
                "cache_prefiltered_items": args[6],
                "metadata": json.loads(args[7]) if isinstance(args[7], str) else {},
                "provider_batch_id": None,
                "submitted_items": 0,
                "fallback_single_call_items": 0,
                "completed_items": 0,
                "failed_items": 0,
                "estimated_sequential_cost_usd": 0.0,
                "estimated_batch_cost_usd": 0.0,
                "provider_error": None,
            }
            return
        if "INSERT INTO anthropic_message_batch_items" in query:
            batch_id = str(args[0])
            custom_id = str(args[1])
            key = (batch_id, custom_id)
            created_at = self.items.get(key, {}).get("created_at") or datetime.now(timezone.utc)
            status = str(args[6])
            self.items[key] = {
                "id": self.items.get(key, {}).get("id") or str(uuid4()),
                "batch_id": batch_id,
                "custom_id": custom_id,
                "stage_id": args[2],
                "artifact_type": args[3],
                "artifact_id": args[4],
                "vendor_name": args[5],
                "status": status,
                "cache_prefiltered": bool(args[7]),
                "fallback_single_call": bool(args[8]),
                "response_text": args[9],
                "input_tokens": int(args[10] or 0),
                "billable_input_tokens": int(args[11] or 0),
                "cached_tokens": int(args[12] or 0),
                "cache_write_tokens": int(args[13] or 0),
                "output_tokens": int(args[14] or 0),
                "cost_usd": float(args[15] or 0.0),
                "provider_request_id": args[16],
                "error_text": args[17],
                "request_metadata": json.loads(args[18]) if isinstance(args[18], str) else {},
                "created_at": created_at,
                "completed_at": datetime.now(timezone.utc)
                if status in {
                    "cache_hit",
                    "batch_succeeded",
                    "fallback_succeeded",
                    "fallback_failed",
                    "batch_errored",
                    "batch_expired",
                    "batch_canceled",
                }
                else None,
            }
            return
        if "UPDATE anthropic_message_batches" in query:
            batch_id = str(args[0])
            job = self.jobs[batch_id]
            job["status"] = args[1]
            if args[2] is not None:
                job["provider_batch_id"] = args[2]
            if args[3] is not None:
                job["submitted_items"] = args[3]
            if args[4] is not None:
                job["fallback_single_call_items"] = args[4]
            if args[5] is not None:
                job["completed_items"] = args[5]
            if args[6] is not None:
                job["failed_items"] = args[6]
            if args[7] is not None:
                job["estimated_sequential_cost_usd"] = float(args[7])
            if args[8] is not None:
                job["estimated_batch_cost_usd"] = float(args[8])
            if args[9] is not None:
                job["provider_error"] = args[9]
            if args[10] is not None:
                job["completed_at"] = args[10]
            return
        if "UPDATE anthropic_message_batch_items" in query and "request_metadata" in query:
            item_id = str(args[0])
            patch = json.loads(args[1]) if isinstance(args[1], str) else {}
            for item in self.items.values():
                if item["id"] == item_id:
                    item["request_metadata"] = {**item["request_metadata"], **patch}
                    return
        raise AssertionError(f"Unexpected execute query: {query}")

    async def fetchrow(self, query, *args):
        if "FROM anthropic_message_batches" in query:
            batch_id = str(args[0])
            return self.jobs.get(batch_id)
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetch(self, query, *args):
        if "FROM anthropic_message_batch_items" in query:
            batch_id = str(args[0])
            return [
                item
                for item in sorted(self.items.values(), key=lambda row: row["created_at"])
                if item["batch_id"] == batch_id
            ]
        raise AssertionError(f"Unexpected fetch query: {query}")


@pytest.mark.asyncio
async def test_run_anthropic_message_batch_returns_prefiltered_only_for_cached_items():
    item = AnthropicBatchItem(
        custom_id="cached-item",
        artifact_type="campaign",
        artifact_id="artifact-1",
        vendor_name="Slack",
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content='{"vendor":"Slack"}'),
        ],
        max_tokens=256,
        temperature=0.1,
        cached_response_text='{"subject":"Cached","body":"Cached body","cta":"Book time"}',
        cached_usage={"input_tokens": 10, "output_tokens": 5},
    )

    execution = await run_anthropic_message_batch(
        llm=_fake_llm([]),
        stage_id="b2b_campaign_generation.content",
        task_name="b2b_campaign_generation",
        items=[item],
        min_batch_size=2,
        pool=None,
    )

    assert execution.status == "prefiltered_only"
    assert execution.provider_batch_id is None
    assert execution.submitted_items == 0
    assert execution.cache_prefiltered_items == 1
    assert execution.results_by_custom_id["cached-item"].cached is True
    assert execution.results_by_custom_id["cached-item"].response_text is not None


@pytest.mark.asyncio
async def test_run_anthropic_message_batch_traces_and_returns_succeeded_items(monkeypatch):
    traced = []

    def _fake_trace_llm_call(*args, **kwargs):
        traced.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.trace_llm_call",
        _fake_trace_llm_call,
    )

    row = SimpleNamespace(
        custom_id="batch-item",
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                id="msg_123",
                content=[SimpleNamespace(type="text", text='{"subject":"Hello"}')],
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=7,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
        ),
    )
    item = AnthropicBatchItem(
        custom_id="batch-item",
        artifact_type="scorecard_narrative",
        artifact_id="slack",
        vendor_name="Slack",
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content='{"vendor":"Slack"}'),
        ],
        max_tokens=256,
        temperature=0.1,
        trace_span_name="b2b.churn_intelligence.scorecard_narrative",
        trace_metadata={"vendor_name": "Slack"},
    )

    execution = await run_anthropic_message_batch(
        llm=_fake_llm([row]),
        stage_id="b2b_churn_reports.scorecard_narrative",
        task_name="b2b_churn_reports",
        items=[item],
        min_batch_size=1,
        pool=None,
    )

    assert execution.status == "ended"
    assert execution.provider_batch_id == "msgbatch_1"
    assert execution.submitted_items == 1
    assert execution.completed_items == 1
    assert execution.failed_items == 0
    assert execution.results_by_custom_id["batch-item"].success is True
    assert execution.results_by_custom_id["batch-item"].response_text == '{"subject":"Hello"}'
    assert len(traced) == 1


@pytest.mark.asyncio
async def test_submit_anthropic_message_batch_persists_pending_and_cache_prefiltered_items():
    pool = _MemoryPool()
    cached = AnthropicBatchItem(
        custom_id="cached-item",
        artifact_type="campaign",
        artifact_id="artifact-cached",
        vendor_name="Slack",
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content='{"vendor":"Slack"}'),
        ],
        max_tokens=256,
        temperature=0.1,
        request_metadata={"replay_handler": "campaign_generation"},
        cached_response_text='{"subject":"Cached"}',
        cached_usage={"input_tokens": 10, "output_tokens": 5},
    )
    pending = AnthropicBatchItem(
        custom_id="pending-item",
        artifact_type="campaign",
        artifact_id="artifact-pending",
        vendor_name="Zoom",
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content='{"vendor":"Zoom"}'),
        ],
        max_tokens=256,
        temperature=0.1,
        request_metadata={"replay_handler": "campaign_generation"},
    )

    execution = await submit_anthropic_message_batch(
        llm=_fake_llm([]),
        stage_id="b2b_campaign_generation.content",
        task_name="b2b_campaign_generation",
        items=[cached, pending],
        min_batch_size=1,
        pool=pool,
    )

    assert execution.provider_batch_id == "msgbatch_1"
    assert execution.submitted_items == 1
    assert execution.cache_prefiltered_items == 1
    assert pool.jobs[execution.local_batch_id]["status"] == "in_progress"
    assert pool.items[(execution.local_batch_id, "cached-item")]["status"] == "cache_hit"
    assert pool.items[(execution.local_batch_id, "pending-item")]["status"] == "pending"
    assert pool.items[(execution.local_batch_id, "pending-item")]["request_metadata"]["replay_handler"] == "campaign_generation"


@pytest.mark.asyncio
async def test_reconcile_anthropic_message_batch_records_success_and_error_for_fallback(monkeypatch):
    traced = []

    def _fake_trace_llm_call(*args, **kwargs):
        traced.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.trace_llm_call",
        _fake_trace_llm_call,
    )

    pool = _MemoryPool()
    items = [
        AnthropicBatchItem(
            custom_id="ok-item",
            artifact_type="campaign",
            artifact_id="artifact-ok",
            vendor_name="Slack",
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content='{"vendor":"Slack"}'),
            ],
            max_tokens=256,
            temperature=0.1,
            trace_span_name="task.b2b_campaign_generation",
            trace_metadata={"vendor_name": "Slack"},
        ),
        AnthropicBatchItem(
            custom_id="bad-item",
            artifact_type="campaign",
            artifact_id="artifact-bad",
            vendor_name="Zoom",
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content='{"vendor":"Zoom"}'),
            ],
            max_tokens=256,
            temperature=0.1,
            trace_span_name="task.b2b_campaign_generation",
            trace_metadata={"vendor_name": "Zoom"},
        ),
    ]

    submit_execution = await submit_anthropic_message_batch(
        llm=_fake_llm([]),
        stage_id="b2b_campaign_generation.content",
        task_name="b2b_campaign_generation",
        items=items,
        min_batch_size=1,
        pool=pool,
    )

    result_rows = [
        SimpleNamespace(
            custom_id="ok-item",
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(
                    id="msg_123",
                    content=[SimpleNamespace(type="text", text='{"subject":"Hello"}')],
                    usage=SimpleNamespace(
                        input_tokens=12,
                        output_tokens=7,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                ),
            ),
        ),
        SimpleNamespace(
            custom_id="bad-item",
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(message="rate_limit"),
            ),
        ),
    ]
    reconcile_execution = await reconcile_anthropic_message_batch(
        llm=_fake_llm(result_rows),
        local_batch_id=submit_execution.local_batch_id,
        pool=pool,
    )

    assert reconcile_execution.status == "ended"
    assert reconcile_execution.completed_items == 1
    assert reconcile_execution.failed_items == 1
    assert reconcile_execution.results_by_custom_id["ok-item"].item_status == "batch_succeeded"
    assert reconcile_execution.results_by_custom_id["ok-item"].response_text == '{"subject":"Hello"}'
    assert reconcile_execution.results_by_custom_id["bad-item"].item_status == "fallback_pending"
    assert reconcile_execution.results_by_custom_id["bad-item"].fallback_required is True
    assert reconcile_execution.results_by_custom_id["bad-item"].error_text == "rate_limit"
    assert pool.items[(submit_execution.local_batch_id, "ok-item")]["status"] == "batch_succeeded"
    assert pool.items[(submit_execution.local_batch_id, "bad-item")]["status"] == "batch_errored"
    assert pool.items[(submit_execution.local_batch_id, "bad-item")]["fallback_single_call"] is True
    assert len(traced) == 1
