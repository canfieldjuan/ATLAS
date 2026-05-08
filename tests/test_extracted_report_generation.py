from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.report_generation import (
    ReportGenerationConfig,
    ReportGenerationService,
    parse_report_response,
)
from extracted_content_pipeline.report_ports import ReportDraft


# -----------------------
# Fake ports (mirror tests/test_extracted_campaign_generation.py:24-120)
# -----------------------


class _Intelligence:
    def __init__(self, opportunities):
        self.opportunities = opportunities
        self.calls = []

    async def read_campaign_opportunities(self, *, scope, target_mode, limit, filters=None):
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": filters,
        })
        return self.opportunities

    async def read_vendor_targets(self, *, scope, target_mode, vendor_name=None):  # pragma: no cover
        raise AssertionError("not used")


class _Reports:
    def __init__(self):
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"report-{index + 1}" for index, _ in enumerate(drafts)]

    async def list_drafts(self, *, scope, status=None, target_mode=None, report_type=None, limit=None):  # pragma: no cover
        raise AssertionError("not used")

    async def update_status(self, report_id, status, *, scope):  # pragma: no cover
        raise AssertionError("not used")


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": dict(metadata or {}),
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return LLMResponse(
                content=response["content"],
                model=response.get("model", "test-model"),
                usage=response.get("usage", {}),
            )
        return LLMResponse(
            content=response,
            model="test-model",
            usage={"input_tokens": 11, "output_tokens": 7},
        )


class _Skills:
    def __init__(self, prompts):
        self.prompts = prompts
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return self.prompts.get(name)


class _ReasoningProvider:
    def __init__(self, context):
        self.context = context
        self.calls = []

    async def read_campaign_reasoning_context(
        self,
        *,
        scope,
        target_id,
        target_mode,
        opportunity,
    ):
        self.calls.append({
            "scope": scope,
            "target_id": target_id,
            "target_mode": target_mode,
        })
        return self.context


def _opportunity():
    return {
        "id": "opp-1",
        "target_id": "vendor-acme",
        "vendor_name": "Acme",
        "company_name": "Acme",
        "evidence": [{"quote": "Renewal pricing is too high"}],
    }


def _valid_report_json(*, title="Acme: Q3 Pressure Report", section_id="findings"):
    return json.dumps({
        "title": title,
        "summary": "Pricing renewal pressure dominates the displacement signal across review cohorts.",
        "report_type": "vendor_pressure",
        "sections": [
            {
                "id": section_id,
                "title": "Findings",
                "body_markdown": "Renewal pricing leads displacement.",
                "claim_ids": ["c1"],
                "evidence_ids": ["r1"],
            }
        ],
        "reference_ids": ["r1"],
    })


def _service(*, opportunities=None, responses=None, prompts=None, reasoning_context=None, config=None):
    intelligence = _Intelligence(opportunities or [_opportunity()])
    reports = _Reports()
    llm = _LLM(responses or [_valid_report_json()])
    if prompts is None:
        prompts = {"digest/report_generation": "TEMPLATE for {target_mode}: {opportunity_json}"}
    skills = _Skills(prompts)
    reasoning_provider = (
        _ReasoningProvider(reasoning_context) if reasoning_context is not None else None
    )
    service = ReportGenerationService(
        intelligence=intelligence,
        reports=reports,
        llm=llm,
        skills=skills,
        reasoning_context=reasoning_provider,
        config=config or ReportGenerationConfig(),
    )
    return service, intelligence, reports, llm, skills, reasoning_provider


# -----------------------
# parse_report_response unit
# -----------------------


def test_parse_report_response_strips_code_fences_and_extracts_first_object() -> None:
    text = (
        "```json\n"
        + _valid_report_json()
        + "\n```"
    )
    parsed = parse_report_response(text)
    assert parsed is not None
    assert parsed["title"] == "Acme: Q3 Pressure Report"
    assert parsed["sections"][0]["id"] == "findings"


def test_parse_report_response_returns_none_when_required_fields_missing() -> None:
    assert parse_report_response("") is None
    assert parse_report_response('{"title": "no sections"}') is None
    assert (
        parse_report_response('{"title": "t", "summary": "s", "sections": []}')
        is None
    )


def test_parse_report_response_handles_braces_inside_string_values() -> None:
    """body_markdown with template syntax / set notation must not break the parser."""

    payload = json.dumps({
        "title": "Title",
        "summary": "Summary text long enough to be valid.",
        "report_type": "vendor_pressure",
        "sections": [
            {
                "id": "s1",
                "title": "Templates",
                "body_markdown": "Pricing tiers: {basic, pro} models. Use {project_id} placeholders.",
                "evidence_ids": ["r1"],
            }
        ],
        "reference_ids": ["r1"],
    })
    parsed = parse_report_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Title"
    assert "{basic, pro}" in parsed["sections"][0]["body_markdown"]


def test_parse_report_response_strips_think_blocks_before_decoding() -> None:
    """Reasoner output may include <think>...</think> scratchpad ahead of the JSON."""

    payload = (
        "<think>Considering the opportunity... draft sections for vendor.</think>\n"
        + json.dumps({
            "title": "Title",
            "summary": "Summary text long enough to be valid.",
            "report_type": "vendor_pressure",
            "sections": [
                {
                    "id": "s1",
                    "title": "Body",
                    "body_markdown": "body",
                    "evidence_ids": ["r1"],
                }
            ],
            "reference_ids": ["r1"],
        })
    )
    parsed = parse_report_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Title"


# -----------------------
# Service: generation
# -----------------------


@pytest.mark.asyncio
async def test_generate_persists_one_report_per_opportunity_via_save_drafts() -> None:
    service, _intel, reports, llm, skills, _rp = _service()

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.requested == 1
    assert result.generated == 1
    assert result.skipped == 0
    assert result.saved_ids == ("report-1",)
    assert len(llm.calls) == 1
    assert llm.calls[0]["metadata"]["asset_type"] == "report"
    saved_drafts = reports.saved[0]["drafts"]
    assert len(saved_drafts) == 1
    draft = saved_drafts[0]
    assert isinstance(draft, ReportDraft)
    assert draft.target_id == "vendor-acme"
    assert draft.target_mode == "vendor"
    assert draft.report_type == "vendor_pressure"
    assert draft.title == "Acme: Q3 Pressure Report"
    assert draft.sections[0].id == "findings"
    assert draft.reference_ids == ("r1",)
    assert draft.metadata["generation_model"] == "test-model"
    assert "digest/report_generation" in skills.calls


@pytest.mark.asyncio
async def test_generate_skips_unparseable_response_and_records_error() -> None:
    service, _intel, reports, _llm, _skills, _rp = _service(
        responses=["not json garbage", "still not json"],
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.requested == 1
    assert result.generated == 0
    assert result.skipped == 1
    assert reports.saved == []
    assert any(
        err.get("reason") == "unparseable_response" for err in result.errors
    )


@pytest.mark.asyncio
async def test_generate_retries_unparseable_response_once_by_default() -> None:
    service, _intel, reports, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_report_json()]
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.generated == 1
    assert len(llm.calls) == 2
    assert llm.calls[0]["metadata"]["attempt_no"] == 1
    assert llm.calls[1]["metadata"]["attempt_no"] == 2
    assert "Previous response excerpt:\nnot json garbage" in llm.calls[1]["messages"][1].content
    draft = reports.saved[0]["drafts"][0]
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_can_disable_parse_retry() -> None:
    service, _intel, reports, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_report_json()],
        config=ReportGenerationConfig(parse_retry_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.generated == 0
    assert len(llm.calls) == 1
    assert reports.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_accumulates_usage_across_parse_retry_attempts() -> None:
    service, _intel, reports, _llm, _skills, _rp = _service(
        responses=[
            {
                "content": "not json garbage",
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_report_json(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ]
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.generated == 1
    draft = reports.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_skips_when_quality_pack_blocks() -> None:
    """A response missing references AND missing per-section evidence_ids gets blocked."""

    bad_response = json.dumps({
        "title": "ok",
        "summary": "non-empty summary",
        "report_type": "vendor_pressure",
        "sections": [
            {
                "id": "s1",
                "title": "Title",
                "body_markdown": "Body",
                "claim_ids": [],
                "evidence_ids": [],
            }
        ],
        "reference_ids": [],
    })
    service, _intel, reports, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.generated == 0
    assert result.skipped == 1
    assert reports.saved == []
    assert any(
        err.get("reason") == "quality_blocked" for err in result.errors
    )
    blockers = result.errors[0]["blockers"]
    assert any("no_references" in b for b in blockers)


@pytest.mark.asyncio
async def test_generate_consumes_reasoning_context_via_provider() -> None:
    """When a CampaignReasoningContextProvider is supplied, its context is fetched per opportunity."""

    context = CampaignReasoningContext(
        top_theses=({"claim": "Renewal pricing", "confidence": 0.9, "source_ids": ["r1"]},),
        canonical_reasoning={"summary": "narrative summary"},
    )
    service, _intel, _reports, _llm, _skills, reasoning_provider = _service(
        reasoning_context=context,
    )

    await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert reasoning_provider is not None
    assert len(reasoning_provider.calls) == 1
    assert reasoning_provider.calls[0]["target_id"] == "vendor-acme"


@pytest.mark.asyncio
async def test_generate_skips_opportunity_with_missing_target_id() -> None:
    bad_opportunity = {"id": None, "target_id": None, "vendor_name": ""}
    service, _intel, reports, _llm, _skills, _rp = _service(
        opportunities=[bad_opportunity],
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    assert result.skipped == 1
    assert reports.saved == []
    assert any(err.get("reason") == "missing_target_id" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_raises_value_error_when_skill_prompt_missing() -> None:
    service, _intel, _reports, _llm, _skills, _rp = _service(
        prompts={},  # skill not registered
    )

    with pytest.raises(ValueError, match="Report generation skill not found"):
        await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")


@pytest.mark.asyncio
async def test_generate_substitutes_template_placeholders_in_system_prompt_only() -> None:
    """Opportunity JSON appears in the system prompt (via {opportunity_json}) but NOT duplicated in the user message."""

    service, _intel, _reports, llm, _skills, _rp = _service(
        prompts={"digest/report_generation": "MODE={target_mode} OPP={opportunity_json}"},
    )

    await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    messages = llm.calls[0]["messages"]
    system_msg = messages[0].content
    user_msg = messages[1].content
    # Opportunity JSON must appear in the system prompt (via the substituted placeholder)
    assert '"target_id":"vendor-acme"' in system_msg
    assert "MODE=vendor" in system_msg
    # And NOT appear in the user message (no duplication)
    assert '"target_id":"vendor-acme"' not in user_msg
    assert "vendor-acme" not in user_msg


@pytest.mark.asyncio
async def test_generate_aggregates_section_evidence_ids_into_reference_ids() -> None:
    response = json.dumps({
        "title": "Title",
        "summary": "Summary text long enough to be valid.",
        "report_type": "vendor_pressure",
        "sections": [
            {
                "id": "s1",
                "title": "Section 1",
                "body_markdown": "body",
                "evidence_ids": ["r1", "r2"],
            },
            {
                "id": "s2",
                "title": "Section 2",
                "body_markdown": "body",
                "evidence_ids": ["r2", "r3"],
            },
        ],
        "reference_ids": ["r0"],
    })
    service, _intel, reports, _llm, _skills, _rp = _service(responses=[response])

    await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    draft = reports.saved[0]["drafts"][0]
    # Top-level reference_ids preserved + section evidence ids unioned without dupes.
    assert draft.reference_ids == ("r0", "r1", "r2", "r3")


# -----------------------
# PR-OptionA-1: per-call default_report_type override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_default_report_type_override_when_llm_omits_type():
    """When the LLM doesn't include a `report_type` in its JSON, the per-call
    override (from the plan's step.config) wins over the construction-time
    default. The LLM-supplied value still wins when present (tested below)."""

    response = json.dumps({
        "title": "Acme",
        "summary": "Findings.",
        # report_type intentionally omitted from LLM output
        "sections": [{"id": "s1", "title": "S", "body_markdown": "b", "evidence_ids": ["r1"]}],
        "reference_ids": ["r1"],
    })
    service, _intel, reports, _llm, _skills, _rp = _service(
        responses=[response],
        config=ReportGenerationConfig(default_report_type="vendor_pressure"),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        default_report_type="customer_health",  # plan-supplied override
    )

    saved_drafts = reports.saved[0]["drafts"]
    assert saved_drafts[0].report_type == "customer_health"


@pytest.mark.asyncio
async def test_generate_llm_supplied_report_type_still_wins_over_per_call_override():
    """The per-call override only fills in when the LLM omits `report_type`.
    LLM JSON value still wins so model-driven type selection remains intact."""

    service, _intel, reports, _llm, _skills, _rp = _service(
        responses=[_valid_report_json()],  # already includes report_type=vendor_pressure
        config=ReportGenerationConfig(default_report_type="vendor_pressure"),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        default_report_type="customer_health",  # ignored: LLM supplied vendor_pressure
    )

    saved_drafts = reports.saved[0]["drafts"]
    assert saved_drafts[0].report_type == "vendor_pressure"


@pytest.mark.asyncio
async def test_generate_per_call_default_report_type_falls_back_when_none():
    """A None override leaves the construction-time default in place."""

    response = json.dumps({
        "title": "Acme",
        "summary": "Findings.",
        "sections": [{"id": "s1", "title": "S", "body_markdown": "b", "evidence_ids": ["r1"]}],
        "reference_ids": ["r1"],
    })
    service, _intel, reports, _llm, _skills, _rp = _service(
        responses=[response],
        config=ReportGenerationConfig(default_report_type="vendor_pressure"),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        default_report_type=None,
    )

    saved_drafts = reports.saved[0]["drafts"]
    assert saved_drafts[0].report_type == "vendor_pressure"


# -----------------------
# PR-OptionA-2: per-call temperature/max_tokens/parse_retry_attempts overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_llm_tuning_overrides_win_over_construction_config():
    """Verify the resolved-value param is what reaches the LLM, not
    self._config -- catches typos in _generate_one that read self._config.X
    instead of the resolved params."""

    service, _intel, _reports, llm, _skills, _rp = _service(
        responses=[
            "not parseable",  # forces a retry
            _valid_report_json(),
        ],
        config=ReportGenerationConfig(
            temperature=0.3,
            max_tokens=4096,
            parse_retry_attempts=0,  # would mean 1 LLM call without override
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        temperature=0.95,
        max_tokens=2048,
        parse_retry_attempts=1,  # 1 + 1 retry = 2 calls
    )

    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call["temperature"] == 0.95
        assert call["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_llm_tuning_kwargs_none_falls_back_to_construction_config():
    service, _intel, _reports, llm, _skills, _rp = _service(
        responses=[_valid_report_json()],
        config=ReportGenerationConfig(temperature=0.7, max_tokens=999),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        temperature=None,
        max_tokens=None,
        parse_retry_attempts=None,
    )

    assert llm.calls[0]["temperature"] == 0.7
    assert llm.calls[0]["max_tokens"] == 999


# -----------------------
# PR-OptionA-3: per-call parse_retry_response_excerpt_chars override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_parse_retry_response_excerpt_chars_override():
    """The override controls how much of a prior invalid response gets
    embedded in the retry user prompt."""

    long_invalid = "X" * 5000
    service, _intel, _reports, llm, _skills, _rp = _service(
        responses=[long_invalid, _valid_report_json()],
        config=ReportGenerationConfig(
            parse_retry_attempts=1,
            parse_retry_response_excerpt_chars=200,
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        parse_retry_response_excerpt_chars=50,
    )

    retry_user_prompt = llm.calls[1]["messages"][1].content
    assert "XXX" in retry_user_prompt
    excerpt_section = retry_user_prompt.split("excerpt:")[1].lstrip()
    assert len(excerpt_section.rstrip()) <= 50


# -----------------------
# PR-OptionA-5: per-call quality_gates_enabled override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_quality_gates_enabled_false_skips_quality_gate() -> None:
    """quality_gates_enabled=False short-circuits the quality gate. A
    response missing references would normally hit no_references blocker;
    with the gate skipped, it persists."""

    bad_response = json.dumps({
        "title": "Acme",
        "summary": "Findings.",
        "report_type": "vendor_pressure",
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": []}
        ],
        "reference_ids": [],  # would normally hit no_references blocker
    })
    service, _intel, reports, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        quality_gates_enabled=False,
    )

    assert result.generated == 1
    assert len(reports.saved[0]["drafts"]) == 1


@pytest.mark.asyncio
async def test_generate_per_call_quality_gates_enabled_true_still_blocks() -> None:
    bad_response = json.dumps({
        "title": "Acme",
        "summary": "Findings.",
        "report_type": "vendor_pressure",
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": []}
        ],
        "reference_ids": [],
    })
    service, _intel, reports, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        quality_gates_enabled=True,
    )

    assert result.generated == 0
    assert reports.saved == []
