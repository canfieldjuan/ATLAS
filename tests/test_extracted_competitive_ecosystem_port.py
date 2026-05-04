import pytest

from extracted_competitive_intelligence.reasoning.ecosystem import (
    EcosystemAnalyzer,
    EcosystemAnalyzerPortNotConfigured,
    configure_ecosystem_analyzer_factory,
)


class _FakeHostAnalyzer:
    def __init__(self, payload):
        self._payload = payload

    async def analyze_all_categories(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_ecosystem_factory():
    configure_ecosystem_analyzer_factory(None)
    yield
    configure_ecosystem_analyzer_factory(None)


@pytest.mark.asyncio
async def test_ecosystem_analyzer_fails_closed_without_host_adapter():
    analyzer = EcosystemAnalyzer(pool=object())

    with pytest.raises(EcosystemAnalyzerPortNotConfigured):
        await analyzer.analyze_all_categories()


@pytest.mark.asyncio
async def test_ecosystem_analyzer_delegates_to_configured_host_adapter():
    payload = {"CRM": {"health": {"market_structure": "fragmenting"}}}
    seen = {}

    def factory(pool):
        seen["pool"] = pool
        return _FakeHostAnalyzer(payload)

    pool = object()
    configure_ecosystem_analyzer_factory(factory)

    result = await EcosystemAnalyzer(pool).analyze_all_categories()

    assert result == payload
    assert seen["pool"] is pool
