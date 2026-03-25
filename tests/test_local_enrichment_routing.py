from unittest.mock import patch

from atlas_brain.autonomous.tasks import complaint_enrichment, deep_enrichment


def test_complaint_get_llm_uses_vllm_first():
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = complaint_enrichment._get_llm(local_only=False)

    assert llm is not None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


def test_complaint_get_llm_keeps_local_only_local():
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return None

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = complaint_enrichment._get_llm(local_only=True)

    assert llm is None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


def test_deep_enrichment_uses_vllm_workload():
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = deep_enrichment._get_deep_enrichment_llm()

    assert llm is not None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]
