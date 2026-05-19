from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_llm_default_resolver_imports_with_saas_auth_enabled() -> None:
    """Bridge-mode live provider resolution must not require standalone mode."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("EXTRACTED_LLM_INFRA_STANDALONE", None)
    env.update({
        "ATLAS_SAAS_ENABLED": "true",
        "ATLAS_SAAS_JWT_SECRET": "test-jwt-secret-not-for-prod",
        "ATLAS_SAAS_API_KEY_PEPPER": "test-api-key-pepper-not-for-prod",
        # Synthetic test KEK, not a live secret.
        "ATLAS_SAAS_BYOK_ENCRYPTION_KEK": (
            "v1:fmrbL_ZK1IndEFcRceIBy63lLHnlk-CHeY6kmnn6QgI="
        ),
        "EXTRACTED_CAMPAIGN_LLM_WORKLOAD": "openrouter",
        "EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL": (
            "anthropic/claude-sonnet-4-5"
        ),
        "OPENROUTER_API_KEY": "test-openrouter-key",
    })

    code = """
from extracted_content_pipeline.campaign_llm_client import create_pipeline_llm_client

client = create_pipeline_llm_client()
llm = client.resolver(
    workload=client.workload,
    prefer_cloud=client.prefer_cloud,
    try_openrouter=client.try_openrouter,
    auto_activate_ollama=client.auto_activate_ollama,
    openrouter_model=client.openrouter_model,
)
assert llm is not None
assert getattr(llm, "name", "") == "openrouter"
assert getattr(llm, "model", "") == "anthropic/claude-sonnet-4-5"
print("resolved", llm.name, llm.model)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "resolved openrouter anthropic/claude-sonnet-4-5" in result.stdout
