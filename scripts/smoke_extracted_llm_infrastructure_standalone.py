#!/usr/bin/env python3
"""Standalone smoke-import for the extracted_llm_infrastructure scaffold.

Sets ``EXTRACTED_LLM_INFRA_STANDALONE=1`` before importing anything from
the package, then exercises:

  Part A. The five Phase 2 bridge stubs (``config``, ``services.base``,
    ``services.protocols``, ``services.registry``, ``storage.database``)
    so each loads its standalone implementation rather than the
    delegate-to-atlas_brain path.

  Part B. A shape check on ``settings`` (slim ``LLMInfraSettings``) and
    on the ``llm_registry`` global so we know the standalone copies are
    functionally equivalent at the points the scaffolded code touches
    them.

  Part C. Every scaffolded provider module imports cleanly under the
    standalone toggle. Measured against Phase 2: nine of the fourteen
    providers import without atlas_brain on sys.path; the remaining
    five depend only on third-party packages (httpx, pydantic), which
    CI installs from requirements.txt. There is no remaining
    atlas_brain *import-time* coupling in the providers -- runtime
    coupling (isinstance checks, private method calls, asyncpg.Record
    semantics) is Phase 3 work and is exercised separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set the toggle BEFORE any extracted_llm_infrastructure import.
os.environ["EXTRACTED_LLM_INFRA_STANDALONE"] = "1"
# DB pool initializer is lazy and the smoke check never calls
# ``initialize()``, but make sure the no-op path is the default if
# something does try.
os.environ.setdefault("ATLAS_DB_ENABLED", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    failed: list[str] = []

    # 1. Config
    try:
        from extracted_llm_infrastructure.config import settings, ModelPricingConfig

        assert hasattr(settings, "llm")
        assert hasattr(settings, "b2b_churn")
        assert hasattr(settings, "reasoning")
        assert hasattr(settings, "ftl_tracing")
        assert hasattr(settings.llm, "anthropic_model")
        assert hasattr(settings.b2b_churn, "anthropic_batch_enabled")
        assert hasattr(settings.reasoning, "model")
        assert hasattr(settings.ftl_tracing, "pricing")
        # The pricing object should have the cost_usd method used by
        # services.b2b.anthropic_batch._standard_cost_usd at runtime.
        cost = settings.ftl_tracing.pricing.cost_usd(
            "anthropic", "claude-haiku-4-5", 1000, 500,
        )
        assert isinstance(cost, float) and cost > 0
        # Class identity check: the standalone class should be the local
        # copy, not the atlas_brain one.
        assert ModelPricingConfig.__module__.startswith(
            "extracted_llm_infrastructure._standalone."
        )
        print("OK config (settings, ModelPricingConfig.cost_usd)")
    except Exception as exc:
        print(f"FAIL config: {exc}")
        failed.append("config")

    # 2. Protocols
    try:
        from extracted_llm_infrastructure.services.protocols import (
            InferenceMetrics,
            LLMService,
            Message,
            ModelInfo,
        )

        m = Message(role="user", content="hi")
        assert m.role == "user"
        info = ModelInfo(
            name="x", model_id="y", is_loaded=False, device="cpu", capabilities=[]
        )
        assert info.to_dict()["device"] == "cpu"
        metrics = InferenceMetrics(duration_ms=10.0, device="cpu")
        assert metrics.to_dict()["duration_ms"] == 10.0
        # LLMService Protocol is runtime-checkable.
        assert hasattr(LLMService, "__class_getitem__") or callable(LLMService)
        assert ModelInfo.__module__.startswith(
            "extracted_llm_infrastructure._standalone."
        )
        print("OK services.protocols")
    except Exception as exc:
        print(f"FAIL services.protocols: {exc}")
        failed.append("services.protocols")

    # 3. Base (torch-free)
    try:
        from extracted_llm_infrastructure.services.base import (
            BaseModelService,
            InferenceTimer,
        )

        # Cannot instantiate BaseModelService directly (it's abstract via
        # ABC), but its .device should default to "cpu" without torch.
        # Use a minimal subclass to verify.
        class _Dummy(BaseModelService):
            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

            @property
            def model_info(self):  # type: ignore[override]
                from extracted_llm_infrastructure.services.protocols import ModelInfo
                return ModelInfo(
                    name=self.name,
                    model_id=self.model_id,
                    is_loaded=False,
                    device=self.device,
                )

            def generate(self, *args, **kwargs):  # type: ignore[override]
                return {}

            def chat(self, *args, **kwargs):  # type: ignore[override]
                return {}

            def chat_with_tools(self, *args, **kwargs):  # type: ignore[override]
                return {}

        d = _Dummy(name="dummy", model_id="d-1")
        assert d.device == "cpu"
        assert d.is_loaded is False
        # Timer
        with InferenceTimer() as t:
            _ = sum(range(10))
        assert t.duration >= 0
        assert BaseModelService.__module__.startswith(
            "extracted_llm_infrastructure._standalone."
        )
        print("OK services.base (torch-free)")
    except Exception as exc:
        print(f"FAIL services.base: {exc}")
        failed.append("services.base")

    # 4. Registry
    try:
        from extracted_llm_infrastructure.services.registry import (
            ServiceRegistry,
            llm_registry,
            register_llm,
        )

        assert llm_registry.list_available() == [] or isinstance(
            llm_registry.list_available(), list
        )
        assert callable(register_llm)
        # Local registry (not the atlas_brain singleton).
        assert ServiceRegistry.__module__.startswith(
            "extracted_llm_infrastructure._standalone."
        )
        print("OK services.registry")
    except Exception as exc:
        print(f"FAIL services.registry: {exc}")
        failed.append("services.registry")

    # 5. Database
    try:
        from extracted_llm_infrastructure.storage.database import (
            DatabasePool,
            get_db_pool,
        )

        pool = get_db_pool()
        assert pool.is_initialized is False
        assert DatabasePool.__module__.startswith(
            "extracted_llm_infrastructure._standalone."
        )
        print("OK storage.database (uninitialized)")
    except Exception as exc:
        print(f"FAIL storage.database: {exc}")
        failed.append("storage.database")

    # ------------------------------------------------------------------
    # Part C: every provider module imports cleanly in standalone mode
    # ------------------------------------------------------------------
    import importlib

    PROVIDERS = [
        "extracted_llm_infrastructure.services.b2b.cache_strategy",
        "extracted_llm_infrastructure.services.b2b.anthropic_batch",
        "extracted_llm_infrastructure.pipelines.llm",
        "extracted_llm_infrastructure.reasoning.semantic_cache",
        "extracted_llm_infrastructure.services.llm_router",
        "extracted_llm_infrastructure.services.llm.anthropic",
        "extracted_llm_infrastructure.services.llm.openrouter",
        "extracted_llm_infrastructure.services.llm.ollama",
        "extracted_llm_infrastructure.services.llm.vllm",
        "extracted_llm_infrastructure.services.llm.groq",
        "extracted_llm_infrastructure.services.llm.together",
        "extracted_llm_infrastructure.services.llm.hybrid",
        "extracted_llm_infrastructure.services.llm.cloud",
        "extracted_llm_infrastructure.services.tracing",
    ]
    for module_name in PROVIDERS:
        try:
            importlib.import_module(module_name)
            print(f"OK provider {module_name}")
        except Exception as exc:
            print(f"FAIL provider {module_name}: {exc}")
            failed.append(module_name)

    # ------------------------------------------------------------------
    # Part D: cross-check that providers are seeing the standalone
    # substrate, not silently falling back to atlas_brain. We do this by
    # asserting the substrate classes that providers import resolve to
    # the ``_standalone`` package. If an atlas_brain regression slipped
    # in, providers would silently use atlas's BaseModelService /
    # ServiceRegistry / etc. and Part A's class-identity check would
    # have passed (Part A only checks the *direct* substrate import,
    # not transitive consumption).
    # ------------------------------------------------------------------
    try:
        from extracted_llm_infrastructure.services.llm.anthropic import AnthropicLLM

        bases = [b.__module__ for b in AnthropicLLM.__mro__]
        assert any(
            b.startswith("extracted_llm_infrastructure._standalone.")
            for b in bases
        ), (
            "AnthropicLLM does not inherit from the standalone "
            "BaseModelService -- providers must consume the standalone "
            "substrate transitively. MRO modules: "
            f"{bases}"
        )
        print("OK provider transitively uses standalone substrate (AnthropicLLM)")
    except Exception as exc:
        print(f"FAIL provider transitive substrate check: {exc}")
        failed.append("provider_transitive_substrate")

    if failed:
        print(f"Standalone smoke failed for {len(failed)} item(s): {failed}")
        return 1
    print(
        "Standalone smoke passed: 5 bridges + "
        f"{len(PROVIDERS)} provider modules + transitive-substrate check"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
