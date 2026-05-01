#!/usr/bin/env python3
"""Standalone smoke-import for the extracted_llm_infrastructure scaffold.

Sets ``EXTRACTED_LLM_INFRA_STANDALONE=1`` before importing anything from
the package, then exercises:

  - The five Phase 2 bridge stubs (``config``, ``services.base``,
    ``services.protocols``, ``services.registry``, ``storage.database``)
    so each loads its standalone implementation rather than the
    delegate-to-atlas_brain path.
  - A representative shape check on ``settings`` (slim
    ``LLMInfraSettings``) and on the ``llm_registry`` global so we know
    the standalone copies are functionally equivalent at the points the
    scaffolded code touches them.

Phase 2 deliberately keeps the scaffolded provider modules (e.g.
``services.llm.anthropic``) on the delegate-to-atlas path -- those still
need their relative imports to resolve through atlas_brain. The standalone
toggle here verifies the *substrate* (settings, base, protocols, registry,
db pool), which Phase 3 work uses to fully decouple the providers.
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

    if failed:
        print(f"Standalone smoke failed for {len(failed)} bridge(s): {failed}")
        return 1
    print("Standalone smoke passed for all 5 Phase 2 bridges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
