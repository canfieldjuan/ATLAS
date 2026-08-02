import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    return env


def test_cloud_llm_provider_and_base_metrics_work_without_torch_installed():
    probe = """
import importlib.abc
import json
import sys


class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("blocked torch for regression probe")
        return None


sys.meta_path.insert(0, _BlockTorch())

from atlas_brain.services.llm.cloud import CloudLLM

service = CloudLLM(groq_api_key="", together_api_key="")
metrics = service.gather_metrics(0.01234).to_dict()
print(json.dumps({
    "cloud_device": service.model_info.device,
    "base_device": service.device,
    "metrics": metrics,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        env=_repo_subprocess_env(),
        text=True,
    )

    observed = json.loads(result.stdout.strip().splitlines()[-1])

    assert observed == {
        "cloud_device": "cloud",
        "base_device": "cpu",
        "metrics": {
            "duration_ms": 12.34,
            "device": "cpu",
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "memory_total_mb": 0.0,
        },
        "torch_loaded": False,
    }
