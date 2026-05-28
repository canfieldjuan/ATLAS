from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from atlas_brain import main


def test_asr_autostart_allows_cpu_device():
    assert main._asr_autostart_blocked_reason("cpu") is None


def test_asr_autostart_blocks_cuda_when_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reason = main._asr_autostart_blocked_reason("cuda")

    assert reason is not None
    assert "requires CUDA" in reason
    assert "CUDA is not available" in reason


def test_start_asr_server_skips_popen_when_cuda_unavailable(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            raise RuntimeError("ASR endpoint is down")

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(main.settings.voice, "asr_url", "http://localhost:8081")
    monkeypatch.setattr(main.settings.voice, "asr_device", "cuda")
    monkeypatch.setattr(
        main.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("ASR subprocess should not start"),
    )

    assert asyncio.run(main._start_asr_server()) is None


def test_start_asr_server_keeps_running_external_asr_before_cuda_guard(monkeypatch):
    class FakeResponse:
        status_code = 200

    class FakeAsyncClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def get(self, _url):
            return FakeResponse()

    fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setattr(main.settings.voice, "asr_url", "http://localhost:8081")
    monkeypatch.setattr(main.settings.voice, "asr_device", "cuda")
    monkeypatch.setattr(
        main,
        "_asr_autostart_blocked_reason",
        lambda _device: pytest.fail("CUDA guard should not run for healthy external ASR"),
    )
    monkeypatch.setattr(
        main.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("ASR subprocess should not start"),
    )

    assert asyncio.run(main._start_asr_server()) is None
