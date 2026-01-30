"""
Tests for edge device configuration.

Tests the EdgeConfig and sub-config classes.
"""

import os
import pytest

from atlas_edge.config import (
    EdgeConfig,
    BrainConfig,
    HomeAssistantConfig,
    STTConfig,
    TTSConfig,
    IntentConfig,
    VADConfig,
    WakeWordConfig,
    get_settings,
)


class TestBrainConfig:
    """Tests for BrainConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrainConfig()
        assert config.url == "ws://localhost:8000"
        assert config.enabled is True
        assert config.reconnect_interval == 5
        assert config.health_check_interval == 30
        assert config.escalation_timeout == 5.0

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_BRAIN_URL", "ws://brain.local:9000")
        monkeypatch.setenv("ATLAS_BRAIN_ENABLED", "false")
        config = BrainConfig()
        assert config.url == "ws://brain.local:9000"
        assert config.enabled is False


class TestHomeAssistantConfig:
    """Tests for HomeAssistantConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HomeAssistantConfig()
        assert config.url == "http://homeassistant.local:8123"
        assert config.token is None
        assert config.enabled is True

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_HA_URL", "http://192.168.1.100:8123")
        monkeypatch.setenv("ATLAS_HA_TOKEN", "test_token_123")
        config = HomeAssistantConfig()
        assert config.url == "http://192.168.1.100:8123"
        assert config.token == "test_token_123"


class TestSTTConfig:
    """Tests for STTConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = STTConfig()
        assert config.model_name == "nvidia/parakeet-tdt-0.6b"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.sample_rate == 16000

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_STT_DEVICE", "cpu")
        monkeypatch.setenv("ATLAS_EDGE_STT_SAMPLE_RATE", "8000")
        config = STTConfig()
        assert config.device == "cpu"
        assert config.sample_rate == 8000


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        assert config.voice == "en_US-ryan-medium"
        assert config.speed == 1.0
        assert config.model_path is None

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_TTS_VOICE", "en_GB-alba-medium")
        monkeypatch.setenv("ATLAS_EDGE_TTS_SPEED", "1.2")
        config = TTSConfig()
        assert config.voice == "en_GB-alba-medium"
        assert config.speed == 1.2


class TestIntentConfig:
    """Tests for IntentConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IntentConfig()
        assert "XLMRoberta" in config.model_id or "Alexa" in config.model_id
        assert config.device == "cuda"
        assert config.confidence_threshold == 0.7

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_INTENT_DEVICE", "cpu")
        monkeypatch.setenv("ATLAS_EDGE_INTENT_CONFIDENCE_THRESHOLD", "0.8")
        config = IntentConfig()
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.8


class TestVADConfig:
    """Tests for VADConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VADConfig()
        assert config.aggressiveness == 1
        assert config.silence_duration_ms == 800
        assert config.frame_duration_ms == 30

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_VAD_AGGRESSIVENESS", "3")
        monkeypatch.setenv("ATLAS_EDGE_VAD_SILENCE_DURATION_MS", "500")
        config = VADConfig()
        assert config.aggressiveness == 3
        assert config.silence_duration_ms == 500


class TestWakeWordConfig:
    """Tests for WakeWordConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WakeWordConfig()
        assert config.enabled is True
        assert config.keyword == "hey_atlas"
        assert config.threshold == 0.5

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_WAKEWORD_ENABLED", "false")
        monkeypatch.setenv("ATLAS_EDGE_WAKEWORD_KEYWORD", "hey_jarvis")
        config = WakeWordConfig()
        assert config.enabled is False
        assert config.keyword == "hey_jarvis"


class TestEdgeConfig:
    """Tests for main EdgeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EdgeConfig()
        assert config.location_id == "default"
        assert config.location_name == "Home"

    def test_sub_configs_exist(self):
        """Test sub-configurations are created."""
        config = EdgeConfig()
        assert isinstance(config.brain, BrainConfig)
        assert isinstance(config.homeassistant, HomeAssistantConfig)
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.intent, IntentConfig)
        assert isinstance(config.vad, VADConfig)
        assert isinstance(config.wakeword, WakeWordConfig)

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ATLAS_EDGE_LOCATION_ID", "kitchen")
        monkeypatch.setenv("ATLAS_EDGE_LOCATION_NAME", "Kitchen Node")
        config = EdgeConfig()
        assert config.location_id == "kitchen"
        assert config.location_name == "Kitchen Node"

    def test_nested_env_override(self, monkeypatch):
        """Test nested config environment override."""
        monkeypatch.setenv("ATLAS_BRAIN_URL", "ws://192.168.1.50:8000")
        monkeypatch.setenv("ATLAS_HA_TOKEN", "my_secret_token")
        config = EdgeConfig()
        assert config.brain.url == "ws://192.168.1.50:8000"
        assert config.homeassistant.token == "my_secret_token"


class TestGetSettings:
    """Tests for get_settings singleton."""

    def test_returns_config(self):
        """Test get_settings returns EdgeConfig instance."""
        settings = get_settings()
        assert isinstance(settings, EdgeConfig)

    def test_singleton_same_instance(self):
        """Test singleton returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
