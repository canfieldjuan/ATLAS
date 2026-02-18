"""
Tests for voice pipeline error recovery TTS fallback phrases.

Covers:
- Config defaults and custom phrases
- ErrorPhrase type marker (isinstance-based, no string collision)
- _speak_error with _on_error_playback_done (resets wake + resumes
  conversation mode without incrementing turn count)
- Empty/None phrase guard in _speak_error
- All pipeline silent-exit paths (ASR empty, agent empty, streaming empty)
- Launcher error returns (TimeoutError vs general exception)
- Streaming partial output + fallback sentence clear sentinel
- Command generation counter (stale command discard)
- Thread-safe sentence collector
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from atlas_brain.voice.pipeline import ErrorPhrase, VoicePipeline


# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #

def _make_raw_pipeline(**overrides):
    """Create a VoicePipeline with __init__ bypassed and common attrs set."""
    from atlas_brain.voice.pipeline import VoicePipeline

    with patch.object(VoicePipeline, "__init__", lambda self, **kw: None):
        p = VoicePipeline.__new__(VoicePipeline)

    p.playback = MagicMock()
    p._on_playback_start = MagicMock()
    p._on_playback_done = MagicMock()
    p.error_asr_empty = "Didn't catch that."
    p.error_agent_timeout = "Too long."
    p.error_agent_failed = "Something broke."
    p.asr_client = MagicMock()
    p.sample_rate = 16000
    p._last_audio_buffer = None
    p._last_speaker_match = None
    p.speaker_id_enabled = False
    p.streaming_llm_enabled = False
    p.streaming_agent_runner = None
    p.agent_runner = MagicMock(return_value="")
    p.session_id = "test-session"
    p.node_id = "test-node"
    p._workflow_active = False
    p._command_gen = 0
    p._command_gen_lock = threading.Lock()
    p.conversation_mode_enabled = False
    p.conversation_start_delay_ms = 500
    p.turn_limit_enabled = False
    p.model = MagicMock()
    p.frame_processor = MagicMock()
    p._build_context = MagicMock(return_value={})
    p._filler_enabled = False
    p._filler_delay = 2.0
    p._filler_phrases = ["One moment..."]
    p._filler_followup_delay = 3.0
    p._filler_followup_phrases = ["Still working on it..."]
    p._last_llm_call_time = 0.0

    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# ------------------------------------------------------------------ #
# Config defaults
# ------------------------------------------------------------------ #


class TestErrorPhraseConfig:

    def test_defaults(self):
        from atlas_brain.config import VoiceClientConfig
        cfg = VoiceClientConfig()
        assert cfg.error_asr_empty == "Sorry, I didn't catch that."
        assert cfg.error_agent_timeout == "Sorry, that took too long. Try again."
        assert cfg.error_agent_failed == "Something went wrong. Try again."

    def test_custom_phrases(self):
        from atlas_brain.config import VoiceClientConfig
        cfg = VoiceClientConfig(
            error_asr_empty="Custom ASR",
            error_agent_timeout="Custom timeout",
            error_agent_failed="Custom fail",
        )
        assert cfg.error_asr_empty == "Custom ASR"
        assert cfg.error_agent_timeout == "Custom timeout"
        assert cfg.error_agent_failed == "Custom fail"


# ------------------------------------------------------------------ #
# ErrorPhrase type marker
# ------------------------------------------------------------------ #


class TestErrorPhraseType:

    def test_isinstance_str(self):
        ep = ErrorPhrase("hello")
        assert isinstance(ep, str)
        assert isinstance(ep, ErrorPhrase)

    def test_equality_with_str(self):
        ep = ErrorPhrase("hello")
        assert ep == "hello"

    def test_plain_str_not_error_phrase(self):
        assert not isinstance("hello", ErrorPhrase)

    def test_is_error_phrase_static(self):
        from atlas_brain.voice.pipeline import VoicePipeline
        assert VoicePipeline._is_error_phrase(ErrorPhrase("x")) is True
        assert VoicePipeline._is_error_phrase("x") is False
        assert VoicePipeline._is_error_phrase("") is False

    def test_no_collision_with_identical_string(self):
        """A plain string matching an error phrase text is NOT detected."""
        from atlas_brain.voice.pipeline import VoicePipeline
        assert VoicePipeline._is_error_phrase("Sorry, I didn't catch that.") is False
        assert VoicePipeline._is_error_phrase(
            ErrorPhrase("Sorry, I didn't catch that.")
        ) is True


# ------------------------------------------------------------------ #
# _speak_error behaviour
# ------------------------------------------------------------------ #


class TestSpeakError:

    def test_calls_playback_with_error_done(self):
        p = _make_raw_pipeline()
        p._speak_error("Test phrase")
        p.playback.speak.assert_called_once()
        kw = p.playback.speak.call_args.kwargs
        assert kw["on_start"] is p._on_playback_start
        assert kw["on_done"].__func__ is VoicePipeline._on_error_playback_done

    def test_on_done_is_error_variant(self):
        """_speak_error passes _on_error_playback_done, NOT _on_playback_done."""
        p = _make_raw_pipeline()
        p._speak_error("Test phrase")
        kw = p.playback.speak.call_args.kwargs
        assert kw["on_done"].__func__ is not VoicePipeline._on_playback_done

    def test_swallows_exception(self):
        p = _make_raw_pipeline()
        p.playback.speak.side_effect = RuntimeError("TTS broken")
        p._speak_error("Test phrase")  # should not raise

    def test_empty_phrase_skips_playback(self):
        p = _make_raw_pipeline()
        p._speak_error("")
        p.playback.speak.assert_not_called()

    def test_none_phrase_skips_playback(self):
        p = _make_raw_pipeline()
        p._speak_error(None)
        p.playback.speak.assert_not_called()


# ------------------------------------------------------------------ #
# _on_error_playback_done
# ------------------------------------------------------------------ #


class TestOnErrorPlaybackDone:

    def test_resets_wake_word_model(self):
        p = _make_raw_pipeline()
        p._on_error_playback_done()
        p.model.reset.assert_called_once()

    def test_does_not_increment_turn_count(self):
        p = _make_raw_pipeline(
            conversation_mode_enabled=True,
            turn_limit_enabled=True,
        )
        p._on_error_playback_done()
        p.frame_processor.increment_turn_count.assert_not_called()

    def test_re_enters_conversation_mode(self):
        """If conversation_mode_enabled, schedules re-entry after error."""
        p = _make_raw_pipeline(conversation_mode_enabled=True)
        with patch("threading.Timer") as mock_timer:
            p._on_error_playback_done()
            mock_timer.assert_called_once()

    def test_no_conversation_when_disabled(self):
        p = _make_raw_pipeline(conversation_mode_enabled=False)
        with patch("threading.Timer") as mock_timer:
            p._on_error_playback_done()
            mock_timer.assert_not_called()


# ------------------------------------------------------------------ #
# Pipeline empty paths
# ------------------------------------------------------------------ #


class TestPipelineEmptyPaths:

    def test_batch_asr_empty(self):
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="")
        p._handle_command(b"\x00" * 3200)
        assert p.playback.speak.call_args[0][0] == "Didn't catch that."

    def test_streaming_asr_empty(self):
        p = _make_raw_pipeline()
        p._handle_streaming_command("", b"\x00" * 3200)
        assert p.playback.speak.call_args[0][0] == "Didn't catch that."

    def test_batch_agent_empty(self):
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="hello")
        p.agent_runner = MagicMock(return_value="")
        p._handle_command(b"\x00" * 3200)
        assert p.playback.speak.call_args[0][0] == "Something broke."

    def test_streaming_agent_empty(self):
        p = _make_raw_pipeline()
        p.agent_runner = MagicMock(return_value="")
        p._handle_streaming_command("hello", b"\x00" * 3200)
        assert p.playback.speak.call_args[0][0] == "Something broke."

    def test_streaming_llm_no_sentences(self):
        p = _make_raw_pipeline(
            streaming_llm_enabled=True,
            streaming_agent_runner=MagicMock(),  # does nothing
        )
        p._handle_streaming_llm_command("hello")
        assert p.playback.speak.call_args[0][0] == "Something broke."

    def test_streaming_llm_empty_transcript(self):
        p = _make_raw_pipeline()
        p._handle_streaming_llm_command("")
        assert p.playback.speak.call_args[0][0] == "Didn't catch that."


# ------------------------------------------------------------------ #
# ErrorPhrase routing through pipeline (no conversation mode on error)
# ------------------------------------------------------------------ #


class TestErrorPhraseRouting:

    def test_batch_error_phrase_uses_speak_error(self):
        """ErrorPhrase from launcher goes through _speak_error, not normal speak."""
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="hello")
        p.agent_runner = MagicMock(return_value=ErrorPhrase("Too long."))
        p._handle_command(b"\x00" * 3200)
        kw = p.playback.speak.call_args.kwargs
        # _speak_error passes _on_error_playback_done, not _on_playback_done
        assert kw["on_done"] is not None and kw["on_done"].__func__ is VoicePipeline._on_error_playback_done  # bound method check

    def test_streaming_cmd_error_phrase_uses_speak_error(self):
        p = _make_raw_pipeline()
        p.agent_runner = MagicMock(return_value=ErrorPhrase("Something broke."))
        p._handle_streaming_command("hello", b"\x00" * 3200)
        kw = p.playback.speak.call_args.kwargs
        assert kw["on_done"] is not None and kw["on_done"].__func__ is VoicePipeline._on_error_playback_done  # bound method check

    def test_normal_reply_uses_on_playback_done(self):
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="hello")
        p.agent_runner = MagicMock(return_value="The weather is nice today.")
        p._handle_command(b"\x00" * 3200)
        kw = p.playback.speak.call_args.kwargs
        assert kw["on_done"] is p._on_playback_done

    def test_streaming_llm_single_error_phrase(self):
        """Single ErrorPhrase sentence routes on_done to _on_error_playback_done."""
        def mock_runner(transcript, context, on_sentence):
            on_sentence(ErrorPhrase("Something broke."))

        p = _make_raw_pipeline(
            streaming_llm_enabled=True,
            streaming_agent_runner=mock_runner,
        )
        p._handle_streaming_llm_command("hello")
        # ErrorPhrase goes through speak_streamed (queue-based), not speak
        kw = p.playback.speak_streamed.call_args.kwargs
        on_done = kw["on_done"]
        # The dynamic on_done closure should call _on_error_playback_done
        # when is_error[0] is True (set by the ErrorPhrase sentence)
        on_done()
        p.model.reset.assert_called()

    def test_plain_string_matching_error_text_not_intercepted(self):
        """A normal LLM reply that happens to match error text goes through
        normal playback (not _speak_error) because it is a plain str."""
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="hello")
        p.agent_runner = MagicMock(return_value="Too long.")
        p._handle_command(b"\x00" * 3200)
        kw = p.playback.speak.call_args.kwargs
        assert kw["on_done"] is p._on_playback_done


# ------------------------------------------------------------------ #
# Launcher error returns
# ------------------------------------------------------------------ #


class TestLauncherAgentRunner:

    def _run_with_exception(self, exc):
        from atlas_brain.voice import launcher

        original_loop = launcher._event_loop
        try:
            launcher._event_loop = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = exc

            with patch("atlas_brain.voice.launcher.get_agent", return_value=MagicMock()), \
                 patch("asyncio.run_coroutine_threadsafe", return_value=mock_future), \
                 patch("atlas_brain.voice.launcher.settings") as mock_settings:
                mock_settings.voice.agent_timeout = 30.0
                mock_settings.voice.error_agent_timeout = "Too slow!"
                mock_settings.voice.error_agent_failed = "Broke!"
                runner = launcher._create_agent_runner()
                return runner("hello", {"session_id": "s1", "node_id": "n1"})
        finally:
            launcher._event_loop = original_loop

    def test_timeout_returns_error_phrase_type(self):
        result = self._run_with_exception(TimeoutError("timed out"))
        assert isinstance(result, ErrorPhrase)
        assert result == "Too slow!"

    def test_exception_returns_error_phrase_type(self):
        result = self._run_with_exception(RuntimeError("boom"))
        assert isinstance(result, ErrorPhrase)
        assert result == "Broke!"


# ------------------------------------------------------------------ #
# Speaker verification exception
# ------------------------------------------------------------------ #


class TestSpeakerVerificationException:

    def test_speaks_error_on_crash(self):
        p = _make_raw_pipeline(
            speaker_id_enabled=True,
            require_known_speaker=True,
            speaker_id_service=MagicMock(),
            speaker_id_timeout=5.0,
            event_loop=MagicMock(),
        )
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("crashed")
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = p._verify_speaker_blocking(b"\x00" * 3200)
        assert result is False
        assert p.playback.speak.call_args[0][0] == "Something broke."


# ------------------------------------------------------------------ #
# FrameProcessor on_asr_error callback
# ------------------------------------------------------------------ #


class TestFrameProcessorAsrError:

    def test_stored_and_defaults_none(self):
        from atlas_brain.voice.frame_processor import FrameProcessor
        cb = MagicMock()
        fp = FrameProcessor(
            wake_predict=MagicMock(return_value={}),
            wake_threshold=0.5,
            segmenter=MagicMock(),
            vad=MagicMock(),
            allow_wake_barge_in=False,
            on_asr_error=cb,
        )
        assert fp.on_asr_error is cb

        fp2 = FrameProcessor(
            wake_predict=MagicMock(return_value={}),
            wake_threshold=0.5,
            segmenter=MagicMock(),
            vad=MagicMock(),
            allow_wake_barge_in=False,
        )
        assert fp2.on_asr_error is None


# ------------------------------------------------------------------ #
# Streaming sentence clear sentinel
# ------------------------------------------------------------------ #


class TestStreamingSentenceClear:

    def test_none_sentinel_clears_partial_sentences(self):
        def mock_runner(transcript, context, on_sentence):
            on_sentence("Partial one.")
            on_sentence("Partial two.")
            on_sentence(None)
            on_sentence("Complete fallback response.")

        p = _make_raw_pipeline(
            streaming_llm_enabled=True,
            streaming_agent_runner=mock_runner,
        )
        p._handle_streaming_llm_command("hello")
        spoken = p.playback.speak.call_args[0][0]
        assert spoken == "Complete fallback response."
        assert "Partial" not in spoken


# ------------------------------------------------------------------ #
# Command generation counter
# ------------------------------------------------------------------ #


class TestCommandGenerationCounter:

    def test_generation_increments(self):
        p = _make_raw_pipeline()
        g1 = p._next_command_gen()
        g2 = p._next_command_gen()
        assert g2 == g1 + 1

    def test_is_current_gen(self):
        p = _make_raw_pipeline()
        g1 = p._next_command_gen()
        assert p._is_current_gen(g1) is True
        g2 = p._next_command_gen()
        assert p._is_current_gen(g1) is False
        assert p._is_current_gen(g2) is True

    def test_stale_command_discards_reply(self):
        """A slow command whose generation is superseded should not speak."""
        p = _make_raw_pipeline()
        p.asr_client.transcribe_pcm = MagicMock(return_value="hello")

        # Simulate: agent_runner is slow, another command claims a new gen
        def slow_agent(transcript, context):
            p._next_command_gen()  # another command started
            return "stale reply"

        p.agent_runner = slow_agent
        p._handle_command(b"\x00" * 3200)
        p.playback.speak.assert_not_called()


# ------------------------------------------------------------------ #
# Thread-safe sentence collector
# ------------------------------------------------------------------ #


class TestThreadSafeSentences:

    def test_concurrent_appends(self):
        """Multiple threads writing to on_sentence should not lose data."""
        import queue as _queue

        p = _make_raw_pipeline(streaming_llm_enabled=True)

        # Capture the queue passed to speak_streamed
        captured_queue = [None]
        original_speak_streamed = p.playback.speak_streamed

        def capture_speak_streamed(q, **kw):
            captured_queue[0] = q
            original_speak_streamed(q, **kw)

        p.playback.speak_streamed = capture_speak_streamed

        barrier = threading.Barrier(4)

        def mock_runner(transcript, context, on_sentence):
            def writer(text):
                barrier.wait(timeout=2)
                on_sentence(text)
            threads = [
                threading.Thread(target=writer, args=(f"s{i}",))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=2)

        p.streaming_agent_runner = mock_runner
        p._handle_streaming_llm_command("hello")

        # Drain the queue to verify all 4 sentences + sentinel
        q = captured_queue[0]
        assert q is not None
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        # Should have 4 sentences + None sentinel
        sentences = [s for s in items if s is not None]
        assert len(sentences) == 4
        assert set(sentences) == {"s0", "s1", "s2", "s3"}
