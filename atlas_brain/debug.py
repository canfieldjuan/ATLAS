"""
Debug logging utilities for Atlas Brain.

Provides colorful, structured console output for debugging the voice pipeline.
"""

import time
from typing import Any, Optional
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

# Global console instance
console = Console()

# Track timing for latency display
_timers: dict[str, float] = {}


class DebugLogger:
    """
    Rich console logger for pipeline debugging.

    Usage:
        from atlas_brain.debug import debug

        debug.state("LISTENING")
        debug.transcript("Atlas turn off the TV", latency_ms=263)
        debug.intent("turn_off", "tv", confidence=0.95)
        debug.tool("device_command", {"device": "tv", "action": "turn_off"}, "success")
        debug.llm("Done, I've turned off the TV", latency_ms=412)
        debug.tts(duration_sec=1.2, latency_ms=89)
        debug.error("Connection failed", exc=e)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_state: Optional[str] = None

    def _timestamp(self) -> str:
        """Get current timestamp."""
        return time.strftime("%H:%M:%S")

    def state(self, state: str, extra: Optional[str] = None) -> None:
        """Log pipeline state change."""
        if not self.enabled:
            return

        # Skip duplicate states
        if state == self._last_state and state in ("LISTENING", "IDLE"):
            return
        self._last_state = state

        colors = {
            "LISTENING": "green",
            "RECORDING": "yellow",
            "TRANSCRIBING": "cyan",
            "PROCESSING": "blue",
            "RESPONDING": "magenta",
            "ERROR": "red",
            "IDLE": "dim",
        }
        color = colors.get(state.upper(), "white")

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("● ", style=color)
        text.append(state.upper(), style=f"bold {color}")
        if extra:
            text.append(f" {extra}", style="dim")

        console.print(text)

    def speech_detected(self) -> None:
        """Log VAD speech detection."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("◆ ", style="yellow")
        text.append("Speech detected", style="yellow")
        console.print(text)

    def transcript(self, text_content: str, latency_ms: Optional[float] = None) -> None:
        """Log STT transcription result."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("✓ ", style="green")
        text.append("STT: ", style="bold green")
        text.append(f'"{text_content}"', style="white")
        if latency_ms is not None:
            text.append(f" ({latency_ms:.0f}ms)", style="dim")

        console.print(text)

    def keyword(self, keyword: str, found: bool) -> None:
        """Log keyword detection result."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        if found:
            text.append("✓ ", style="green")
            text.append(f'Keyword "{keyword}" detected', style="green")
        else:
            text.append("✗ ", style="dim")
            text.append(f'No keyword "{keyword}" - ignoring', style="dim")

        console.print(text)

    def intent(
        self,
        action: str,
        target: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Log intent parsing result."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("→ ", style="cyan")
        text.append("INTENT: ", style="bold cyan")
        text.append(action, style="white")
        if target:
            text.append(f" → {target}", style="white")
        if confidence is not None:
            color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
            text.append(f" ({confidence:.2f})", style=color)

        console.print(text)

    def tool(
        self,
        tool_name: str,
        params: Optional[dict[str, Any]] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log tool execution."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("⚡ ", style="magenta")
        text.append("TOOL: ", style="bold magenta")
        text.append(tool_name, style="white")

        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            text.append(f"({param_str})", style="dim")

        if error:
            text.append(f" → ", style="dim")
            text.append(f"ERROR: {error}", style="red")
        elif result:
            text.append(f" → ", style="dim")
            text.append(result, style="green")

        console.print(text)

    def action(
        self,
        action: str,
        device: str,
        success: bool,
        message: Optional[str] = None,
    ) -> None:
        """Log device action execution."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("⚡ ", style="magenta")
        text.append("ACTION: ", style="bold magenta")
        text.append(f"{action}({device})", style="white")
        text.append(f" → ", style="dim")

        if success:
            text.append("success", style="green")
        else:
            text.append("failed", style="red")

        if message:
            text.append(f" - {message}", style="dim")

        console.print(text)

    def llm(
        self,
        response: str,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
    ) -> None:
        """Log LLM response."""
        if not self.enabled:
            return

        # Truncate long responses
        display_response = response[:80] + "..." if len(response) > 80 else response

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("✓ ", style="blue")
        text.append("LLM: ", style="bold blue")
        text.append(f'"{display_response}"', style="white")

        if latency_ms is not None:
            text.append(f" ({latency_ms:.0f}ms)", style="dim")
        if model:
            text.append(f" [{model}]", style="dim")

        console.print(text)

    def tts(
        self,
        duration_sec: Optional[float] = None,
        latency_ms: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> None:
        """Log TTS synthesis."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("♪ ", style="magenta")
        text.append("TTS: ", style="bold magenta")

        if duration_sec is not None:
            text.append(f"{duration_sec:.1f}s audio", style="white")

        if latency_ms is not None:
            text.append(f" ({latency_ms:.0f}ms)", style="dim")

        if voice:
            text.append(f" [{voice}]", style="dim")

        console.print(text)

    def error(
        self,
        message: str,
        exc: Optional[Exception] = None,
        context: Optional[str] = None,
    ) -> None:
        """Log error."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("✗ ", style="red")
        text.append("ERROR: ", style="bold red")
        text.append(message, style="red")

        if context:
            text.append(f" [{context}]", style="dim")

        console.print(text)

        if exc:
            console.print(f"           {type(exc).__name__}: {exc}", style="dim red")

    def info(self, message: str) -> None:
        """Log general info."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"{self._timestamp()} ", style="dim")
        text.append("ℹ ", style="blue")
        text.append(message, style="white")
        console.print(text)

    def separator(self, label: Optional[str] = None) -> None:
        """Print a visual separator."""
        if not self.enabled:
            return

        if label:
            console.rule(label, style="dim")
        else:
            console.rule(style="dim")

    # Timer helpers for measuring latency
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        _timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and return elapsed ms."""
        if name in _timers:
            elapsed = (time.perf_counter() - _timers[name]) * 1000
            del _timers[name]
            return elapsed
        return None


# Global debug logger instance
debug = DebugLogger(enabled=True)
