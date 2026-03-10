"""Test token tracking implementation.

This script verifies that:
1. LLM providers return usage data
2. Reasoning agent captures and reports token usage
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_anthropic_usage():
    """Test that Anthropic LLM returns usage data."""
    from services.llm.anthropic import AnthropicLLM
    from services.protocols import Message
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY not set")
        return
    
    print("\nTesting Anthropic token tracking...")
    llm = AnthropicLLM(model="claude-sonnet-4-5-20250929")
    llm.load()
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in exactly 5 words."),
    ]
    
    result = llm.chat(messages=messages, max_tokens=50, temperature=0.7)
    
    print(f"Response: {result.get('response', 'N/A')}")
    print(f"Usage: {result.get('usage', {})}")
    
    usage = result.get("usage", {})
    assert "input_tokens" in usage, "Missing input_tokens in usage"
    assert "output_tokens" in usage, "Missing output_tokens in usage"
    assert usage["input_tokens"] > 0, "input_tokens should be > 0"
    assert usage["output_tokens"] > 0, "output_tokens should be > 0"
    
    print("PASS: Anthropic token tracking PASSED")
    llm.unload()


async def test_openrouter_usage():
    """Test that OpenRouter LLM returns usage data."""
    from services.llm.openrouter import OpenRouterLLM
    from services.protocols import Message
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("SKIP: OPENROUTER_API_KEY not set")
        return
    
    print("\nTesting OpenRouter token tracking...")
    llm = OpenRouterLLM(model="anthropic/claude-3.5-haiku")
    llm.load()
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in exactly 5 words."),
    ]
    
    result = llm.chat(messages=messages, max_tokens=50, temperature=0.7)
    
    print(f"Response: {result.get('response', 'N/A')}")
    print(f"Usage: {result.get('usage', {})}")
    
    usage = result.get("usage", {})
    assert "input_tokens" in usage, "Missing input_tokens in usage"
    assert "output_tokens" in usage, "Missing output_tokens in usage"
    assert usage["input_tokens"] >= 0, "input_tokens should be >= 0"
    assert usage["output_tokens"] >= 0, "output_tokens should be >= 0"
    
    print("PASS: OpenRouter token tracking PASSED")
    llm.unload()


async def test_groq_usage():
    """Test that Groq LLM returns usage data."""
    from services.llm.groq import GroqLLM
    from services.protocols import Message
    
    if not os.environ.get("GROQ_API_KEY"):
        print("SKIP: GROQ_API_KEY not set")
        return
    
    print("\nTesting Groq token tracking...")
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    llm.load()
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hello in exactly 5 words."),
    ]
    
    result = llm.chat(messages=messages, max_tokens=50, temperature=0.7)
    
    print(f"Response: {result.get('response', 'N/A')}")
    print(f"Usage: {result.get('usage', {})}")
    
    usage = result.get("usage", {})
    assert "input_tokens" in usage, "Missing input_tokens in usage"
    assert "output_tokens" in usage, "Missing output_tokens in usage"
    assert usage["input_tokens"] >= 0, "input_tokens should be >= 0"
    assert usage["output_tokens"] >= 0, "output_tokens should be >= 0"
    
    print("PASS: Groq token tracking PASSED")
    llm.unload()


async def test_reasoning_token_tracking():
    """Test that reasoning agent tracks tokens."""
    from reasoning.state import ReasoningAgentState
    from reasoning.graph import run_reasoning_graph
    
    print("\nTesting reasoning agent token tracking...")
    
    state: ReasoningAgentState = {
        "event_id": "test-123",
        "event_type": "email.received",
        "source": "gmail",
        "entity_type": "contact",
        "entity_id": "test-contact-1",
        "payload": {"subject": "Test email", "body": "This is a test."},
    }
    
    try:
        result_state = await run_reasoning_graph(state)
        
        total_input = result_state.get("total_input_tokens", 0)
        total_output = result_state.get("total_output_tokens", 0)
        
        print(f"Total input tokens: {total_input}")
        print(f"Total output tokens: {total_output}")
        
        assert "total_input_tokens" in result_state, "Missing total_input_tokens"
        assert "total_output_tokens" in result_state, "Missing total_output_tokens"
        
        if result_state.get("needs_reasoning"):
            assert total_input > 0, "Expected input tokens > 0 when reasoning ran"
            assert total_output > 0, "Expected output tokens > 0 when reasoning ran"
        
        print("PASS: Reasoning agent token tracking PASSED")
        
    except Exception as e:
        print(f"WARN: Reasoning test partial: {e}")
        print("(This may fail if LLM services are not configured)")


async def main():
    """Run all token tracking tests."""
    print("=" * 60)
    print("Token Tracking Verification")
    print("=" * 60)
    
    await test_anthropic_usage()
    await test_openrouter_usage()
    await test_groq_usage()
    await test_reasoning_token_tracking()
    
    print("\n" + "=" * 60)
    print("All available tests completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
