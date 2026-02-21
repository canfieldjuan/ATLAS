"""
Ollama LLM Client for graphiti-core.

Provides structured output support by prompting the model to return JSON
and parsing the response, since Ollama doesn't support instructor/response_model.
"""
import json
import logging
import re
from typing import Any

import httpx
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


class OllamaLLMClient(LLMClient):
    """
    LLM client for Ollama using OpenAI-compatible API.

    Handles structured outputs by:
    1. Adding JSON schema instructions to the prompt
    2. Parsing JSON from the model's response
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        base_url: str = "http://localhost:11434/v1",
        max_tokens: int = 4096,
        model: str = "hermes3:8b-q4",
    ):
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
            )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from model response."""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _build_schema_prompt(self, response_model: type) -> str:
        """Build a prompt describing the expected JSON schema."""
        if response_model is None:
            return ""

        # Get schema from Pydantic model if available
        if hasattr(response_model, "model_json_schema"):
            schema = response_model.model_json_schema()
            return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\nRespond ONLY with the JSON object, no other text."

        return "\n\nRespond with valid JSON only."

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Internal method to generate response from Ollama.

        This implements the abstract method from LLMClient.
        """
        await self._ensure_client()

        # Convert messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Add JSON schema instructions if structured output needed
        if response_model and formatted_messages:
            schema_prompt = self._build_schema_prompt(response_model)
            if schema_prompt:
                # Append to the last message
                formatted_messages[-1]["content"] += schema_prompt

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0.7,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.debug("Ollama response: %s", content[:200])

            # If structured output expected, parse JSON
            if response_model:
                parsed = self._extract_json(content)
                if parsed:
                    return parsed
                else:
                    logger.warning("Failed to parse JSON from response: %s", content[:200])
                    # Return raw content as fallback
                    return {"content": content}

            return {"content": content}

        except httpx.HTTPStatusError as e:
            logger.error("Ollama API error: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate response from Ollama.

        If response_model is provided, instructs the model to return JSON
        and parses the response.
        """
        return await self._generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            **kwargs,
        )


def create_ollama_llm_client(
    model: str = "hermes3:8b-q4",
    base_url: str = "http://localhost:11434/v1",
    max_tokens: int = 4096,
) -> OllamaLLMClient:
    """Factory function to create Ollama LLM client."""
    return OllamaLLMClient(
        base_url=base_url,
        max_tokens=max_tokens,
        model=model,
    )
