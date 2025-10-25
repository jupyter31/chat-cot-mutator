"""OpenAI LLM client."""
from typing import Any, Dict

import openai

from .base_llm_client import BaseLLMClient, retry_on_error


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    @retry_on_error(max_retries=3, initial_wait=30.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a chat completion request with retry logic for 429/503."""
        response = self.client.chat.completions.create(
            model=model_name,
            messages=request["messages"],
            temperature=request.get("temperature", 0.0),
            seed=request.get("seed"),
        )
        return response.model_dump()


__all__ = ["OpenAIClient"]
