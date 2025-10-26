"""OpenAI LLM client."""
from typing import Any, Dict

import openai

from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    @retry_on_error(max_retries=3, initial_wait=30.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """Send a chat completion request with retry logic for 429/503."""
        response = self.client.chat.completions.create(
            model=model_name,
            messages=request["messages"],
            temperature=request.get("temperature", 0.0),
            seed=request.get("seed"),
        )
        payload = response.model_dump()
        choices = payload.get("choices") or []
        message = choices[0].get("message") if choices else {}
        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                    elif isinstance(item, str):
                        parts.append(item)
                content = "".join(parts)
            if not isinstance(content, str):
                content = str(content or "")

        usage = payload.get("usage") or {}
        result: ChatResult = {
            "text": (content or "").strip(),
            "usage": usage,
            "raw": payload,
            "reasoning_text": None,
            "process_tokens": None,
            "flags": {"leak_think": False},
        }
        return result


__all__ = ["OpenAIClient"]
