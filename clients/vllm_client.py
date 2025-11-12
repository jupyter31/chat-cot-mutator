"""vLLM local server client with reasoning support."""
from typing import Any, Dict, Optional
import logging

import openai

from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    """Client for locally hosted vLLM server with OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        """
        Initialize vLLM client.
        
        Args:
            base_url: The base URL of your vLLM server (default: http://localhost:8000/v1)
            api_key: API key (vLLM uses "EMPTY" by default for local servers)
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        logger.info(f"Initialized vLLM client with base_url: {base_url}")

    @retry_on_error(max_retries=3, initial_wait=5.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """
        Send a chat completion request to vLLM server.
        
        For Phi-4-reasoning with --enable-reasoning, the model may return
        reasoning in special tags or as part of the response.
        """
        # Build the request parameters
        params = {
            "model": model_name,
            "messages": request["messages"],
            "temperature": request.get("temperature", 0.0),
        }
        
        # Add seed if provided
        if "seed" in request and request["seed"] is not None:
            params["seed"] = request["seed"]
        
        # Add max_tokens if provided (important for reasoning models)
        if "max_tokens" in request:
            params["max_tokens"] = request["max_tokens"]
        
        # vLLM supports extra parameters
        if "extra_body" in request:
            params["extra_body"] = request["extra_body"]
        
        logger.debug(f"Sending vLLM request with params: {params}")
        
        # Send request to vLLM server
        response = self.client.chat.completions.create(**params)
        
        # Parse response
        payload = response.model_dump()
        choices = payload.get("choices") or []
        message = choices[0].get("message") if choices else {}
        
        # Extract content
        content = ""
        reasoning_text = None
        
        if isinstance(message, dict):
            content = message.get("content") or ""
            
            # Handle list-type content (multimodal responses)
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
            
            # Check for reasoning in the response
            # vLLM with --enable-reasoning --reasoning-parser deepseek_r1
            # may include reasoning in <think> tags or separate field
            if "<think>" in content and "</think>" in content:
                import re
                think_pattern = r"<think>\s*(?P<body>.+?)\s*</think>"
                match = re.search(think_pattern, content, re.DOTALL)
                if match:
                    reasoning_text = match.group("body").strip()
                    # Optionally remove thinking tags from final content
                    # content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        
        # Extract usage information
        usage = payload.get("usage") or {}
        
        # Check for reasoning_content in vLLM's extended response
        # Some vLLM versions may provide reasoning in a separate field
        if not reasoning_text and "reasoning_content" in message:
            reasoning_text = message["reasoning_content"]
        
        # Build ChatResult
        result: ChatResult = {
            "text": (content or "").strip(),
            "usage": usage,
            "raw": payload,
            "reasoning_text": reasoning_text,
            "process_tokens": usage.get("prompt_tokens"),  # vLLM includes this in usage
            "flags": {
                "leak_think": reasoning_text is not None,  # Flag if reasoning was extracted
            },
        }
        
        logger.debug(f"vLLM response: text_len={len(result['text'])}, "
                    f"reasoning_len={len(reasoning_text) if reasoning_text else 0}")
        
        return result


__all__ = ["VLLMClient"]
