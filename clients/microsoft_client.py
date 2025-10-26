"""Azure-hosted client with DeepSeek-style reasoning capture."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from ._think_splitter import ThinkSplitter
from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error


def _as_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[no-any-return]
    if hasattr(obj, "__dict__"):
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    return {}


def _iter_text_chunks(choice: Mapping[str, Any]) -> Iterable[str]:
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, Iterable):
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        yield text
                elif isinstance(item, str):
                    yield item
    elif hasattr(delta, "content"):
        content = getattr(delta, "content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, Iterable):
            for item in content:
                if isinstance(item, str):
                    yield item


class MicrosoftLLMClient(BaseLLMClient):
    """Client that targets Azure OpenAI-compatible endpoints with think-tag support."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        think_chars_cap: int = 4096 * 4,
        **client_kwargs: Any,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be provided for MicrosoftLLMClient")
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - exercised in external environments
            raise ImportError(
                "MicrosoftLLMClient requires the 'openai' package with AzureOpenAI support"
            ) from exc

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            **client_kwargs,
        )
        self._think_cap = think_chars_cap

    def _build_params(self, model_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "model": model_name,
            "messages": request.get("messages", []),
            "temperature": request.get("temperature", 0.0),
        }
        if "seed" in request:
            params["seed"] = request["seed"]
        return params

    def _finalize_result(
        self,
        splitter: ThinkSplitter,
        usage: Mapping[str, Any],
        process_tokens: Optional[int],
    ) -> ChatResult:
        result = splitter.finish()
        visible = result.visible.strip()
        reasoning = result.reasoning.strip()
        if not visible and reasoning:
            # If the think block consumed everything, keep the raw visible text
            visible = ""
        flags = {
            "leak_think": "<think>" in visible.lower(),
            "think_capped": result.capped,
        }
        raw = {
            "message": {"role": "assistant", "content": visible},
            "reasoning": reasoning,
        }
        return {
            "text": visible,
            "usage": dict(usage),
            "raw": raw,
            "reasoning_text": reasoning or None,
            "process_tokens": process_tokens,
            "flags": flags,
        }

    @retry_on_error(max_retries=3, initial_wait=15.0)
    def send_chat_request(self, model_name: str, request: Dict[str, Any]) -> ChatResult:
        params = self._build_params(model_name, request)
        splitter = ThinkSplitter(cap_chars=self._think_cap)
        usage: Mapping[str, Any] = {}
        process_tokens: Optional[int] = None

        try:
            stream = self.client.chat.completions.create(stream=True, **params)
            for chunk in stream:
                data = _as_dict(chunk)
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    choice = _as_dict(choices[0])
                    for text_piece in _iter_text_chunks(choice):
                        splitter.feed(text_piece)
                    if process_tokens is None:
                        details = choice.get("content_filter_results")
                        if isinstance(details, Mapping):
                            reasoning_tokens = details.get("reasoning_tokens")
                            if isinstance(reasoning_tokens, int):
                                process_tokens = reasoning_tokens
                usage_blob = data.get("usage")
                if isinstance(usage_blob, Mapping):
                    usage = usage_blob
                    details = usage_blob.get("output_tokens_details")
                    if isinstance(details, Mapping):
                        reasoning_tokens = details.get("reasoning_tokens")
                        if isinstance(reasoning_tokens, int):
                            process_tokens = reasoning_tokens
        except Exception:
            # Fall back to non-streaming call if streaming unsupported
            response = self.client.chat.completions.create(stream=False, **params)
            payload = _as_dict(response)
            choices = payload.get("choices")
            content = ""
            if isinstance(choices, list) and choices:
                message = _as_dict(choices[0]).get("message")
                if isinstance(message, Mapping):
                    content = message.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text") if isinstance(part, Mapping) else str(part)
                            for part in content
                        )
            splitter.feed(str(content))
            usage = _as_dict(payload.get("usage", {}))
            return self._finalize_result(splitter, usage, process_tokens)

        return self._finalize_result(splitter, usage, process_tokens)


__all__ = ["MicrosoftLLMClient"]
