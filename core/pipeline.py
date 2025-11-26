"""Core pipeline helpers for headless batch runs."""
from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

from core.schema import (
    FrozenContextRecord,
    FrozenPassageRecord,
    FrozenToolRecord,
    SampleRecord,
)
from eval.judges import judge_grounding, judge_answer_correctness


CitationPattern = re.compile(r"\[\[([^\]]+)\]\]")
_CITE_BRACKETS = re.compile(r"\s*\[\[[^\]]+\]\]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")
_REASONING_PATTERN = re.compile(
    r"Reasoning:\s*(?P<body>.+?)\s*Final Answer:", re.IGNORECASE | re.DOTALL
)

# Think tag patterns for reasoning models (e.g., Phi-4, DeepSeek-R1)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_THINK_TAG_PATTERN = re.compile(
    r"<think>\s*(?P<body>.+?)\s*</think>", re.IGNORECASE | re.DOTALL
)


def _strip_citations(text: str) -> str:
    return _CITE_BRACKETS.sub(" ", text or "").strip()


def _normalize_answer(text: str) -> str:
    """Lowercase, strip citations/punct/articles, collapse spaces (SQuAD-style)."""
    s = _strip_citations(text)
    s = unicodedata.normalize("NFKD", s).lower()
    s = _NON_ALNUM.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _answers_match(pred: str, gold: str) -> bool:
    """Exact after normalization, or gold-tokens ⊆ pred-tokens (handles name variants)."""
    pn, gn = _normalize_answer(pred), _normalize_answer(gold)
    if pn == gn:
        return True
    pset, gset = set(pn.split()), set(gn.split())
    return gset.issubset(pset)


@dataclass
class PromptTemplates:
    condition_to_template: Mapping[str, str]
    system_prompt: str = "You are a careful assistant who cites evidence accurately."
    evidence_channel: str = "tool"
    tool_variant: str = "direct"
    cot_injection_channel: str = "system"
    cot_injection_add_user_prompt: bool = False  # Add follow-up user message after injecting mutated CoT

    def __getitem__(self, condition: str) -> str:
        return self.condition_to_template[condition]


def _format_passage_label(passage: FrozenPassageRecord, idx: int) -> str:
    if passage.cite:
        return passage.cite
    if passage.doc_id:
        return passage.doc_id
    return f"passage_{idx+1}"


def _format_tool_output(tool: FrozenToolRecord) -> str:
    input_repr = tool.input if isinstance(tool.input, str) else json.dumps(tool.input)
    output_repr = tool.output if isinstance(tool.output, str) else json.dumps(tool.output)
    return f"- {tool.tool} | input={input_repr} | output={output_repr}"


def _build_evidence_entries(context: FrozenContextRecord) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, passage in enumerate(context.passages):
        label = _format_passage_label(passage, idx)
        entry: Dict[str, Any] = {
            "type": "passage",
            "index": idx,
            "label": label,
            "text": passage.text,
            "display": f"[{label}] {passage.text}",
        }
        if passage.doc_id:
            entry["doc_id"] = passage.doc_id
        if passage.cite:
            entry["cite"] = passage.cite
        entries.append(entry)

    for tool_idx, tool in enumerate(context.tool_outputs):
        display = _format_tool_output(tool)
        entry = {
            "type": "tool_output",
            "index": tool_idx,
            "label": f"tool_{tool_idx + 1}",
            "tool": tool.tool,
            "input": tool.input,
            "output": tool.output,
            "text": display,
            "display": display,
        }
        entries.append(entry)
    return entries


def _format_evidence(context: FrozenContextRecord) -> str:
    entries = _build_evidence_entries(context)
    lines: List[str] = []
    tool_section_started = False
    for entry in entries:
        if entry["type"] == "passage":
            lines.append(entry["display"])
        else:
            if not tool_section_started:
                lines.append("\nFROZEN TOOL OUTPUTS:")
                tool_section_started = True
            lines.append(entry["display"])
    return "\n".join(lines).strip()


def _find_last_user_content(messages: List[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                collected: List[str] = []
                for item in content:
                    if isinstance(item, Mapping):
                        text = item.get("text")
                        if isinstance(text, str):
                            collected.append(text)
                    elif isinstance(item, str):
                        collected.append(item)
                if collected:
                    return "".join(collected).strip()
    return ""


def assemble_messages(
    condition: str,
    sample: SampleRecord,
    mutated_cot: Optional[str] = None,
    prompts: Optional[PromptTemplates] = None,
    use_string_arguments: bool = True,
) -> List[Dict[str, Any]]:
    """Assemble chat messages for the given condition.
    
    Args:
        condition: Condition identifier (A, B, C, D, etc.)
        sample: Sample record with query and context
        mutated_cot: Optional mutated CoT text
        prompts: Prompt templates
        use_string_arguments: If True, use JSON string for tool_calls arguments (Microsoft/OpenAI).
                            If False, use dict objects (Ollama). Default: True for compatibility.
    """

    if prompts is None:
        raise ValueError("Prompt templates must be provided")

    template = prompts[condition]
    evidence_entries = _build_evidence_entries(sample.frozen_context)
    evidence_text = _format_evidence(sample.frozen_context)
    mutated_cot_text = _as_text(mutated_cot) if mutated_cot is not None else ""

    messages: List[Dict[str, Any]] = []
    
    # Extract instructions and question from template
    # Instructions are everything except the {query} placeholder
    # We'll split on "Question:" or similar patterns to separate them
    format_args: Dict[str, Any] = {
        "query": sample.query,
        "evidence": evidence_text,
    }

    if "{cot}" in template:
        format_args["cot"] = mutated_cot_text
    if "{mutated_cot_text}" in template:
        format_args["mutated_cot_text"] = mutated_cot_text
    if "{evidence_block}" in template:
        format_args["evidence_block"] = evidence_text

    # Format the complete template to get instructions text
    full_text = template.format(**format_args).strip()
    
    # Extract instructions by replacing the actual query with a marker
    # and splitting on it to get the instructions portion
    instructions_text = template.replace("{query}", "").strip()
    
    # Build system message with all instructions
    system_content_parts = []
    
    # Add base system prompt if exists
    system_prompt = (prompts.system_prompt or "").strip()
    if system_prompt:
        system_content_parts.append(system_prompt)
    
    # Add condition-specific instructions
    system_content_parts.append(instructions_text)
    
    messages.append({"role": "system", "content": "\n\n".join(system_content_parts)})
    
    # Add first user message with just the question
    messages.append({"role": "user", "content": f"Question: {sample.query}"})

    # Add evidence as tool messages
    tool_variant = (prompts.tool_variant or "direct").lower()
    evidence_channel = prompts.evidence_channel or "tool"
    
    logger.info(f"[DEBUG] tool_variant={tool_variant}, evidence_channel={evidence_channel}")
    logger.info(f"[DEBUG] prompts.tool_variant={prompts.tool_variant}, prompts.evidence_channel={prompts.evidence_channel}")

    if tool_variant not in {"direct", "function_call"}:
        raise ValueError(f"Unsupported tool_variant: {tool_variant}")

    for idx, entry in enumerate(evidence_entries):
        payload = json.dumps(entry, ensure_ascii=False)
        needs_tool_call = tool_variant == "function_call" or evidence_channel == "tool"
        call_id = f"tool_call_{idx}"

        if needs_tool_call:
            # Note: Different APIs expect different formats for tool_calls arguments:
            # - Microsoft/OpenAI: requires JSON string
            # - Ollama: accepts either dict or JSON string (more flexible)
            # Use use_string_arguments parameter to control format
            arguments_value = json.dumps(entry, ensure_ascii=False) if use_string_arguments else entry
            
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "evidence_passage",
                                "arguments": arguments_value,
                            },
                        }
                    ],
                }
            )

        tool_message: Dict[str, Any] = {
            "role": evidence_channel,
            "name": "evidence_passage",
            "content": payload,  # Keep as JSON string for content
        }

        if needs_tool_call:
            tool_message["tool_call_id"] = call_id
            if evidence_channel == "tool":
                tool_message["role"] = "tool"

        messages.append(tool_message)

    # For conditions C, D, C_prime, D_prime: inject mutated CoT as assistant message FIRST
    # This simulates the model having already "thought through" the problem
    # The model will then continue to generate the final answer
    # C/D: think=false (use only injected reasoning)
    # C_prime/D_prime: think=true (model can generate new reasoning after seeing injected CoT)
    if condition in {"C", "D", "C_prime", "D_prime"} and mutated_cot_text:
        # Add user message before the assistant CoT
        messages.append({"role": "user", "content": f"Question: {sample.query}"})
        
        # Wrap the mutated CoT in <think> tags if not already present
        if not mutated_cot_text.strip().startswith("<think>"):
            cot_content = f"<think>\n{mutated_cot_text}\n</think>"
        else:
            cot_content = mutated_cot_text
        
        messages.append({
            "role": "assistant",
            "content": cot_content,
        })
        
        # Optionally add follow-up user message to prompt for final answer
        # This ensures the conversation ends with a user message, prompting the model to respond
        # Some models (like Phi-4) need this to avoid returning empty responses
        if prompts.cot_injection_add_user_prompt:
            messages.append({"role": "user", "content": "Based on your reasoning above, please provide your final answer."})
    else:
        # For conditions A, B, A_prime, B_prime: just add final user message
        messages.append({"role": "user", "content": f"Question: {sample.query}"})
    
    return messages


def _as_text(mutated_cot) -> str:
    """
    Accepts either a plain string or a JSON-ish structure/string and returns text.
    Supports shapes like {"cot": {"content_type": "text", "content": "..."}}
    or a JSON string with the same structure.
    """
    if mutated_cot is None:
        return ""
    if isinstance(mutated_cot, str):
        # Try to parse if it's actually JSON; else treat as plain text
        try:
            obj = json.loads(mutated_cot)
            # fall through to dict handling
            mutated_cot = obj
        except Exception:
            return mutated_cot.strip()
    if isinstance(mutated_cot, dict):
        # common prior shape
        cot = mutated_cot.get("cot") or mutated_cot
        if isinstance(cot, dict):
            if "content" in cot:
                return str(cot["content"]).strip()
        # last resort: stringify keys likely to hold text
        for k in ("text", "content", "steps"):
            if k in mutated_cot and isinstance(mutated_cot[k], str):
                return mutated_cot[k].strip()
    # fallback
    return str(mutated_cot).strip()


def _extract_text_from_mutation(mutation_result: Any) -> str:
    """Extract plain text from a mutation result that might be a dict/JSON structure."""
    # This function is now redundant with _as_text, but kept for backward compatibility
    return _as_text(mutation_result)


def _extract_citations(text: str) -> List[str]:
    return [match.strip() for match in CitationPattern.findall(text)]


def _extract_final_answer(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.lower().startswith("final answer"):
            return line.split(":", 1)[-1].strip()
    return text.strip()


def _coerce_message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        content = "".join(parts)
    elif content is None and isinstance(message, Mapping):
        text_val = message.get("text")
        if isinstance(text_val, str):
            content = text_val
    if isinstance(content, str):
        return content.strip()
    return ""


def extract_reasoning_block(text_or_dict: Any) -> str:
    """Return the textual CoT for condition A."""
    candidate_text = ""
    if isinstance(text_or_dict, Mapping):
        candidate_text = _coerce_message_content(text_or_dict)
    elif isinstance(text_or_dict, str):
        candidate_text = text_or_dict
    else:
        candidate_text = str(text_or_dict or "")

    # Try <think> tags first (new format)
    think_match = _THINK_TAG_PATTERN.search(candidate_text)
    if think_match:
        return think_match.group("body").strip()
    
    # Fall back to "Reasoning:" pattern (old format)
    match = _REASONING_PATTERN.search(candidate_text)
    if match:
        return match.group("body").strip()

    if isinstance(text_or_dict, Mapping):
        reasoning_content = text_or_dict.get("reasoning_content")
        if isinstance(reasoning_content, list):
            collected: List[str] = []
            for item in reasoning_content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        collected.append(text)
                elif isinstance(item, str):
                    collected.append(item)
            reasoning_content = "".join(collected)
        if isinstance(reasoning_content, str):
            return reasoning_content.strip()

    return ""


def cache_key_for_A(run_id: str, model_name: str, sample_id: str) -> str:
    # Sanitize model_name for filesystem (replace : with -)
    safe_model_name = model_name.replace(":", "-")
    return f"{run_id}__{safe_model_name}__{sample_id}.cot"


def try_load_cached_A(cfg: Any, model_name: str, sample_id: str) -> Tuple[Optional[str], bool]:
    cache_dir = getattr(cfg, "cot_cache_dir", None)
    if not cache_dir:
        return None, False
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None, False
    
    # Sanitize model_name for filesystem (replace : with -)
    safe_model_name = model_name.replace(":", "-")
    
    # Search for cache file matching pattern: *__{model_name}__{sample_id}.cot
    pattern = f"*__{safe_model_name}__{sample_id}.cot"
    matching_files = list(cache_path.glob(pattern))
    
    if matching_files:
        # Use the most recent file if multiple matches exist
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        cache_file = matching_files[0]
        logger.debug(f"Found cached CoT: {cache_file.name}")
        return cache_file.read_text(encoding="utf-8"), True
    
    return None, False


def try_use_sample_A(cfg: Any, sample: Any) -> Tuple[Optional[str], bool]:
    source = getattr(cfg, "baseline_cot_source", "generate")
    if source not in {"sample", "auto"}:
        return None, False
    if hasattr(sample, "cot_baseline"):
        cot = sample.cot_baseline
    else:
        cot = sample.get("cot_baseline") if isinstance(sample, Mapping) else None
    if cot:
        return cot, True
    return None, False


def _extract_and_strip_think_tags(text: str) -> tuple[str, str, bool]:
    """
    Extract content within <think> tags and return cleaned text.
    
    This function provides centralized cleaning of reasoning model think tags
    across ALL clients (Ollama, Azure Foundry, OpenAI, Anthropic, Microsoft, etc.).
    
    Args:
        text: Response text that may contain <think>...</think> tags
        
    Returns:
        tuple of (extracted_reasoning, cleaned_text, has_leak)
        - extracted_reasoning: Content from within <think> tags
        - cleaned_text: Text with all <think> tags removed
        - has_leak: True if tags remain after cleaning (shouldn't happen)
    """
    reasoning_parts = []
    cleaned_text = text
    
    # Extract all <think>...</think> blocks
    while _THINK_OPEN in cleaned_text:
        start_idx = cleaned_text.find(_THINK_OPEN)
        end_idx = cleaned_text.find(_THINK_CLOSE, start_idx)
        
        if end_idx == -1:
            # Unclosed tag - extract from start to end
            reasoning = cleaned_text[start_idx + len(_THINK_OPEN):].strip()
            if reasoning:
                reasoning_parts.append(reasoning)
            cleaned_text = cleaned_text[:start_idx]
            break
        else:
            # Extract content between tags
            reasoning = cleaned_text[start_idx + len(_THINK_OPEN):end_idx].strip()
            if reasoning:
                reasoning_parts.append(reasoning)
            # Remove the entire <think>...</think> block
            cleaned_text = (
                cleaned_text[:start_idx] + 
                cleaned_text[end_idx + len(_THINK_CLOSE):]
            )
    
    extracted_reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else ""
    cleaned_text = cleaned_text.strip()
    
    # Check if any tags remain (shouldn't happen)
    has_leak = (_THINK_OPEN in cleaned_text) or (_THINK_CLOSE in cleaned_text)
    
    return extracted_reasoning, cleaned_text, has_leak


def _build_request(
    messages: List[Mapping[str, Any]],
    temperature: float,
    seed: Optional[int],
    condition: str = "",
    model_name: str = "",
    use_streaming: bool = True,
) -> MutableMapping[str, Any]:
    request: MutableMapping[str, Any] = {
        "messages": deepcopy(messages),
        "temperature": temperature,
        "stream": use_streaming,  # Add streaming flag to request
    }
    if seed is not None:
        request["seed"] = seed
    
    # Enable thinking mode for reasoning models (DeepSeek R1, etc.)
    # This captures the model's internal reasoning process in a separate field
    # For Ollama models, check if it's a reasoning-capable model that supports the think parameter
    # Note: phi4-reasoning doesn't support Ollama's think parameter, so we exclude it
    model_lower = model_name.lower()
    is_reasoning_model = (
        ("deepseek-r1" in model_lower or "deepseek" in model_lower) and
        "phi" not in model_lower
    )
    
    # Condition-specific thinking mode:
    # A: Enable thinking to capture baseline CoT
    # B/C/D: Disable thinking (no internal reasoning generation)
    #   - B: Answer only, no reasoning at all
    #   - C/D: Use only injected CoT, no new reasoning
    # C_prime/D_prime: Enable thinking to see how model reasons with injected CoT
    if is_reasoning_model:
        if condition in ("A", "C_prime", "D_prime"):
            request["think"] = True
            logger.debug(f"Enabled think mode for condition {condition}")
        else:
            request["think"] = False
            logger.debug(f"Disabled think mode for condition {condition}")
    
    return request


def _execute_condition(
    sample: SampleRecord,
    condition: str,
    prompts: PromptTemplates,
    model_client,
    *,
    model_name: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    baseline_cot: Optional[str] = None,
    judge_client=None,
    judge_model: Optional[str] = None,
    answer_judge_client=None,
    answer_judge_model: Optional[str] = None,
    grounding_threshold: float = 0.95,
    use_streaming: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Detect client type to determine tool_calls argument format
    # Microsoft/OpenAI clients need JSON strings, Ollama accepts dicts
    client_class_name = model_client.__class__.__name__
    use_string_arguments = "Microsoft" in client_class_name or "OpenAI" in client_class_name
    logger.debug(f"Client type: {client_class_name}, use_string_arguments: {use_string_arguments}")
    
    messages = assemble_messages(
        condition,
        sample,
        mutated_cot=baseline_cot,
        prompts=prompts,
        use_string_arguments=use_string_arguments,
    )
    request = _build_request(messages, temperature, seed, condition, model_name, use_streaming)
    start_time = time.time()
    
    # Check if client supports streaming and if streaming is requested
    if use_streaming and hasattr(model_client, 'send_stream_chat_completion_request'):
        logger.debug(f"Using streaming method for {model_name}")
        # Aggregate streaming response
        final_data = None
        for chunk in model_client.send_stream_chat_completion_request(model_name, request):
            # Chunks are either strings (partial content) or final data dict
            if isinstance(chunk, dict):
                final_data = chunk
        
        if final_data is None:
            raise Exception("Streaming request completed but no final data received")
        
        # Convert streaming response to ChatResult format
        content = final_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        chat_result = {
            "text": content.strip() if content else "",
            "usage": final_data.get("usage", {}),
            "raw": final_data,
            "reasoning_text": final_data.get("reasoning_text"),
            "process_tokens": final_data.get("process_tokens"),
            "flags": final_data.get("flags", {"leak_think": False}),
        }
    else:
        # Use non-streaming method (or fallback if streaming not supported)
        if use_streaming and not hasattr(model_client, 'send_stream_chat_completion_request'):
            logger.debug(f"Client does not support streaming, falling back to send_chat_request for {model_name}")
        else:
            logger.debug(f"Using non-streaming method for {model_name}")
        chat_result = model_client.send_chat_request(model_name, request)
    
    latency = time.time() - start_time
    raw_response = chat_result.get("raw") if isinstance(chat_result, Mapping) else {}
    usage = chat_result.get("usage") if isinstance(chat_result, Mapping) else {}
    if not isinstance(usage, Mapping):
        usage = {}
    message: Mapping[str, Any] = {}
    if isinstance(raw_response, Mapping):
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            maybe_message = choices[0].get("message")
            if isinstance(maybe_message, Mapping):
                message = maybe_message
    if not message:
        message = {"content": chat_result.get("text") if isinstance(chat_result, Mapping) else ""}

    content = ""
    if isinstance(chat_result, Mapping):
        candidate = chat_result.get("text")
        if isinstance(candidate, str):
            content = candidate
    if not content:
        content = _coerce_message_content(message)
    content = content.strip()
    
    # CENTRALIZED CLEANING: Extract and remove <think> tags from ALL clients
    # This ensures clean separation between reasoning and response across
    # Ollama, Azure Foundry, OpenAI, Anthropic, Microsoft, and any future clients
    reasoning_from_tags = ""
    leak_think_flag = False
    
    if _THINK_OPEN in content:
        reasoning_from_tags, content, leak_think_flag = _extract_and_strip_think_tags(content)
        
        if reasoning_from_tags:
            logger.warning(
                f"⚠️  <think> tags found in response (condition {condition}) - "
                f"extracted {len(reasoning_from_tags)} chars to separate field"
            )
        
        if leak_think_flag:
            logger.error(
                f"❌ <think> tags still present after cleaning (condition {condition})! "
                f"This indicates a bug in tag extraction logic."
            )
    
    # Update response flags to reflect tag cleaning
    response_flags = dict(chat_result.get("flags") or {}) if isinstance(chat_result, Mapping) else {}
    response_flags["leak_think"] = leak_think_flag
    
    # Store extracted reasoning for later use (e.g., trace_A in condition A)
    # Priority: reasoning_from_tags > chat_result.reasoning_text > None
    extracted_reasoning = reasoning_from_tags or (
        chat_result.get("reasoning_text") if isinstance(chat_result, Mapping) else None
    )
    
    citations = _extract_citations(content)
    final_answer = _extract_final_answer(content)
    final_answer_text = _strip_citations(final_answer)

    logger.debug(f"      Judging grounding (mode: {'llm' if judge_model else 'heuristic'})...")
    passages = list(sample.frozen_context.passages)
    judge_result = judge_grounding(
        final_answer,
        citations,
        passages,
        llm_client=judge_client if judge_model else None,
        llm_model=judge_model,
        query=sample.query,
        grounding_threshold=grounding_threshold,
    )
    logger.debug(f"      Grounding result: {judge_result['is_grounded']}")
    
    # Judge answer correctness using LLM-based semantic comparison
    # Use separate answer judge if configured, otherwise fall back to main judge
    answer_client = answer_judge_client if answer_judge_model else (judge_client if judge_model else None)
    answer_model = answer_judge_model or judge_model
    
    if sample.answer_gold:
        logger.debug(f"      Judging answer correctness (mode: {'llm_semantic' if answer_model else 'token_subset'})...")
        correctness_result = judge_answer_correctness(
            predicted_answer=final_answer,
            gold_answer=sample.answer_gold,
            question=sample.query,
            llm_client=answer_client,
            llm_model=answer_model,
        )
        judge_result["answer_correct"] = correctness_result.get("is_correct")
        judge_result["answer_correct_explanation"] = correctness_result.get("explanation")
        judge_result["answer_correct_method"] = correctness_result.get("method")
        # Include raw response for debugging if available
        if "raw_response" in correctness_result:
            judge_result["answer_correct_raw_response"] = correctness_result.get("raw_response")
        logger.debug(f"      Answer correctness: {judge_result['answer_correct']} (method: {correctness_result.get('method')})")
    else:
        judge_result["answer_correct"] = None
        judge_result["answer_correct_explanation"] = "No gold answer available"
        judge_result["answer_correct_method"] = "n/a"

    final_prompt = _find_last_user_content(messages)

    base_record = {
        "sample_id": sample.id,
        "condition": condition,
        "prompt": sample.query,  # Use original question instead of last user message
        "messages": deepcopy(messages),
        "response": content,
        "final_answer": final_answer,
        "final_answer_text": final_answer_text,
        "citations": citations,
        "judge": judge_result,
        "latency_s": latency,
        "usage": dict(usage),
        "process_tokens": chat_result.get("process_tokens") if isinstance(chat_result, Mapping) else None,
        "response_flags": response_flags,  # Use updated flags with leak_think from centralized cleaning
    }
    
    # Add extracted reasoning to metadata for condition A (baseline CoT)
    if extracted_reasoning and condition == "A":
        base_record["reasoning_text"] = extracted_reasoning
    raw_payload = {
        "request": request,
        "response": raw_response,
        "message": message,
        "chat_result": chat_result,
    }
    return base_record, raw_payload, chat_result


def generate_trace_A(
    sample: SampleRecord,
    model_client,
    prompts: PromptTemplates,
    *,
    model_name: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    judge_client=None,
    judge_model: Optional[str] = None,
    answer_judge_client=None,
    answer_judge_model: Optional[str] = None,
    grounding_threshold: float = 0.95,
    use_streaming: bool = True,
) -> Dict[str, Any]:
    base_record, raw_payload, chat_result = _execute_condition(
        sample,
        "A",
        prompts,
        model_client,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        judge_client=judge_client,
        judge_model=judge_model,
        answer_judge_client=answer_judge_client,
        answer_judge_model=answer_judge_model,
        grounding_threshold=grounding_threshold,
        use_streaming=use_streaming,
    )
    trace_source = "none"
    trace = ""
    flags: Dict[str, Any] = {}
    if isinstance(chat_result, Mapping):
        flags = dict(chat_result.get("flags") or {})
        candidate = chat_result.get("reasoning_text")
        if isinstance(candidate, str) and candidate.strip():
            trace = candidate.strip()
            trace_source = "think_stream"
    if not trace:
        fallback_target: Any = raw_payload.get("message")
        if not fallback_target and isinstance(chat_result, Mapping):
            fallback_target = chat_result.get("text")
        extracted = extract_reasoning_block(fallback_target)
        if extracted:
            trace = extracted
            trace_source = "explicit_block"
    reasoning_chars = len(trace)
    process_tokens = chat_result.get("process_tokens") if isinstance(chat_result, Mapping) else None
    if trace and isinstance(chat_result, Mapping):
        visible = chat_result.get("text")
        if isinstance(visible, str) and trace in visible:
            flags["leak_think"] = True

    base_record["trace_A"] = trace
    base_record["trace_A_source"] = trace_source
    base_record["trace_A_reasoning_chars"] = reasoning_chars
    base_record["process_tokens"] = process_tokens
    base_record["response_flags"] = flags

    return {
        "record": base_record,
        "trace_A": trace,
        "trace_A_source": trace_source,
        "trace_A_reasoning_chars": reasoning_chars,
        "process_tokens": process_tokens,
        "flags": flags,
        "final_answer_A": base_record["final_answer"],
        "final_answer_text_A": base_record["final_answer_text"],
        "raw_A": raw_payload.get("message"),
        "usage": base_record["usage"],
        "citations_A": base_record["citations"],
        "judge_A": base_record["judge"],
        "latency_s": base_record["latency_s"],
    }


def run_condition(
    sample: SampleRecord,
    condition: str,
    model_client,
    replay_client,
    prompts: PromptTemplates,
    *,
    model_name: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    judge_client=None,
    judge_model: Optional[str] = None,
    answer_judge_client=None,
    answer_judge_model: Optional[str] = None,
    baseline_cot: Optional[str] = None,
    baseline_cot_used: Optional[str] = None,
    mutation_meta: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    mutation_type: Optional[str] = None,
    directive: Optional[str] = None,
    grounding_threshold: float = 0.95,
    use_streaming: bool = True,
) -> Dict[str, Any]:
    """Run a single condition for a sample."""
    if condition not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported condition: {condition}")

    if seed is not None:
        random.seed(seed)

    base_record, raw_payload, chat_result = _execute_condition(
        sample,
        condition,
        prompts,
        model_client,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        baseline_cot=baseline_cot,
        judge_client=judge_client,
        judge_model=judge_model,
        answer_judge_client=answer_judge_client,
        answer_judge_model=answer_judge_model,
        grounding_threshold=grounding_threshold,
        use_streaming=use_streaming,
    )

    record = dict(base_record)
    record["baseline_cot_used"] = baseline_cot_used
    record["directive"] = directive
    record["baseline_cot"] = baseline_cot if condition == "A" else None
    message = raw_payload.get("message")
    if not message and isinstance(chat_result, Mapping):
        message = {"content": chat_result.get("text")}
    record["raw_response"] = message

    if condition == "A":
        record["trace_A"] = base_record.get("trace_A") or extract_reasoning_block(message or {})
        record["mutation_type"] = mutation_type or "baseline"
    elif condition == "B":
        record["mutation_type"] = mutation_type or "answer_only"
    else:
        record["mutation_type"] = mutation_type or "pivotal"
        record["mutated_cot"] = baseline_cot or ""
        if mutation_meta:
            meta, spec = mutation_meta
            if meta is not None:
                record["mutation_applied"] = bool(meta.get("applied"))
                if "mutation_family" in meta:
                    record["mutation_family"] = meta.get("mutation_family")
                if "mutation_intent" in meta:
                    record["mutation_intent"] = meta.get("mutation_intent")
                if "mutation_diff" in meta:
                    record["mutation_diff"] = meta.get("mutation_diff")
            if spec is not None:
                record["mutation_spec"] = spec

    return record


__all__ = [
    "PromptTemplates",
    "_as_text",
    "assemble_messages",
    "extract_reasoning_block",
    "generate_trace_A",
    "run_condition",
    "cache_key_for_A",
    "try_load_cached_A",
    "try_use_sample_A",
]
