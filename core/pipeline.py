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
from eval.judges import judge_grounding


CitationPattern = re.compile(r"\[\[([^\]]+)\]\]")
_CITE_BRACKETS = re.compile(r"\s*\[\[[^\]]+\]\]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")
_REASONING_PATTERN = re.compile(
    r"Reasoning:\s*(?P<body>.+?)\s*Final Answer:", re.IGNORECASE | re.DOTALL
)

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
) -> List[Dict[str, Any]]:
    """Assemble chat messages for the given condition."""

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

    if tool_variant not in {"direct", "function_call"}:
        raise ValueError(f"Unsupported tool_variant: {tool_variant}")

    for idx, entry in enumerate(evidence_entries):
        payload = json.dumps(entry, ensure_ascii=False)
        needs_tool_call = tool_variant == "function_call" or evidence_channel == "tool"
        call_id = f"tool_call_{idx}"

        if needs_tool_call:
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
                                "arguments": payload,
                            },
                        }
                    ],
                }
            )

        tool_message: Dict[str, Any] = {
            "role": evidence_channel,
            "name": "evidence_passage",
            "content": payload,
        }

        if needs_tool_call:
            tool_message["tool_call_id"] = call_id
            if evidence_channel == "tool":
                tool_message["role"] = "tool"

        messages.append(tool_message)

    # Add final user message repeating just the question
    messages.append({"role": "user", "content": f"Question: {sample.query}"})
    
    # For conditions C and D, inject mutated CoT as assistant message
    # This simulates the model having already "thought through" the problem
    # The model will then continue to generate the final answer
    if condition in {"C", "D"} and mutated_cot_text:
        # Wrap the mutated CoT in <think> tags if not already present
        if not mutated_cot_text.strip().startswith("<think>"):
            cot_content = f"<think>\n{mutated_cot_text}\n</think>"
        else:
            cot_content = mutated_cot_text
        
        messages.append({
            "role": "assistant",
            "content": cot_content,
        })
    
    return messages
    
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
    return f"{run_id}__{model_name}__{sample_id}.cot"


def try_load_cached_A(cfg: Any, model_name: str, sample_id: str) -> Tuple[Optional[str], bool]:
    cache_dir = getattr(cfg, "cot_cache_dir", None)
    run_id = getattr(cfg, "run_id", None)
    if not cache_dir or not run_id:
        return None, False
    cache_path = Path(cache_dir) / cache_key_for_A(run_id, model_name, sample_id)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), True
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


def _build_request(
    messages: List[Mapping[str, Any]],
    temperature: float,
    seed: Optional[int],
) -> MutableMapping[str, Any]:
    request: MutableMapping[str, Any] = {
        "messages": deepcopy(messages),
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed
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
    grounding_threshold: float = 0.95,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    messages = assemble_messages(
        condition,
        sample,
        mutated_cot=baseline_cot,
        prompts=prompts,
    )
    request = _build_request(messages, temperature, seed)
    start_time = time.time()
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
    if sample.answer_gold:
        judge_result["answer_correct"] = _answers_match(final_answer, sample.answer_gold)
    else:
        judge_result["answer_correct"] = None

    final_prompt = _find_last_user_content(messages)

    base_record = {
        "sample_id": sample.id,
        "condition": condition,
        "prompt": final_prompt,
        "messages": deepcopy(messages),
        "response": content,
        "final_answer": final_answer,
        "final_answer_text": final_answer_text,
        "citations": citations,
        "judge": judge_result,
        "latency_s": latency,
        "usage": dict(usage),
        "process_tokens": chat_result.get("process_tokens") if isinstance(chat_result, Mapping) else None,
        "response_flags": dict(chat_result.get("flags") or {}) if isinstance(chat_result, Mapping) else {},
    }
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
    grounding_threshold: float = 0.95,
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
        grounding_threshold=grounding_threshold,
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
    baseline_cot: Optional[str] = None,
    baseline_cot_used: Optional[str] = None,
    mutation_meta: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    mutation_type: Optional[str] = None,
    directive: Optional[str] = None,
    grounding_threshold: float = 0.95,
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
        grounding_threshold=grounding_threshold,
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
