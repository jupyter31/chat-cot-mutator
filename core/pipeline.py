"""Core pipeline helpers for headless batch runs."""
from __future__ import annotations

import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

from core.schema import FrozenContextRecord, FrozenPassageRecord, SampleRecord
from eval.judges import judge_grounding


CitationPattern = re.compile(r"\[\[([^\]]+)\]\]")
_CITE_BRACKETS = re.compile(r"\s*\[\[[^\]]+\]\]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")
_REASONING_PATTERN = re.compile(
    r"Reasoning:\s*(?P<body>.+?)\s*Final Answer:", re.IGNORECASE | re.DOTALL
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

    def __getitem__(self, condition: str) -> str:
        return self.condition_to_template[condition]


def _format_passage_label(passage: FrozenPassageRecord, idx: int) -> str:
    if passage.cite:
        return passage.cite
    if passage.doc_id:
        return passage.doc_id
    return f"passage_{idx+1}"


def _format_evidence(context: FrozenContextRecord) -> str:
    lines: List[str] = []
    for idx, passage in enumerate(context.passages):
        label = _format_passage_label(passage, idx)
        lines.append(f"[{label}] {passage.text}")
    if context.tool_outputs:
        lines.append("\nFROZEN TOOL OUTPUTS:")
        for tool_idx, tool in enumerate(context.tool_outputs):
            input_repr = tool.input if isinstance(tool.input, str) else str(tool.input)
            output_repr = tool.output if isinstance(tool.output, str) else str(tool.output)
            lines.append(f"- {tool.tool} | input={input_repr} | output={output_repr}")
    return "\n".join(lines).strip()


def _extract_text_from_mutation(mutation_result: Any) -> str:
    """Extract plain text from a mutation result that might be a dict/JSON structure."""
    if mutation_result is None:
        return ""
    
    # If it's already a string, return it
    if isinstance(mutation_result, str):
        return mutation_result
    
    # If it's a dict, try to extract the content
    if isinstance(mutation_result, dict):
        # Check for nested cot structure
        if "cot" in mutation_result and isinstance(mutation_result["cot"], dict):
            return mutation_result["cot"].get("content", "")
        # Check for direct content field
        if "content" in mutation_result:
            content = mutation_result["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, dict) and "content" in content:
                return content["content"]
        # Check for text field
        if "text" in mutation_result:
            return mutation_result["text"]
    
    # If it's a list, concatenate text parts
    if isinstance(mutation_result, list):
        parts = []
        for item in mutation_result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(item["content"])
        return "".join(parts)
    
    # Fallback to string representation
    return str(mutation_result)

def assemble_prompt(
    condition: str,
    sample: SampleRecord,
    mutated_cot: Optional[str] = None,
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """Assemble the prompt for the given condition."""
    if prompts is None:
        raise ValueError("Prompt templates must be provided")
    template = prompts[condition]
    evidence = _format_evidence(sample.frozen_context)
    format_args: Dict[str, Any] = {
        "query": sample.query,
        "evidence": evidence,
    }
    if "{cot}" in template:
        # Ensure mutated_cot is plain text
        steps = _extract_text_from_mutation(mutated_cot) if mutated_cot else ""
        format_args["cot"] = steps
    prompt_text = template.format(**format_args)
    return prompt_text.strip()


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


def _build_request(prompt: str, temperature: float, seed: Optional[int]) -> MutableMapping[str, Any]:
    request: MutableMapping[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": "You are a careful assistant who cites evidence accurately.",
            },
            {"role": "user", "content": prompt},
        ],
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = assemble_prompt(condition, sample, mutated_cot=baseline_cot, prompts=prompts)
    request = _build_request(prompt, temperature, seed)
    start_time = time.time()
    response = model_client.send_chat_request(model_name, request)
    latency = time.time() - start_time
    message = response.get("choices", [{}])[0].get("message", {}) if isinstance(response, Mapping) else {}
    content = _coerce_message_content(message)
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
    )
    logger.debug(f"      Grounding result: {judge_result['is_grounded']}")
    if sample.answer_gold:
        judge_result["answer_correct"] = _answers_match(final_answer, sample.answer_gold)
    else:
        judge_result["answer_correct"] = None

    usage = response.get("usage", {}) if isinstance(response, Mapping) else {}

    base_record = {
        "sample_id": sample.id,
        "condition": condition,
        "prompt": prompt,
        "response": content,
        "final_answer": final_answer,
        "final_answer_text": final_answer_text,
        "citations": citations,
        "judge": judge_result,
        "latency_s": latency,
        "usage": usage,
    }
    raw_payload = {
        "request": request,
        "response": response,
        "message": message,
    }
    return base_record, raw_payload


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
) -> Dict[str, Any]:
    base_record, raw_payload = _execute_condition(
        sample,
        "A",
        prompts,
        model_client,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        judge_client=judge_client,
        judge_model=judge_model,
    )
    trace = extract_reasoning_block(raw_payload.get("message", {}))
    return {
        "record": base_record,
        "trace_A": trace,
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
) -> Dict[str, Any]:
    """Run a single condition for a sample."""
    if condition not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported condition: {condition}")

    if seed is not None:
        random.seed(seed)

    base_record, raw_payload = _execute_condition(
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
    )

    record = dict(base_record)
    record["baseline_cot_used"] = baseline_cot_used
    record["directive"] = directive
    record["baseline_cot"] = baseline_cot if condition == "A" else None
    record["raw_response"] = raw_payload.get("message")

    if condition == "A":
        record["trace_A"] = extract_reasoning_block(raw_payload.get("message", {}))
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
    "assemble_prompt",
    "extract_reasoning_block",
    "generate_trace_A",
    "run_condition",
    "cache_key_for_A",
    "try_load_cached_A",
    "try_use_sample_A",
]
