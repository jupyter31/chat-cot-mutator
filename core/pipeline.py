"""Core pipeline helpers for headless batch runs."""
from __future__ import annotations

import random
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from core.schema import FrozenContextRecord, FrozenPassageRecord, SampleRecord
from eval.judges import judge_grounding
from mutations.registry import mutate


CitationPattern = re.compile(r"\[\[([^\]]+)\]\]")
_CITE_BRACKETS = re.compile(r"\s*\[\[[^\]]+\]\]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")

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
        steps = mutated_cot or sample.cot_baseline or ""
        if condition == "D" and sample.grounding_rule:
            steps = f"{steps}\n\nGROUNDING RULE: {sample.grounding_rule}".strip()
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
    mutation_override: Optional[str] = None,
    judge_client=None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single condition for a sample."""
    if condition not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported condition: {condition}")

    if seed is not None:
        random.seed(seed)

    baseline_cot = sample.cot_baseline or ""
    directive = mutation_override if mutation_override is not None else sample.mutation_directive
    mutated_cot = None
    mutation_type = None
    if condition in {"C", "D"}:
        mutated_cot = mutate(
            baseline_cot,
            directive,
            model_client=model_client,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )
        if directive is None:
            mutation_type = "none"
        elif re.search(r"paraphrase|reorder", directive, re.IGNORECASE):
            mutation_type = "control"
        else:
            mutation_type = "pivotal"
    elif condition == "A":
        mutation_type = "baseline"
    else:
        mutation_type = "answer_only"

    prompt = assemble_prompt(condition, sample, mutated_cot=mutated_cot, prompts=prompts)

    request: MutableMapping[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a careful assistant who cites evidence accurately."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed

    start_time = time.time()
    response = model_client.send_chat_request(model_name, request)
    latency = time.time() - start_time
    message = response.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "").strip()
    citations = _extract_citations(content)
    final_answer = _extract_final_answer(content)
    final_answer_text = _strip_citations(final_answer)

    passages = list(sample.frozen_context.passages)
    judge_result = judge_grounding(
        final_answer,
        citations,
        passages,
        llm_client=judge_client if judge_model else None,
        llm_model=judge_model,
        query=sample.query,
    )
    if sample.answer_gold:
        judge_result["answer_correct"] = _answers_match(final_answer, sample.answer_gold)
    else:
        judge_result["answer_correct"] = None

    usage = response.get("usage", {})

    return {
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
        "mutation_type": mutation_type,
        "mutated_cot": mutated_cot,
        "baseline_cot": baseline_cot,
        "directive": directive,
    }


__all__ = ["PromptTemplates", "assemble_prompt", "run_condition"]
