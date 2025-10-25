"""LLM-backed registry for chain-of-thought mutations."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParsedDirective:
    """Structured representation of a mutation directive."""

    kind: str
    params: Mapping[str, str]


@dataclass(frozen=True)
class MutationMeta:
    applied: bool
    mutation_family: Optional[str]
    mutation_intent: Optional[str]
    mutation_diff: Optional[str]


_DIRECTIVE_PATTERN = re.compile(r"(?P<name>[A-Za-z_:-]+)\s*(?:\((?P<args>.*)\))?$")

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mutations"


def _normalize_kind(raw: str) -> str:
    return re.sub(r"[^a-z]", "", raw.lower())


def _parse_entity_swap_args(raw_args: str | None) -> Mapping[str, str]:
    if not raw_args:
        return {"old": "", "new": ""}
    arrow_match = re.search(r"([^->=]+?)->([^=]+)$", raw_args)
    if arrow_match:
        left = arrow_match.group(1)
        right = arrow_match.group(2)
    else:
        parts = [p.strip() for p in raw_args.split(",") if p.strip()]
        if len(parts) >= 2:
            left = parts[0].split("=")[-1]
            right = parts[1].split("=")[-1]
        else:
            left = ""
            right = ""
    return {"old": left.strip(), "new": right.strip()}


def _parse_directive(directive: str | None) -> Optional[ParsedDirective]:
    if not directive:
        return None
    match = _DIRECTIVE_PATTERN.match(directive.strip())
    if not match:
        return None
    name = match.group("name") or ""
    args = match.group("args") or ""
    normalized = _normalize_kind(name)
    if normalized.startswith("entityswap"):
        return ParsedDirective("entity_swap", _parse_entity_swap_args(args))
    if normalized.startswith("saliencedrop"):
        return ParsedDirective("salience_drop", {})
    if normalized.startswith("claimaligneddeletion"):
        return ParsedDirective("claim_aligned_deletion", {})
    if normalized.startswith("topicdilution"):
        return ParsedDirective("topic_dilution", {})
    if normalized.startswith("paraphrase") or normalized.startswith("neutralcontrolparaphrase"):
        return ParsedDirective("paraphrase", {})
    if normalized.startswith("reorder") or normalized.startswith("neutralcontrolreorder"):
        return ParsedDirective("reorder", {})
    return None


def _format_tool_results(cot: str) -> str:
    payload = [
        {
            "reference_id": "cot",
            "results": [
                {
                    "content_type": "text",
                    "content": cot.strip(),
                }
            ],
        }
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _inject_directive_hints(content: str, parsed: ParsedDirective) -> str:
    hints: Sequence[str] = []
    if parsed.kind == "entity_swap":
        old = parsed.params.get("old", "").strip() or "(unspecified)"
        new = parsed.params.get("new", "").strip()
        if not new:
            raise ValueError("Entity swap directives require both source and target entities.")
        hints = [f"Swap every mention of '{old}' with '{new}'."]
    elif parsed.kind == "paraphrase":
        hints = ["Produce a paraphrased version with identical meaning and answer."]
    elif parsed.kind == "reorder":
        hints = ["Return the same reasoning steps in a different logical order."]

    if not hints:
        return content

    hint_block = "\n\n## Mutation directive\n" + "\n".join(hints)
    if "## Tool results" in content:
        return content.replace("## Tool results", f"## Tool results{hint_block}")
    return content + hint_block


def _load_messages(parsed: ParsedDirective) -> Sequence[Mapping[str, str]]:
    filename = None
    replacements: Mapping[str, str] = {}
    if parsed.kind == "salience_drop":
        filename = "salience_drop.jsonl"
    elif parsed.kind == "claim_aligned_deletion":
        filename = "claim_aligned_deletion.jsonl"
        replacements = {"{{number}}": "5"}
    elif parsed.kind == "topic_dilution":
        filename = "topic_dilution.jsonl"
    elif parsed.kind == "entity_swap":
        filename = "entity_swap.jsonl"
        replacements = {
            "{{written_entity_types}}": "names",
            "{{entity_plural}}": "entity",
            "{{number}}": "1",
        }
    else:
        filename = "free_form.jsonl"
        replacements = {"{{mutation_request}}": parsed.kind.replace("_", " ")}

    prompt_path = PROMPT_DIR / filename
    lines = prompt_path.read_text(encoding="utf-8").strip().splitlines()
    messages = []
    for line in lines:
        record = json.loads(line)
        content = record.get("content", "")
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        messages.append({"role": record.get("role", "user"), "content": content})
    return messages


def _compute_diff(original: str, mutated: str) -> str:
    diff_lines = difflib.unified_diff(
        original.splitlines(),
        mutated.splitlines(),
        fromfile="original",
        tofile="mutated",
        lineterm="",
    )
    return "\n".join(diff_lines)


def mutate(
    directive: str,
    cot_text: str,
    *,
    model_client=None,
    model_name: Optional[str] = None,
    temperature: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply a mutation directive to a chain-of-thought text."""

    parsed = _parse_directive(directive)
    spec = {
        "directive": directive,
        "parsed_kind": parsed.kind if parsed else None,
    }

    if not directive or not cot_text.strip() or parsed is None:
        meta = {
            "applied": False,
            "mutation_family": parsed.kind if parsed else None,
            "mutation_intent": directive,
            "mutation_diff": None,
        }
        return cot_text, meta, spec

    if model_client is None or model_name is None:
        meta = {
            "applied": False,
            "mutation_family": parsed.kind,
            "mutation_intent": directive,
            "mutation_diff": None,
        }
        return cot_text, meta, spec

    messages = _load_messages(parsed)

    transformed_messages = []
    for message in messages:
        content = message.get("content", "")
        content = content.replace("{{tool_results}}", _format_tool_results(cot_text))
        content = _inject_directive_hints(content, parsed)
        if "{{" in content:
            content = re.sub(r"\{\{[^}]+\}\}", "", content)
        if "Chain-of-thought:" not in content:
            content = f"{content}\n\nChain-of-thought:\n{cot_text.strip()}".strip()
        transformed_messages.append({"role": message.get("role", "user"), "content": content})

    request = {
        "messages": transformed_messages,
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed

    response = model_client.send_chat_request(model_name, request)
    message = response.get("choices", [{}])[0].get("message", {})
    
    # Extract the actual text content from the response
    content = message.get("content", "")
    
    # If content is a dict/JSON structure, extract the text
    if isinstance(content, dict):
        if "content" in content:
            content = content["content"]
        elif "cot" in content and isinstance(content["cot"], dict):
            content = content["cot"].get("content", "")
    elif isinstance(content, list):
        # Handle list of content parts
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        content = "".join(text_parts)
    
    # Ensure we return a string, not a dict
    if not isinstance(content, str):
        content = str(content)
    
    meta = {
        "applied": bool(content and content != cot_text),
        "mutation_family": "llm_directed",
        "mutation_intent": directive,
    }
    
    spec = {
        "directive": directive,
        "temperature": temperature,
        "seed": seed,
        "model": model_name,
    }
    
    return (meta, spec), content  # Return the string content, not a dict


__all__ = ["mutate", "MutationMeta", "ParsedDirective"]
