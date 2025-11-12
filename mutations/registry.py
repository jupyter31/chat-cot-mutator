"""LLM-backed registry for chain-of-thought mutations."""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


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
COT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "cot_mutations"


def _normalize_kind(raw: str) -> str:
    return re.sub(r"[^a-z]", "", raw.lower())


def _parse_entity_swap_args(raw_args: str | None) -> Mapping[str, str]:
    if not raw_args:
        return {"old": "", "new": ""}
    
    # Remove common prefixes like "name=", "old=", etc.
    cleaned = re.sub(r'^(name|old|entity)=', '', raw_args.strip())
    
    # Try arrow format: "Rembrandt->Vermeer" or "old=Rembrandt->new=Vermeer"
    arrow_match = re.search(r'([^->=]+?)->(.+)$', cleaned)
    if arrow_match:
        left = arrow_match.group(1).strip()
        right = arrow_match.group(2).strip()
        # Remove "new=" prefix if present
        right = re.sub(r'^new=', '', right)
        return {"old": left, "new": right}
    
    # Try comma-separated format: "old=X, new=Y"
    parts = [p.strip() for p in raw_args.split(",") if p.strip()]
    if len(parts) >= 2:
        left = parts[0].split("=")[-1].strip()
        right = parts[1].split("=")[-1].strip()
        return {"old": left, "new": right}
    
    # Fallback: empty values
    return {"old": "", "new": ""}


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
        old = parsed.params.get("old", "").strip()
        new = parsed.params.get("new", "").strip()
        if old and new:
            hints = [f"Swap every mention of '{old}' with '{new}'."]
        elif new:
            hints = [f"Replace the main entity with '{new}'."]
        else:
            # Generic entity swap without specific targets
            hints = ["Swap the main entity mentioned in the reasoning with a different but related entity."]
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


def _load_messages(parsed: ParsedDirective, use_cot_prompts: bool = False) -> Sequence[Mapping[str, str]]:
    """Load mutation prompt messages.
    
    Args:
        parsed: Parsed mutation directive
        use_cot_prompts: If True, use CoT-specific prompts instead of tool-based prompts
    """
    filename = None
    replacements: Mapping[str, str] = {}
    base_dir = COT_PROMPT_DIR if use_cot_prompts else PROMPT_DIR
    
    if use_cot_prompts:
        # Map to CoT-specific mutations
        kind_map = {
            "entity_swap": "entity_swap.jsonl",
            "entityswap": "entity_swap.jsonl",
            "date_number_jitter": "date_number_jitter.jsonl",
            "datenumberjitter": "date_number_jitter.jsonl",
            "step_shuffle": "step_shuffle.jsonl",
            "stepshuffle": "step_shuffle.jsonl",
            "passage_shuffle": "step_shuffle.jsonl",  # Similar to step shuffle
            "conclusion_negation": "conclusion_negation.jsonl",
            "conclusionnegation": "conclusion_negation.jsonl",
            "evidence_fabrication": "evidence_fabrication.jsonl",
            "evidencefabrication": "evidence_fabrication.jsonl",
            "paraphrase": "paraphrase.jsonl",
            "control": "paraphrase.jsonl",  # Paraphrase is the control mutation
        }
        filename = kind_map.get(parsed.kind.lower())
        if filename is None:
            # Default to paraphrase for unknown mutations
            filename = "paraphrase.jsonl"
    else:
        # Original tool-based mutations
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

    prompt_path = base_dir / filename
    if not prompt_path.exists():
        # Fallback to tool-based if CoT prompt doesn't exist
        if use_cot_prompts:
            prompt_path = PROMPT_DIR / "free_form.jsonl"
            replacements = {"{{mutation_request}}": f"Mutate the reasoning: {parsed.kind.replace('_', ' ')}"}
    
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
    """Compute a simple diff summary between original and mutated text."""
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=False),
        mutated.splitlines(keepends=False),
        fromfile="original",
        tofile="mutated",
        lineterm="",
        n=1  # Minimal context
    )
    return "\n".join(list(diff_lines)[:20])  # Limit to first 20 lines for meta


def _extract_text_from_response(content: Any) -> str:
    """Extract plain text from various response formats."""
    if content is None:
        return ""
    
    # If already a string, check if it's JSON
    if isinstance(content, str):
        # Try to parse as JSON if it looks like JSON
        content = content.strip()
        if content.startswith("{") or content.startswith("["):
            try:
                parsed = json.loads(content)
                return _extract_text_from_response(parsed)
            except json.JSONDecodeError:
                # Not valid JSON, might be a truncated response
                # Try to extract what we can
                if "{'cot':" in content:
                    # Handle malformed Python dict representation
                    return ""  # Return empty string for malformed content
                return content
        return content
    
    # If it's a dict, extract content
    if isinstance(content, dict):
        # Handle nested cot structure {"cot": {"content_type": "text", "content": "..."}}
        if "cot" in content:
            cot = content["cot"]
            if isinstance(cot, dict):
                if "content" in cot:
                    return str(cot["content"]).strip()
                elif cot == {"content_type": "text"}:
                    # Incomplete response - no actual content
                    return ""
            elif isinstance(cot, str):
                return cot.strip()
        
        # Direct content field
        if "content" in content:
            return _extract_text_from_response(content["content"])
        
        # Other common fields
        for field in ["text", "result", "output", "response"]:
            if field in content:
                return _extract_text_from_response(content[field])
    
    # If it's a list, concatenate text parts
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for field in ["text", "content"]:
                    if field in item:
                        parts.append(str(item[field]))
                        break
        return " ".join(parts).strip()
    
    # Fallback to string conversion
    return str(content).strip()


def mutate(
    directive: str,
    cot_text: str,
    *,
    model_client=None,
    model_name: Optional[str] = None,
    temperature: float = 0.5,
    seed: Optional[int] = None,
    question: Optional[str] = None,
) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], str]:
    """Apply a mutation directive to a chain-of-thought text.
    
    Args:
        directive: Mutation directive string (e.g., "EntitySwap(Rembrandt->Vermeer)")
        cot_text: Plain string containing the chain-of-thought to mutate
        model_client: LLM client for generating mutations
        model_name: Name of the model to use
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        question: Optional question context for the CoT
    
    Returns:
        Tuple of ((meta, spec), mutated_text) where:
        - meta: Dictionary with mutation metadata
        - spec: Dictionary with mutation specification
        - mutated_text: Plain string of the mutated chain-of-thought
    """
    parsed = _parse_directive(directive)
    
    # Build spec
    spec = {
        "directive": directive,
        "parsed_kind": parsed.kind if parsed else None,
        "temperature": temperature,
        "seed": seed,
        "model": model_name,
    }
    
    # Handle edge cases - return original text unchanged
    if not directive or not cot_text.strip() or parsed is None:
        meta = {
            "applied": False,
            "mutation_family": parsed.kind if parsed else None,
            "mutation_intent": directive,
            "mutation_diff": None,
        }
        return (meta, spec), cot_text
    
    # No model client - can't apply mutation
    if model_client is None or model_name is None:
        meta = {
            "applied": False,
            "mutation_family": parsed.kind,
            "mutation_intent": directive,
            "mutation_diff": None,
        }
        return (meta, spec), cot_text
    
    # Prepare messages for LLM - use CoT-specific prompts
    messages = _load_messages(parsed, use_cot_prompts=True)
    transformed_messages = []
    for message in messages:
        content = message.get("content", "")
        # Replace CoT-specific placeholders
        content = content.replace("{{cot_text}}", cot_text.strip())
        content = content.replace("{{question}}", question or "")
        # Legacy placeholder for tool-based mutations (fallback)
        content = content.replace("{{tool_results}}", _format_tool_results(cot_text))
        content = _inject_directive_hints(content, parsed)
        # Clean up any remaining placeholders
        if "{{" in content:
            content = re.sub(r"\{\{[^}]+\}\}", "", content)
        transformed_messages.append({"role": message.get("role", "user"), "content": content})
    
    # Build request
    request = {
        "messages": transformed_messages,
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed
    
    # Send to LLM
    try:
        response = model_client.send_chat_request(model_name, request)
    except Exception as e:
        logger.error(f"Mutation LLM call failed: {e}")
        meta = {
            "applied": False,
            "mutation_family": parsed.kind,
            "mutation_intent": directive,
            "mutation_diff": None,
            "error": str(e),
        }
        return (meta, spec), cot_text
    
    content = ""
    if isinstance(response, Mapping):
        text = response.get("text")
        if isinstance(text, str):
            content = text
        else:
            raw = response.get("raw")
            if isinstance(raw, Mapping):
                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message")
                    if isinstance(message, Mapping):
                        maybe_content = message.get("content")
                        if isinstance(maybe_content, str):
                            content = maybe_content

    # Extract plain text from the response
    try:
        mutated_text = _extract_text_from_response(content)
    except Exception as e:
        logger.error(f"Failed to extract text from mutation response: {e}")
        meta = {
            "applied": False,
            "mutation_family": parsed.kind,
            "mutation_intent": directive,
            "mutation_diff": None,
            "error": f"Text extraction failed: {e}",
        }
        return (meta, spec), cot_text
    
    # If extraction failed or returned empty/malformed content, log and use original
    if not mutated_text or mutated_text == "{'cot': {'content_type': 'text'}}":
        logger.warning(f"Mutation returned empty or malformed content. Directive: {directive}")
        logger.warning(f"Raw response (first 200 chars): {content[:200] if content else '(empty)'}")
        mutated_text = cot_text  # Fall back to original
        applied = False
    else:
        applied = bool(mutated_text.strip() != cot_text.strip())
    
    # Ensure we have a string
    if not isinstance(mutated_text, str):
        mutated_text = str(mutated_text)
    
    # Build metadata
    applied = bool(mutated_text and mutated_text.strip() != cot_text.strip())
    
    # Log if mutation didn't actually change anything (potential issue)
    if not applied and parsed.kind not in ["paraphrase", "reorder"]:
        # For non-control mutations, unchanged text might indicate a problem
        logger.warning(f"Mutation '{directive}' did not change the text. This may indicate a problem.")
        logger.debug(f"Original text (first 100 chars): {cot_text[:100]}")
        logger.debug(f"Mutated text (first 100 chars): {mutated_text[:100]}")
    
    # Compute diff summary if mutation was applied
    diff_summary = None
    if applied and parsed and parsed.kind == "entity_swap":
        old_entity = parsed.params.get("old", "")
        new_entity = parsed.params.get("new", "")
        if old_entity and new_entity:
            diff_summary = f"{old_entity} → {new_entity}"
    elif applied:
        # For other mutations, compute a text diff (limited for meta)
        diff_summary = _compute_diff(cot_text, mutated_text)
    
    meta = {
        "applied": applied,
        "mutation_family": parsed.kind if parsed else "unknown",
        "mutation_intent": directive,
        "mutation_diff": diff_summary,
    }
    
    # Add pivot info if applicable
    if parsed and parsed.kind in ["entity_swap", "salience_drop", "claim_aligned_deletion"]:
        meta["pivot_info"] = {
            "type": parsed.kind,
            "params": dict(parsed.params) if parsed.params else {},
            "should_change_answer": True,
        }
    elif parsed and parsed.kind in ["paraphrase", "reorder"]:
        meta["pivot_info"] = {
            "type": parsed.kind,
            "params": {},
            "should_change_answer": False,
        }
    
    return (meta, spec), mutated_text


__all__ = ["mutate", "MutationMeta", "ParsedDirective"]
