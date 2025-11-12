"""Two-stage groundedness judge: claim extraction + claim scoring."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.schema import FrozenPassageRecord


def _response_text(response: Mapping[str, Any]) -> str:
    text = response.get("text") if isinstance(response, Mapping) else ""
    if isinstance(text, str) and text.strip():
        return text.strip()
    raw = response.get("raw") if isinstance(response, Mapping) else None
    if isinstance(raw, Mapping):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def _normalize_text(s: str) -> str:
    s = re.sub(r"\s*\[\[[^\]]+\]\]\s*", " ", s or "")
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def _content_words(s: str) -> set[str]:
    toks = [t for t in _normalize_text(s).split() if len(t) > 2]
    return set(toks)


def _load_prompt_messages(prompt_name: str) -> List[Dict[str, str]]:
    """Load prompt messages from JSONL file."""
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts" / "judge"
    prompt_path = prompt_dir / f"{prompt_name}.jsonl"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    messages = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def _format_search_results(passages: List[FrozenPassageRecord]) -> str:
    """Format passages as search results for the judge."""
    results = []
    for idx, passage in enumerate(passages):
        label = passage.cite or passage.doc_id or f"passage_{idx+1}"
        results.append(f"[{idx+1}] {label}: {passage.text}")
    return "\n".join(results)


def _extract_claims(
    query: str,
    answer: str,
    llm_client,
    model_name: str,
) -> List[Dict[str, Any]]:
    """Stage 1: Break answer into claims using claimbreak prompt."""
    if llm_client is None or model_name is None:
        return []
    
    messages = _load_prompt_messages("claimbreak")
    
    # Replace placeholders in the user message
    formatted_messages = []
    for msg in messages:
        content = msg["content"]
        content = content.replace("{{user_query}}", query)
        content = content.replace("{{assistant_reply}}", answer)
        formatted_messages.append({"role": msg["role"], "content": content})
    
    request = {
        "messages": formatted_messages,
    }
    
    try:
        response = llm_client.send_chat_request(model_name, request)
        content = _response_text(response)

        # Parse JSON claims from response
        claims = json.loads(content)
        return claims if isinstance(claims, list) else []
    except (json.JSONDecodeError, KeyError, Exception) as e:
        # If claim extraction fails, return empty list
        return []


def _score_claims(
    claims: List[Dict[str, Any]],
    passages: List[FrozenPassageRecord],
    llm_client,
    model_name: str,
) -> Dict[str, Any]:
    """Stage 2: Score each claim against search results using score_all prompt."""
    if not claims or llm_client is None or model_name is None:
        return {"scores": [], "average_score": 0.0}
    
    messages = _load_prompt_messages("score_all")
    
    # Format search results and claims
    search_results = _format_search_results(passages)
    claims_text = "\n".join([
        f"Claim {i+1}: {claim.get('Claim', '')}" 
        for i, claim in enumerate(claims)
    ])
    
    # Replace placeholders in the user message
    formatted_messages = []
    for msg in messages:
        content = msg["content"]
        content = content.replace("{{search_results}}", search_results)
        content = content.replace("{{claims}}", claims_text)
        formatted_messages.append({"role": msg["role"], "content": content})
    
    request = {
        "messages": formatted_messages,
    }
    
    try:
        response = llm_client.send_chat_request(model_name, request)
        content = _response_text(response)
        
        scores = []
        
        # Try to extract scores from "Final Scores:" section (multi-claim format)
        if "Final Scores:" in content:
            scores_section = content.split("Final Scores:")[-1].strip()
            for line in scores_section.split("\n"):
                if ":" in line:
                    try:
                        score_str = line.split(":")[-1].strip()
                        score = float(score_str)
                        scores.append(score)
                    except ValueError:
                        continue
        
        # If no scores found, try parsing JSON objects for "Final score" field (single-claim format)
        # Example: '{"Claim 1": "...", "Final score": 1}'
        if not scores:
            # Try to find JSON objects with "Final score" field
            json_pattern = r'\{[^}]*"Final score"\s*:\s*([0-9.]+)[^}]*\}'
            matches = re.finditer(json_pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                except ValueError:
                    continue
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "scores": scores,
            "average_score": avg_score,
            "raw_response": content,
        }
    except Exception as e:
        return {"scores": [], "average_score": 0.0, "error": str(e)}


def judge_grounding(
    answer: str,
    citations: List[str],
    passages: List[FrozenPassageRecord],
    *,
    llm_client: Any = None,
    llm_model: Optional[str] = None,
    query: Optional[str] = None,
    grounding_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Two-stage groundedness judge:
    1. Extract claims from the answer using claimbreak prompt
    2. Score each claim against passages using score_all prompt
    
    Falls back to simple heuristic if LLM client not provided.
    """
    
    # If no LLM client, use improved heuristic with citation validation and overlap
    if llm_client is None or llm_model is None:
        labels = []
        label_to_text = {}
        for idx, p in enumerate(passages):
            label = p.cite or p.doc_id or f"passage_{idx+1}"
            labels.append(label)
            label_to_text[label] = p.text
        valid = [c for c in citations if c in labels]
        if not valid:
            return {
                "is_grounded": False,
                "method": "heuristic_fallback",
                "citations_provided": citations,
                "num_passages": len(passages),
            }
        ans_words = _content_words(answer)
        max_overlap = 0.0
        for c in valid:
            pw = _content_words(label_to_text[c])
            denom = max(1, len(ans_words))
            overlap = len(ans_words & pw) / denom
            if overlap > max_overlap:
                max_overlap = overlap
        return {
            "is_grounded": max_overlap >= grounding_threshold,
            "method": "heuristic_fallback",
            "citations_provided": citations,
            "num_passages": len(passages),
            "overlap_max": max_overlap,
        }
    
    # Stage 1: Extract claims from answer
    claims = _extract_claims(
        query=query or "Unknown query",
        answer=answer,
        llm_client=llm_client,
        model_name=llm_model,
    )
    
    # Stage 2: Score claims against passages
    scoring_result = _score_claims(
        claims=claims,
        passages=passages,
        llm_client=llm_client,
        model_name=llm_model,
    )
    
    # Determine if grounded based on average score
    # Threshold: configurable via grounding_threshold parameter
    avg_score = scoring_result.get("average_score", 0.0)
    is_grounded = avg_score >= grounding_threshold
    
    result = {
        "is_grounded": is_grounded,
        "method": "two_stage_llm",
        "num_claims": len(claims),
        "claims": claims,
        "claim_scores": scoring_result.get("scores", []),
        "average_score": avg_score,
        "citations_provided": citations,
        "num_passages": len(passages),
    }
    
    # Include raw scoring response if available for debugging
    if "raw_response" in scoring_result:
        result["scoring_raw_response"] = scoring_result["raw_response"]
    if "error" in scoring_result:
        result["scoring_error"] = scoring_result["error"]
    
    return result


def judge_answer_correctness(
    predicted_answer: str,
    gold_answer: str,
    question: str,
    *,
    llm_client: Any = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Judge if two answers are semantically equivalent using an LLM.
    
    Args:
        predicted_answer: The answer generated by the model
        gold_answer: The reference/correct answer
        question: The question being answered
        llm_client: The LLM client to use for judging
        llm_model: The model name to use for judging
    
    Returns:
        Dict with keys:
        - is_correct: bool, whether answers are semantically equivalent
        - explanation: str, explanation of the decision
        - method: str, always "llm_semantic"
        - raw_response: str, raw LLM response (for debugging)
    """
    if not llm_client or not llm_model:
        # Fallback to simple token-based matching if no LLM available
        from core.pipeline import _answers_match
        is_correct = _answers_match(predicted_answer, gold_answer)
        return {
            "is_correct": is_correct,
            "explanation": "Fallback to token-based matching (no LLM available)",
            "method": "token_subset",
        }
    
    # Load the answer_correctness prompt
    messages = _load_prompt_messages("answer_correctness")
    
    # Fill in the template variables
    for msg in messages:
        content = msg.get("content", "")
        content = content.replace("{{question}}", question)
        content = content.replace("{{predicted_answer}}", predicted_answer)
        content = content.replace("{{gold_answer}}", gold_answer)
        msg["content"] = content
    
    # Send request to LLM
    # Note: Reasoning models (o1, o3, deepseek-r1) don't support temperature or max_completion_tokens
    request = {
        "model": llm_model,
        "messages": messages,
    }
    
    # Only add temperature and token limits for non-reasoning models
    # Reasoning models (o1*, o3*, deepseek-r1*, r1-*) handle these internally
    model_lower = llm_model.lower()
    is_reasoning_model = any(pattern in model_lower for pattern in [
        'o1', 'o3', 'deepseek-r1', 'r1-', '-r1', 'reasoning'
    ])
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[ANSWER_JUDGE] Model: {llm_model}, is_reasoning: {is_reasoning_model}")
    
    if not is_reasoning_model:
        request["temperature"] = 0.0  # Use deterministic evaluation
        request["max_completion_tokens"] = 500  # For newer OpenAI models
        logger.info(f"[ANSWER_JUDGE] Added temperature=0.0 and max_completion_tokens=500")
    else:
        logger.info(f"[ANSWER_JUDGE] Skipped temperature (reasoning model)")
    
    logger.info(f"[ANSWER_JUDGE] Request keys: {list(request.keys())}")
    
    try:
        response = llm_client.send_chat_request(llm_model, request)
        raw_text = _response_text(response)
        
        # Try to parse JSON response
        # Look for JSON object in the response
        json_match = re.search(r'\{[^{}]*"semantically_equivalent"[^{}]*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result_data = json.loads(json_str)
            is_correct = result_data.get("semantically_equivalent", False)
            explanation = result_data.get("explanation", "")
        else:
            # Fallback: look for YES/NO or true/false in response
            raw_lower = raw_text.lower()
            if "semantically_equivalent\": true" in raw_lower or "\"yes\"" in raw_lower:
                is_correct = True
                explanation = "Parsed from non-JSON response"
            elif "semantically_equivalent\": false" in raw_lower or "\"no\"" in raw_lower:
                is_correct = False
                explanation = "Parsed from non-JSON response"
            else:
                # Can't parse, default to False
                is_correct = False
                explanation = f"Could not parse response: {raw_text[:100]}"
        
        return {
            "is_correct": is_correct,
            "explanation": explanation,
            "method": "llm_semantic",
            "raw_response": raw_text,
        }
    
    except Exception as e:
        # On error, fallback to token-based matching
        from core.pipeline import _answers_match
        is_correct = _answers_match(predicted_answer, gold_answer)
        return {
            "is_correct": is_correct,
            "explanation": f"LLM judge failed ({str(e)}), used token-based fallback",
            "method": "token_subset_fallback",
            "error": str(e),
        }


__all__ = ["judge_grounding", "judge_answer_correctness"]
