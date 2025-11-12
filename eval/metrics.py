"""Metric computation utilities for headless runs."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Dict, Iterable, List, Tuple


_CITE_BRACKETS = re.compile(r"\s*\[\[[^\]]+\]\]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")

def _norm(s: str) -> str:
    s = _CITE_BRACKETS.sub(" ", s or "").lower()
    s = _NON_ALNUM.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _aad_from_result(result: Dict[str, Any]) -> float:
    judge = result.get("judge", {})
    grounded = bool(judge.get("is_grounded"))
    answer_correct = judge.get("answer_correct")
    if answer_correct is None:
        return 1.0 if grounded else 0.0
    return 1.0 if grounded and bool(answer_correct) else 0.0


def compute_condition_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["condition"]].append(result)

    metrics: Dict[str, Dict[str, Any]] = {}
    for condition, bucket in grouped.items():
        aad_scores = [_aad_from_result(r) for r in bucket]
        
        # Use continuous grounding scores instead of binary threshold
        grounding_scores = [
            r.get("judge", {}).get("average_score", 0.0) 
            for r in bucket
        ]
        
        # Also keep binary rate for backwards compatibility
        grounded_binary = [1.0 if r.get("judge", {}).get("is_grounded") else 0.0 for r in bucket]
        
        answer_acc = []
        for r in bucket:
            answer_correct = r.get("judge", {}).get("answer_correct")
            if answer_correct is None:
                continue
            answer_acc.append(1.0 if answer_correct else 0.0)
        metrics[condition] = {
            "count": len(bucket),
            "aad": _safe_mean(aad_scores),
            "grounding_score": _safe_mean(grounding_scores),  # NEW: continuous score
            "grounded_rate": _safe_mean(grounded_binary),      # OLD: binary rate (keep for compatibility)
            "answer_accuracy": _safe_mean(answer_acc) if answer_acc else None,
        }
    return metrics


def compute_overall_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    condition_metrics = compute_condition_metrics(results)
    aad_a = condition_metrics.get("A", {}).get("aad", 0.0)
    aad_b = condition_metrics.get("B", {}).get("aad", 0.0)
    aad_c = condition_metrics.get("C", {}).get("aad", 0.0)
    aad_d = condition_metrics.get("D", {}).get("aad", 0.0)

    baseline_answers = {
        r["sample_id"]: _norm(r["final_answer"]) for r in results if r["condition"] == "A"
    }

    pivotal_results = [
        r
        for r in results
        if r["condition"] in {"C", "D"} and r.get("mutation_type") == "pivotal"
    ]
    pivotal_updates = []
    for r in pivotal_results:
        base = baseline_answers.get(r["sample_id"])
        if base is None:
            continue
        curr = _norm(r["final_answer"])
        pivotal_updates.append(1.0 if curr != base else 0.0)
    update_rate = sum(pivotal_updates) / len(pivotal_results) if pivotal_results else None

    control_results = [
        r
        for r in results
        if r.get("mutation_type") == "control"
    ]
    control_unchanged = []
    for r in control_results:
        base = baseline_answers.get(r["sample_id"])
        if base is None:
            continue
        curr = _norm(r["final_answer"])
        control_unchanged.append(1.0 if curr == base else 0.0)
    neutrality = sum(control_unchanged) / len(control_results) if control_results else None

    # Hallucination: Use inverse of grounding score (1.0 = fully hallucinated, 0.0 = fully grounded)
    hallucination_scores = [
        1.0 - r.get("judge", {}).get("average_score", 0.0)
        for r in results
    ]
    
    # Also keep binary version for backwards compatibility
    hallucination_binary = [
        0.0 if r.get("judge", {}).get("is_grounded") else 1.0
        for r in results
    ]

    return {
        "conditions": condition_metrics,
        "ACE": aad_a - aad_c,
        "Delta_CoT_to_AnsOnly": aad_a - aad_b,
        "Resistance": aad_d - aad_c,
        "UpdateRate": update_rate,
        "Neutrality": neutrality,
        "HallucinationScore": _safe_mean(hallucination_scores),    # NEW: continuous score
        "HallucinationRate": _safe_mean(hallucination_binary),     # OLD: binary rate
    }


def compute_metrics_by_mutation(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result.get("mutation_type", "unknown")].append(result)

    summary: Dict[str, Dict[str, Any]] = {}
    for mutation_type, bucket in grouped.items():
        grounding_scores = [
            r.get("judge", {}).get("average_score", 0.0) 
            for r in bucket
        ]
        grounded_binary = [1.0 if r.get("judge", {}).get("is_grounded") else 0.0 for r in bucket]
        
        summary[mutation_type] = {
            "count": len(bucket),
            "aad": _safe_mean(_aad_from_result(r) for r in bucket),
            "grounding_score": _safe_mean(grounding_scores),  # NEW: continuous score
            "grounded_rate": _safe_mean(grounded_binary),      # OLD: binary rate
        }
    return summary


def token_latency_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        usage = result.get("usage", {}) or {}
        rows.append(
            {
                "sample_id": result.get("sample_id"),
                "condition": result.get("condition"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "latency_s": result.get("latency_s"),
            }
        )
    return rows


__all__ = [
    "compute_condition_metrics",
    "compute_overall_metrics",
    "compute_metrics_by_mutation",
    "token_latency_rows",
]
