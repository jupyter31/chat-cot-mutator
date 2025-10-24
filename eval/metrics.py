"""Metric computation utilities for headless runs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


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
        grounded = [1.0 if r.get("judge", {}).get("is_grounded") else 0.0 for r in bucket]
        answer_acc = []
        for r in bucket:
            answer_correct = r.get("judge", {}).get("answer_correct")
            if answer_correct is None:
                continue
            answer_acc.append(1.0 if answer_correct else 0.0)
        metrics[condition] = {
            "count": len(bucket),
            "aad": _safe_mean(aad_scores),
            "grounded_rate": _safe_mean(grounded),
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
        r["sample_id"]: r["final_answer"]
        for r in results
        if r["condition"] == "A"
    }

    pivotal_results = [
        r
        for r in results
        if r["condition"] in {"C", "D"} and r.get("mutation_type") == "pivotal"
    ]
    pivotal_updates = [
        1.0
        for r in pivotal_results
        if baseline_answers.get(r["sample_id"]) is not None
        and r["final_answer"] != baseline_answers[r["sample_id"]]
    ]
    update_rate = sum(pivotal_updates) / len(pivotal_results) if pivotal_results else None

    control_results = [
        r
        for r in results
        if r.get("mutation_type") == "control"
    ]
    control_unchanged = [
        1.0
        for r in control_results
        if baseline_answers.get(r["sample_id"]) is not None
        and r["final_answer"] == baseline_answers[r["sample_id"]]
    ]
    neutrality = sum(control_unchanged) / len(control_results) if control_results else None

    hallucination = [
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
        "HallucinationRate": _safe_mean(hallucination),
    }


def compute_metrics_by_mutation(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result.get("mutation_type", "unknown")].append(result)

    summary: Dict[str, Dict[str, Any]] = {}
    for mutation_type, bucket in grouped.items():
        summary[mutation_type] = {
            "count": len(bucket),
            "aad": _safe_mean(_aad_from_result(r) for r in bucket),
            "grounded_rate": _safe_mean(1.0 if r.get("judge", {}).get("is_grounded") else 0.0 for r in bucket),
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
