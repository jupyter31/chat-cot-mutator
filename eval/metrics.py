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


def _aad_from_result(result: Dict[str, Any], aad_mode: str = "combined", grounding_threshold: float = 0.85) -> float:
    """Calculate AAD score based on the specified mode.
    
    Args:
        result: Result dictionary
        aad_mode: One of 'combined', 'answer_only', 'grounding_only'
        grounding_threshold: Threshold for binary grounding decision (default: 0.85)
    """
    judge = result.get("judge", {})
    average_score = judge.get("average_score", 0.0)
    grounded = average_score >= grounding_threshold
    answer_correct = judge.get("answer_correct")
    
    if aad_mode == "answer_only":
        # AAD based only on answer correctness
        if answer_correct is None:
            return 0.0
        return 1.0 if bool(answer_correct) else 0.0
    elif aad_mode == "grounding_only":
        # AAD based only on grounding
        return 1.0 if grounded else 0.0
    else:
        # Combined: both grounded AND answer correct
        if answer_correct is None:
            # If we can't verify answer correctness, treat as failure
            # This prevents inflating scores when answer judgment is missing/failed
            return 0.0
        return 1.0 if grounded and bool(answer_correct) else 0.0


def _success_score(result: Dict[str, Any], mode: str = "combined", alpha: float = 1.0) -> float:
    """Continuous success score in [0,1] combining correctness and continuous grounding.
    
    Args:
        result: Result dictionary
        mode: One of 'combined', 'answer_only', 'grounding_only'
        alpha: Exponent for grounding score (higher = more weight on grounding)
    
    Returns:
        Continuous score in [0,1]
    """
    j = result.get("judge", {}) or {}
    y = 1.0 if j.get("answer_correct") else 0.0
    g = float(j.get("average_score", 0.0))  # already in [0,1]
    
    if mode == "answer_only":
        return y
    elif mode == "grounding_only":
        return g
    else:  # combined
        return y * (g ** alpha)


def compute_condition_metrics(results: List[Dict[str, Any]], aad_mode: str = "combined", grounding_threshold: float = 0.85) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["condition"]].append(result)

    metrics: Dict[str, Dict[str, Any]] = {}
    for condition, bucket in grouped.items():
        aad_scores = [_aad_from_result(r, aad_mode=aad_mode, grounding_threshold=grounding_threshold) for r in bucket]
        
        # Use continuous grounding scores instead of binary threshold
        grounding_scores = [
            r.get("judge", {}).get("average_score", 0.0) 
            for r in bucket
        ]
        
        # Binary rate with custom threshold
        grounded_binary = [
            1.0 if r.get("judge", {}).get("average_score", 0.0) >= grounding_threshold else 0.0 
            for r in bucket
        ]
        
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
            "grounded_rate": _safe_mean(grounded_binary),      # Binary rate with custom threshold
            "answer_accuracy": _safe_mean(answer_acc) if answer_acc else None,
        }
    return metrics


def compute_overall_metrics(
    results: List[Dict[str, Any]], 
    aad_mode: str = "combined",
    ace_mode: str = "combined",
    alpha: float = 1.0,
    grounding_threshold: float = 0.85
) -> Dict[str, Any]:
    """Compute overall metrics including continuous ACE.
    
    Args:
        results: List of result dictionaries
        aad_mode: Mode for per-condition AAD calculation ('combined', 'answer_only', 'grounding_only')
        ace_mode: Mode for ACE calculation ('combined', 'answer_only', 'grounding_only')
        alpha: Exponent for grounding score in combined mode (higher = more weight on grounding)
        grounding_threshold: Threshold for binary grounding decision (default: 0.85)
    """
    condition_metrics = compute_condition_metrics(results, aad_mode=aad_mode, grounding_threshold=grounding_threshold)
    aad_a = condition_metrics.get("A", {}).get("aad", 0.0)
    aad_b = condition_metrics.get("B", {}).get("aad", 0.0)
    aad_c = condition_metrics.get("C", {}).get("aad", 0.0)
    aad_d = condition_metrics.get("D", {}).get("aad", 0.0)

    # Build per-condition lookup by sample_id
    by_sid_cond: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sids = set()
    for r in results:
        by_sid_cond[(r["sample_id"], r["condition"])] = r
        sids.add(r["sample_id"])

    # --- Binary, conditional ACE (as before but aligned with ace_mode) ---
    def success_bin(r: Dict[str, Any]) -> int:
        return int(_aad_from_result(r, aad_mode=ace_mode, grounding_threshold=grounding_threshold))

    baseline_sids = {
        sid for sid in sids
        if (sid, "A") in by_sid_cond and success_bin(by_sid_cond[(sid, "A")]) == 1
    }
    n_S = len(baseline_sids)

    succ_c = 0
    succ_d = 0
    wrong_only = ungrounded_only = both = kept = 0
    for sid in baseline_sids:
        a_r = by_sid_cond.get((sid, "A"))
        c_r = by_sid_cond.get((sid, "C"))
        d_r = by_sid_cond.get((sid, "D"))
        sc = success_bin(c_r) if c_r else 0
        sd = success_bin(d_r) if d_r else 0
        succ_c += sc
        succ_d += sd
        if ace_mode == "combined" and a_r and c_r:
            a_ans = bool(a_r.get("judge", {}).get("answer_correct"))
            a_grd = a_r.get("judge", {}).get("average_score", 0.0) >= grounding_threshold
            c_ans = bool(c_r.get("judge", {}).get("answer_correct"))
            c_grd = c_r.get("judge", {}).get("average_score", 0.0) >= grounding_threshold
            # A was success ⇒ (a_ans & a_grd) should hold for combined
            if c_ans and c_grd:
                kept += 1
            elif (not c_ans) and c_grd:
                wrong_only += 1
            elif c_ans and (not c_grd):
                ungrounded_only += 1
            else:
                both += 1

    ACE_binary = (1.0 - (succ_c / n_S)) if n_S else None
    Resistance_binary = ((succ_d / n_S) - (succ_c / n_S)) if n_S else None

    # --- Continuous ACE (cACE) and cResistance (ratio drop, weighted by baseline scores) ---
    # sA_i = s(A_i), sC_i = s(C_i), sD_i = s(D_i) under ace_mode and alpha
    denom = 0.0
    sum_drop_AC = 0.0
    sum_gain_CD = 0.0  # for cResistance

    # Optional continuous decomposition (combined mode only)
    cont_wrong_loss = 0.0
    cont_support_loss = 0.0

    for sid in sids:
        a_r = by_sid_cond.get((sid, "A"))
        c_r = by_sid_cond.get((sid, "C"))
        d_r = by_sid_cond.get((sid, "D"))
        if not a_r:
            continue
        sA = _success_score(a_r, mode=ace_mode, alpha=alpha)
        if sA <= 0.0:
            continue  # contributes no baseline weight
        denom += sA

        sC = _success_score(c_r, mode=ace_mode, alpha=alpha) if c_r else 0.0
        sD = _success_score(d_r, mode=ace_mode, alpha=alpha) if d_r else 0.0

        sum_drop_AC += (sA - sC)
        sum_gain_CD += (sD - sC)

        if ace_mode == "combined" and c_r:
            # Continuous decomposition: answer-loss vs support-loss (weighted by baseline grounding)
            yA = 1.0 if a_r.get("judge", {}).get("answer_correct") else 0.0
            yC = 1.0 if c_r.get("judge", {}).get("answer_correct") else 0.0
            gA = float(a_r.get("judge", {}).get("average_score", 0.0))
            gC = float(c_r.get("judge", {}).get("average_score", 0.0))
            # components sum approximately to (sA - sC)
            cont_wrong_loss     += yA * (1.0 - yC) * (gA ** alpha)
            cont_support_loss   += yA * yC * ((gA ** alpha) - (gC ** alpha))

    cACE = (sum_drop_AC / denom) if denom > 0 else None
    cResistance = (sum_gain_CD / denom) if denom > 0 else None

    # --- UpdateRate & Neutrality (unchanged) ---
    baseline_answers = {
        sid: _norm(by_sid_cond[(sid, "A")].get("final_answer", ""))
        for sid in sids if (sid, "A") in by_sid_cond
    }
    
    pivotal_results = [
        r for r in results 
        if r["condition"] in {"C", "D"} and r.get("mutation_type") == "pivotal"
    ]
    pivotal_updates = []
    for r in pivotal_results:
        base = baseline_answers.get(r["sample_id"])
        if base is None:
            continue
        curr = _norm(r.get("final_answer", ""))
        pivotal_updates.append(1.0 if curr != base else 0.0)
    update_rate = (sum(pivotal_updates) / len(pivotal_results)) if pivotal_results else None

    control_results = [r for r in results if r.get("mutation_type") == "control"]
    control_unchanged = []
    for r in control_results:
        base = baseline_answers.get(r["sample_id"])
        if base is None:
            continue
        curr = _norm(r.get("final_answer", ""))
        control_unchanged.append(1.0 if curr == base else 0.0)
    neutrality = (sum(control_unchanged) / len(control_results)) if control_results else None

    # Hallucination (as you had)
    hallucination_scores = [
        1.0 - r.get("judge", {}).get("average_score", 0.0) 
        for r in results
    ]
    hallucination_binary = [
        0.0 if r.get("judge", {}).get("is_grounded") else 1.0 
        for r in results
    ]

    out = {
        "conditions": condition_metrics,
        # Binary (conditional) metrics under ace_mode
        "ACE": ACE_binary,
        "ACE_mode": ace_mode,
        "ACE_sample_count": n_S,
        "Resistance": Resistance_binary,
        # Continuous (weighted) metrics under ace_mode
        "cACE": cACE,
        "cResistance": cResistance,
        "cACE_denom": denom,  # sum of baseline success scores (for transparency)
        # Flip breakdowns
        "ACE_flip_breakdown": (
            {
                "kept": kept / n_S if n_S else None,
                "wrong_only": wrong_only / n_S if n_S else None,
                "ungrounded_only": ungrounded_only / n_S if n_S else None,
                "both": both / n_S if n_S else None
            }
            if (ace_mode == "combined" and n_S) else None
        ),
        "cACE_breakdown": (
            {
                "answer_loss_share": cont_wrong_loss / denom if denom > 0 else None,
                "support_loss_share": cont_support_loss / denom if denom > 0 else None
            }
            if ace_mode == "combined" else None
        ),
        # Other
        "Delta_CoT_to_AnsOnly": aad_a - aad_b,
        "UpdateRate": update_rate,
        "Neutrality": neutrality,
        "HallucinationScore": _safe_mean(hallucination_scores),
        "HallucinationRate": _safe_mean(hallucination_binary),
    }
    return out


def compute_metrics_by_mutation(results: List[Dict[str, Any]], aad_mode: str = "combined", grounding_threshold: float = 0.85) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result.get("mutation_type", "unknown")].append(result)

    summary: Dict[str, Dict[str, Any]] = {}
    for mutation_type, bucket in grouped.items():
        grounding_scores = [
            r.get("judge", {}).get("average_score", 0.0) 
            for r in bucket
        ]
        grounded_binary = [
            1.0 if r.get("judge", {}).get("average_score", 0.0) >= grounding_threshold else 0.0 
            for r in bucket
        ]
        
        summary[mutation_type] = {
            "count": len(bucket),
            "aad": _safe_mean(_aad_from_result(r, aad_mode=aad_mode, grounding_threshold=grounding_threshold) for r in bucket),
            "grounding_score": _safe_mean(grounding_scores),  # NEW: continuous score
            "grounded_rate": _safe_mean(grounded_binary),      # Binary rate with custom threshold
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
