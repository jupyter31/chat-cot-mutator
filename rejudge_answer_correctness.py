"""Re-judge answer correctness for all samples in an experiment using LLM-based semantic matching."""
import json
from pathlib import Path
import sys
import argparse
from typing import Dict, Any

from eval.judges import judge_answer_correctness
from clients.client_factory import create_llm_client


def rejudge_answer_correctness(
    results_dir: Path,
    judge_model: str,
    input_file: Path = None,
    force: bool = False,
    aad_mode: str = "combined"
):
    """Re-judge all samples with LLM-based answer correctness evaluation.
    
    Args:
        results_dir: Directory containing samples.jsonl
        judge_model: Model spec for answer judging (e.g., 'microsoft_internal:dev-gpt-o3-mini')
        input_file: Optional path to input data with gold answers
        force: If True, re-judge all samples even if already judged by LLM
        aad_mode: AAD calculation mode for metrics ('combined', 'answer_only', 'grounding_only')
    """
    samples_file = results_dir / "samples.jsonl"
    if not samples_file.exists():
        print(f"❌ Error: {samples_file} not found")
        return
    
    # Parse judge model spec
    print(f"🔧 Creating judge client for {judge_model}...")
    if ":" in judge_model:
        provider, model_name = judge_model.split(":", 1)
    else:
        provider = "openai"
        model_name = judge_model
    
    # Create judge client
    judge_client = create_llm_client(provider, endpoint=None)
    print(f"✓ Judge client ready: {model_name}")
    
    # Load all samples
    samples = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"📊 Loaded {len(samples)} samples from {samples_file}")
    
    # Load gold answers from input file if provided
    gold_answers = {}
    if input_file and input_file.exists():
        print(f"📖 Loading gold answers from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sample_id = record.get("id", "")
                    answer_gold = record.get("answer_gold", "")
                    if sample_id and answer_gold:
                        gold_answers[sample_id] = answer_gold
        print(f"✓ Loaded {len(gold_answers)} gold answers")
    
    # Analyze existing judgments
    needs_rejudge = []
    for sample in samples:
        judge_data = sample.get("judge", {})
        method = judge_data.get("answer_correct_method", "unknown")
        
        # Check if needs re-judging
        if force or method in ["token_subset", "token_subset_fallback", "unknown"]:
            needs_rejudge.append(sample)
    
    print(f"🔍 Found {len(needs_rejudge)} samples to re-judge (force={force})")
    if not needs_rejudge:
        print("✓ All samples already have LLM-based answer correctness. Use --force to re-judge anyway.")
        return
    
    # Re-judge each sample
    updated_count = 0
    error_count = 0
    
    for idx, sample in enumerate(needs_rejudge, 1):
        sample_id = sample.get("sample_id", "unknown")
        condition = sample.get("condition", "?")
        
        judge_data = sample.get("judge", {})
        old_method = judge_data.get("answer_correct_method", "unknown")
        
        # Get required fields
        predicted_answer = sample.get("final_answer_text") or sample.get("final_answer", "")
        gold_answer = gold_answers.get(sample_id) or judge_data.get("gold_answer", "")
        question = sample.get("prompt", "")
        
        # Clean up question
        if question.startswith("Question: "):
            question = question[len("Question: "):]
        
        if not predicted_answer:
            print(f"  ⚠️  [{idx}/{len(needs_rejudge)}] {sample_id} ({condition}): Missing predicted answer, skipping")
            error_count += 1
            continue
        
        if not gold_answer:
            print(f"  ⚠️  [{idx}/{len(needs_rejudge)}] {sample_id} ({condition}): Missing gold answer, skipping")
            error_count += 1
            continue
        
        # Re-judge with LLM
        try:
            result = judge_answer_correctness(
                predicted_answer=predicted_answer,
                gold_answer=gold_answer,
                question=question,
                llm_client=judge_client,
                llm_model=model_name,
            )
            
            # Update judge data
            old_correct = judge_data.get("answer_correct")
            judge_data["answer_correct"] = result["is_correct"]
            judge_data["answer_correct_explanation"] = result["explanation"]
            judge_data["answer_correct_method"] = result["method"]
            if "raw_response" in result:
                judge_data["answer_correct_raw_response"] = result["raw_response"]
            
            sample["judge"] = judge_data
            updated_count += 1
            
            # Show status
            status = "✓" if result["is_correct"] else "✗"
            changed = " (CHANGED)" if old_correct != result["is_correct"] else ""
            print(f"  {status} [{idx}/{len(needs_rejudge)}] {sample_id} ({condition}): {old_method} -> {result['method']}{changed}")
            
        except Exception as e:
            print(f"  ❌ [{idx}/{len(needs_rejudge)}] {sample_id} ({condition}): Error: {e}")
            error_count += 1
            continue
    
    # Save updated samples
    if updated_count > 0:
        backup_file = samples_file.with_suffix(".jsonl.backup")
        print(f"\n💾 Backing up original to {backup_file}...")
        import shutil
        shutil.copy2(samples_file, backup_file)
        
        with open(samples_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"✓ Saved {len(samples)} samples with {updated_count} re-judged")
        
        # Recompute metrics
        print(f"\n📊 Recomputing metrics with aad_mode='{aad_mode}'...")
        from eval.metrics import compute_overall_metrics, compute_condition_metrics
        
        overall = compute_overall_metrics(samples, aad_mode=aad_mode)
        condition_metrics = compute_condition_metrics(samples, aad_mode=aad_mode)
        
        overall_file = results_dir / "metrics_overall.json"
        with open(overall_file, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)
        print(f"✓ Updated {overall_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Updated Metrics Summary:")
        print("="*60)
        for condition, metrics in condition_metrics.items():
            print(f"\nCondition {condition}:")
            print(f"  Count: {metrics['count']}")
            print(f"  AAD: {metrics.get('aad', 0):.3f}")
            answer_acc = metrics.get('answer_accuracy')
            if answer_acc is not None:
                print(f"  Answer Accuracy: {answer_acc:.3f}")
        print("="*60)
        
        if error_count > 0:
            print(f"\n⚠️  {error_count} samples had errors during re-judging")
    else:
        print("\n✓ No samples were updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-judge answer correctness for all samples using LLM-based semantic matching"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory (e.g., results/exp_ollama_deepseek_r1_32b_hotpot_1k)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="microsoft_internal:dev-gpt-o3-mini",
        help="Judge model spec (default: microsoft_internal:dev-gpt-o3-mini)"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to input data file with gold answers (e.g., data/hotpot_distractor_1k.jsonl)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge all samples even if already judged by LLM"
    )
    parser.add_argument(
        "--aad-mode",
        choices=["combined", "answer_only", "grounding_only"],
        default="combined",
        help="AAD calculation mode for metrics (default: combined)"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Error: {args.results_dir} does not exist")
        sys.exit(1)
    
    # Try to auto-detect input file if not provided
    if not args.input_file:
        # Look for common patterns in the experiment name
        exp_name = args.results_dir.name
        if "hotpot" in exp_name.lower():
            if "30" in exp_name:
                args.input_file = Path("data/hotpot_distractor_30.jsonl")
            else:
                args.input_file = Path("data/hotpot_distractor_1k.jsonl")
        elif "gsm8k" in exp_name.lower():
            if "100" in exp_name:
                args.input_file = Path("data/gsm8k_100.jsonl")
            elif "300" in exp_name or "500" in exp_name:
                args.input_file = Path("data/gsm8k_500.jsonl")
        
        if args.input_file and args.input_file.exists():
            print(f"🔍 Auto-detected input file: {args.input_file}")
        else:
            print("⚠️  Warning: Could not auto-detect input file. Specify with --input-file")
            args.input_file = None
    
    rejudge_answer_correctness(
        args.results_dir,
        args.judge_model,
        args.input_file,
        args.force,
        args.aad_mode
    )
