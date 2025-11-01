"""Re-judge answer correctness for existing samples with the fixed judge."""
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

from eval.judges import judge_answer_correctness
from clients.client_factory import create_llm_client


def rejudge_samples(results_dir: Path, judge_model: str, input_file: Path = None):
    """Re-judge all samples in a results directory with the fixed judge."""
    samples_file = results_dir / "samples.jsonl"
    if not samples_file.exists():
        print(f"Error: {samples_file} not found")
        return
    
    # Parse judge model spec to get provider and model name
    print(f"Creating judge client for {judge_model}...")
    if ":" in judge_model:
        provider, model_name = judge_model.split(":", 1)
    else:
        provider = "openai"
        model_name = judge_model
    
    # Create judge client
    judge_client = create_llm_client(provider, endpoint=None)
    
    # Load all samples
    samples = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {samples_file}")
    
    # Load gold answers from input file
    gold_answers = {}
    if input_file and input_file.exists():
        print(f"Loading gold answers from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sample_id = record.get("id", "")
                    answer_gold = record.get("answer_gold", "")
                    if sample_id and answer_gold:
                        gold_answers[sample_id] = answer_gold
        print(f"Loaded {len(gold_answers)} gold answers")
    else:
        print("Warning: No input file provided, will try to extract gold answers from samples")
    
    # Count samples that need re-judging
    fallback_count = sum(
        1 for s in samples 
        if s.get("judge", {}).get("answer_correct_method") == "token_subset_fallback"
    )
    print(f"Found {fallback_count} samples with fallback judge (will re-judge all)")
    
    # Re-judge each sample
    updated_count = 0
    for idx, sample in enumerate(samples, 1):
        sample_id = sample.get("sample_id", "unknown")
        condition = sample.get("condition", "?")
        
        # Get answer correctness fields
        judge_data = sample.get("judge", {})
        old_method = judge_data.get("answer_correct_method", "unknown")
        old_correct = judge_data.get("answer_correct", None)
        
        # Skip if not using fallback (already judged correctly)
        if old_method != "token_subset_fallback":
            continue
        
        # Get required fields for judging
        predicted_answer = sample.get("final_answer_text", "")
        
        # Get gold answer from loaded gold_answers dict
        gold_answer = gold_answers.get(sample_id, "")
        
        # Get question from prompt
        question = sample.get("prompt", "")
        if question.startswith("Question: "):
            question = question[len("Question: "):]
        
        if not predicted_answer:
            print(f"  ⚠️  Sample {sample_id} ({condition}): Missing predicted answer, skipping")
            continue
        
        if not gold_answer:
            print(f"  ⚠️  Sample {sample_id} ({condition}): Missing gold answer, skipping")
            continue
        
        # Re-judge
        try:
            result = judge_answer_correctness(
                predicted_answer=predicted_answer,
                gold_answer=gold_answer,
                question=question,
                llm_client=judge_client,
                llm_model=model_name,
            )
            
            # Update judge data
            judge_data["answer_correct"] = result["is_correct"]
            judge_data["answer_correct_explanation"] = result["explanation"]
            judge_data["answer_correct_method"] = result["method"]
            if "raw_response" in result:
                judge_data["answer_correct_raw"] = result["raw_response"]
            
            sample["judge"] = judge_data
            updated_count += 1
            
            status = "✓" if result["is_correct"] else "✗"
            method = result["method"]
            print(f"  {status} Sample {sample_id} ({condition}): {method} -> {result['is_correct']}")
            
        except Exception as e:
            print(f"  ❌ Sample {sample_id} ({condition}): Error re-judging: {e}")
            continue
    
    # Save updated samples
    if updated_count > 0:
        backup_file = samples_file.with_suffix(".jsonl.backup")
        print(f"\nBacking up original to {backup_file}...")
        samples_file.rename(backup_file)
        
        with open(samples_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"✓ Saved {len(samples)} samples with {updated_count} re-judged")
        print(f"✓ Original backed up to {backup_file}")
        
        # Recompute metrics
        print("\nRecomputing metrics...")
        from eval.metrics import compute_overall_metrics, compute_metrics_by_mutation
        
        overall = compute_overall_metrics(samples)
        by_mutation = compute_metrics_by_mutation(samples)
        
        overall_file = results_dir / "metrics_overall.json"
        with open(overall_file, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)
        print(f"✓ Updated {overall_file}")
        
        by_mutation_file = results_dir / "metrics_by_mutation.json"
        with open(by_mutation_file, "w", encoding="utf-8") as f:
            json.dump(by_mutation, f, indent=2)
        print(f"✓ Updated {by_mutation_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Updated Metrics Summary:")
        print("="*60)
        for condition, metrics in overall["conditions"].items():
            print(f"\nCondition {condition}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Answer Accuracy: {metrics.get('answer_accuracy', 'N/A'):.3f}")
        print("="*60)
    else:
        print("\nNo samples needed re-judging!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rejudge_samples.py <results_dir> [judge_model] [input_file]")
        print("\nExample:")
        print("  python rejudge_samples.py results/exp_ollama_phi4_reasoning_30-1")
        print("  python rejudge_samples.py results/exp_ollama_phi4_reasoning_30-1 microsoft_internal:dev-gpt-o3-mini")
        print("  python rejudge_samples.py results/exp_ollama_phi4_reasoning_30-1 microsoft_internal:dev-gpt-o3-mini data/hotpot_distractor_30.jsonl")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)
    
    # Get judge model from args or try to read from config
    if len(sys.argv) >= 3:
        judge_model = sys.argv[2]
    else:
        # Try to infer from original experiment
        judge_model = "microsoft_internal:dev-gpt-o3-mini"  # Default
        print(f"Using default judge model: {judge_model}")
    
    # Get input file with gold answers
    input_file = None
    if len(sys.argv) >= 4:
        input_file = Path(sys.argv[3])
    else:
        # Try to auto-detect from results directory
        # Check if there's an input path in a config file or guess common paths
        possible_inputs = [
            Path("data/hotpot_distractor_30.jsonl"),
            Path("data/hotpot_distractor_1k.jsonl"),
        ]
        for path in possible_inputs:
            if path.exists():
                input_file = path
                print(f"Auto-detected input file: {input_file}")
                break
    
    if not input_file or not input_file.exists():
        print("Warning: Could not find input file with gold answers. Please specify it:")
        print(f"  python rejudge_samples.py {results_dir} {judge_model} <input_file>")
        sys.exit(1)
    
    rejudge_samples(results_dir, judge_model, input_file)
