"""Recompute metrics for existing experiment results."""
import json
from pathlib import Path
import sys

from eval.metrics import compute_overall_metrics, compute_metrics_by_mutation, token_latency_rows


def recompute_metrics(results_dir: Path):
    """Recompute metrics with new grounding score calculations."""
    samples_file = results_dir / "samples.jsonl"
    if not samples_file.exists():
        print(f"Error: {samples_file} not found")
        return
    
    # Load all samples
    results = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"Loaded {len(results)} samples from {samples_file}")
    
    # Recompute metrics
    overall = compute_overall_metrics(results)
    by_mutation = compute_metrics_by_mutation(results)
    
    # Save updated metrics
    overall_file = results_dir / "metrics_overall.json"
    with open(overall_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"✓ Wrote {overall_file}")
    
    by_mutation_file = results_dir / "metrics_by_mutation.json"
    with open(by_mutation_file, "w", encoding="utf-8") as f:
        json.dump(by_mutation, f, indent=2)
    print(f"✓ Wrote {by_mutation_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Updated Metrics Summary:")
    print("="*60)
    for condition, metrics in overall["conditions"].items():
        print(f"\nCondition {condition}:")
        print(f"  Count: {metrics['count']}")
        print(f"  AAD: {metrics['aad']:.3f}")
        print(f"  Grounding Score (NEW): {metrics.get('grounding_score', 'N/A'):.3f}")
        print(f"  Grounded Rate (OLD): {metrics['grounded_rate']:.3f}")
        print(f"  Answer Accuracy: {metrics.get('answer_accuracy', 'N/A')}")
    
    print(f"\nOverall:")
    print(f"  Hallucination Score (NEW): {overall.get('HallucinationScore', 'N/A'):.3f}")
    print(f"  Hallucination Rate (OLD): {overall['HallucinationRate']:.3f}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Default to the hotpot experiment
        results_dir = Path("results/exp_ollama_deepseek_8b_cots-hotpot_30")
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)
    
    recompute_metrics(results_dir)
