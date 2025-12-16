"""Recompute metrics for existing experiment results."""
import json
from pathlib import Path
import sys

from eval.metrics import compute_overall_metrics, compute_metrics_by_mutation, token_latency_rows


def recompute_metrics(
    results_dir: Path, 
    aad_mode: str = "combined",
    ace_mode: str = "combined",
    alpha: float = 1.0,
    grounding_threshold: float = 0.85
):
    """Recompute metrics with specified AAD and ACE modes.
    
    Args:
        results_dir: Path to experiment results directory
        aad_mode: One of 'combined' (default), 'answer_only', 'grounding_only'
        ace_mode: Mode for ACE calculation (default: same as aad_mode)
        alpha: Exponent for grounding score in continuous ACE (default: 1.0)
        grounding_threshold: Threshold for binary grounding decision (default: 0.85)
    """
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
    
    # Recompute metrics with specified modes
    overall = compute_overall_metrics(
        results, 
        aad_mode=aad_mode,
        ace_mode=ace_mode,
        alpha=alpha,
        grounding_threshold=grounding_threshold
    )
    by_mutation = compute_metrics_by_mutation(results, aad_mode=aad_mode, grounding_threshold=grounding_threshold)
    
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
    print(f"  ACE (binary): {overall.get('ACE', 'N/A')}")
    print(f"  cACE (continuous): {overall.get('cACE', 'N/A')}")
    print(f"  Resistance (binary): {overall.get('Resistance', 'N/A')}")
    print(f"  cResistance (continuous): {overall.get('cResistance', 'N/A')}")
    print(f"  Hallucination Score (NEW): {overall.get('HallucinationScore', 'N/A'):.3f}")
    print(f"  Hallucination Rate (OLD): {overall['HallucinationRate']:.3f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recompute metrics for experiment results")
    parser.add_argument("results_dir", nargs="?", 
                       default="results/exp_ollama_deepseek_8b_cots-hotpot_30",
                       help="Path to experiment results directory")
    parser.add_argument("--aad-mode", 
                       choices=["combined", "answer_only", "grounding_only"],
                       default="answer_only",
                       help="AAD calculation mode (default: answer_only for GSM8K)")
    parser.add_argument("--ace-mode",
                       choices=["combined", "answer_only", "grounding_only"],
                       default=None,
                       help="ACE calculation mode (default: same as aad-mode)")
    parser.add_argument("--alpha",
                       type=float,
                       default=1.0,
                       help="Exponent for grounding score in continuous ACE (default: 1.0)")
    parser.add_argument("--grounding-threshold",
                       type=float,
                       default=0.85,
                       help="Threshold for binary grounding decision (default: 0.85)")
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)
    
    # Default ace_mode to aad_mode if not specified
    ace_mode = args.ace_mode if args.ace_mode else args.aad_mode
    
    print(f"Recomputing metrics with aad_mode='{args.aad_mode}', ace_mode='{ace_mode}', alpha={args.alpha}, grounding_threshold={args.grounding_threshold}...\n")
    recompute_metrics(
        results_dir, 
        aad_mode=args.aad_mode,
        ace_mode=ace_mode,
        alpha=args.alpha,
        grounding_threshold=args.grounding_threshold
    )
