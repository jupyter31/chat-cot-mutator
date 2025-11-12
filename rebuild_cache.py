"""Rebuild CoT cache from existing samples.jsonl file."""
import json
from pathlib import Path

results_dir = Path("results/exp_ollama_phi4_reasoning_30-1")
samples_file = results_dir / "samples.jsonl"

# Cache directory from config
cache_dir = Path("cache/phi4_reasoning_hotpot_30_cots")
cache_dir.mkdir(parents=True, exist_ok=True)

# Model and run_id info
run_id = "exp_ollama_phi4_reasoning_30-1"
model_name = "phi4-reasoning:latest"
safe_model_name = model_name.replace(":", "-")

print(f"Reading samples from: {samples_file}")
print(f"Cache directory: {cache_dir}")
print(f"Run ID: {run_id}")
print(f"Model: {model_name} (safe: {safe_model_name})")
print()

# Load all condition A results
condition_a_count = 0
with open(samples_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("condition") != "A":
            continue
        
        condition_a_count += 1
        sample_id = record["sample_id"]
        trace_a = record.get("trace_A", "")
        
        # Cache filename format: {run_id}__{safe_model_name}__{sample_id}.cot
        cache_filename = f"{run_id}__{safe_model_name}__{sample_id}.cot"
        cache_path = cache_dir / cache_filename
        meta_path = cache_path.with_suffix(".cot.json")
        
        # Write CoT text file
        cache_path.write_text(trace_a or "", encoding="utf-8")
        
        # Write metadata JSON
        meta_data = {
            "trace_A": trace_a,
            "final_answer_A": record.get("final_answer"),
            "final_answer_text_A": record.get("final_answer_text"),
            "raw_A": record.get("raw_response"),
            "usage": record.get("usage", {}),
            "citations_A": record.get("citations", []),
            "judge_A": record.get("judge", {}),
            "latency_s": record.get("latency_s", 0.0),
            "trace_A_source": record.get("trace_A_source", "generated"),
            "trace_A_reasoning_chars": len(trace_a or ""),
            "process_tokens": record.get("process_tokens"),
            "flags": record.get("response_flags", {}),
            "record": record,
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"✓ Created cache for sample: {sample_id}")
        print(f"  CoT length: {len(trace_a)} chars")
        print(f"  Files: {cache_path.name}, {meta_path.name}")
        print()

print(f"✓ Cache rebuild complete! Created {condition_a_count} cache files in {cache_dir}")
