import json
import sys
from collections import Counter

# Get experiment path from command line or use default
exp_path = sys.argv[1] if len(sys.argv) > 1 else r'results\exp_ollama_phi4_reasoning_30-1'
samples_file = f'{exp_path}\\samples.jsonl'

print(f"Analyzing: {samples_file}")
print("="*60)

# Load samples
samples = []
with open(samples_file, 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Total samples: {len(samples)}")

# Condition distribution
print("\nCondition distribution:")
print(Counter(s['condition'] for s in samples))

# Check for <think> tags in response field
print("\n" + "="*60)
print("CRITICAL ISSUE: <think> tags in response field")
print("="*60)
leak_count = sum(1 for s in samples if '<think>' in s.get('response', ''))
print(f"Samples with <think> in response: {leak_count}/{len(samples)} ({100*leak_count/len(samples):.1f}%)")

# Check leak_think flag
leak_flag_count = sum(1 for s in samples if s.get('response_flags', {}).get('leak_think', False))
print(f"Samples with leak_think=True: {leak_flag_count}/{len(samples)} ({100*leak_flag_count/len(samples):.1f}%)")

# Show examples by condition
print("\n" + "="*60)
print("Sample responses by condition (first 300 chars):")
print("="*60)
for cond in ['A', 'B', 'C', 'D']:
    sample = next((s for s in samples if s['condition'] == cond), None)
    if sample:
        print(f"\nCondition {cond}:")
        resp = sample['response'][:300]
        print(f"  Response preview: {resp}...")
        has_think = '<think>' in sample['response']
        print(f"  Has <think> tags: {has_think}")
        print(f"  leak_think flag: {sample.get('response_flags', {}).get('leak_think', 'N/A')}")

# Check if trace_A or reasoning_text exists
print("\n" + "="*60)
print("Reasoning fields check:")
print("="*60)
for cond in ['A', 'B', 'C', 'D']:
    cond_samples = [s for s in samples if s['condition'] == cond]
    has_trace = sum(1 for s in cond_samples if s.get('trace_A'))
    has_reasoning = sum(1 for s in cond_samples if s.get('reasoning_text'))
    print(f"Condition {cond}: {len(cond_samples)} samples")
    print(f"  - Has trace_A: {has_trace}/{len(cond_samples)}")
    print(f"  - Has reasoning_text: {has_reasoning}/{len(cond_samples)}")

# Check grounding scores
print("\n" + "="*60)
print("Grounding scores (may be affected by contaminated responses):")
print("="*60)
for cond in ['A', 'B', 'C', 'D']:
    cond_samples = [s for s in samples if s['condition'] == cond]
    grounded = sum(1 for s in cond_samples if s.get('judge', {}).get('is_grounded', False))
    print(f"Condition {cond}: {grounded}/{len(cond_samples)} grounded ({100*grounded/len(cond_samples) if cond_samples else 0:.1f}%)")

print("\n" + "="*60)
print("VERDICT:")
print("="*60)
if leak_count == len(samples):
    print("⚠️  ALL samples have <think> tags in response field!")
    print("⚠️  Results are CONTAMINATED - responses include reasoning traces")
    print("⚠️  Grounding evaluation is measuring contaminated text, not clean answers")
elif leak_count > 0:
    print(f"⚠️  {leak_count} samples have <think> tags in response field!")
    print(f"⚠️  {100*leak_count/len(samples):.1f}% contamination rate")
else:
    print("✓ No <think> tags found in response fields - results are clean")
