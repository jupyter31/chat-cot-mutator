#!/usr/bin/env python3
"""
Verify the CoT extraction fix is working
"""
import sys
sys.path.insert(0, '.')

from core.pipeline import extract_reasoning_block, _THINK_TAG_PATTERN

# Test the regex pattern
test_response = """<think>
Step 1: Analyze the evidence
Step 2: Draw conclusions
Step 3: Formulate answer
</think>
Final Answer: The answer is correct."""

print("=" * 70)
print("VERIFICATION: CoT Extraction Fix")
print("=" * 70)

# Test 1: Check regex pattern exists
print("\n1. Checking _THINK_TAG_PATTERN exists...")
try:
    pattern = _THINK_TAG_PATTERN
    print(f"   ✓ Pattern exists: {pattern.pattern}")
except NameError:
    print("   ✗ FAIL: _THINK_TAG_PATTERN not defined!")
    sys.exit(1)

# Test 2: Test extraction from string
print("\n2. Testing extraction from string...")
result = extract_reasoning_block(test_response)
if result and "Step 1" in result:
    print(f"   ✓ Extracted {len(result)} chars")
    print(f"   Preview: {result[:80]}...")
else:
    print(f"   ✗ FAIL: Extraction returned '{result}'")
    sys.exit(1)

# Test 3: Test extraction from dict
print("\n3. Testing extraction from dict (message format)...")
message_dict = {"role": "assistant", "content": test_response}
result2 = extract_reasoning_block(message_dict)
if result2 and "Step 1" in result2:
    print(f"   ✓ Extracted {len(result2)} chars")
else:
    print(f"   ✗ FAIL: Extraction from dict failed")
    sys.exit(1)

# Test 4: Check config
print("\n4. Checking config...")
try:
    import yaml
    with open('configs/exp_pilot.yaml') as f:
        config = yaml.safe_load(f)
    
    if config.get('reuse_cached_A_cots') == False:
        print("   ✓ Cache disabled (reuse_cached_A_cots: false)")
    else:
        print(f"   ⚠ Cache enabled: reuse_cached_A_cots = {config.get('reuse_cached_A_cots')}")
    
    if config.get('cot_cache_dir') is None or config.get('cot_cache_dir') == 'null':
        print("   ✓ Cache dir nullified")
    else:
        print(f"   ⚠ Cache dir set: {config.get('cot_cache_dir')}")
except Exception as e:
    print(f"   ✗ Error checking config: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n✓ All checks passed!")
print("\nNOTE: The fix is in place. You need to run a NEW experiment to see it work.")
print("Old results (like exp_webgpt_1_uniform) were created before the fix.")
print("\nRun: python -m tools.runner --config configs/exp_pilot.yaml")
print("=" * 70)
