"""Test centralized think tag cleaning in pipeline."""
import sys
sys.path.insert(0, '.')

from core.pipeline import _extract_and_strip_think_tags

# Test cases
test_cases = [
    {
        "name": "Simple think tags",
        "input": "<think>This is my reasoning</think>The final answer is 42.",
        "expected_reasoning": "This is my reasoning",
        "expected_text": "The final answer is 42.",
        "expected_leak": False,
    },
    {
        "name": "Multiple think blocks",
        "input": "<think>First thought</think>Some text<think>Second thought</think>More text",
        "expected_reasoning": "First thought\n\nSecond thought",
        "expected_text": "Some textMore text",
        "expected_leak": False,
    },
    {
        "name": "No think tags",
        "input": "Just plain text with no tags",
        "expected_reasoning": "",
        "expected_text": "Just plain text with no tags",
        "expected_leak": False,
    },
    {
        "name": "Unclosed think tag",
        "input": "<think>Reasoning that never closes",
        "expected_reasoning": "Reasoning that never closes",
        "expected_text": "",
        "expected_leak": False,
    },
    {
        "name": "Nested tags (malformed - edge case)",
        "input": "<think>Outer<think>Inner</think>Back to outer</think>Text",
        "expected_reasoning": "Outer<think>Inner",  # Extracts up to first close tag
        "expected_text": "Back to outer</think>Text",  # Remaining content may have artifacts
        "expected_leak": True,  # Malformed nesting will leave artifacts
        "note": "This is an edge case that shouldn't occur with well-behaved models"
    },
]

print("Testing Centralized Think Tag Extraction:")
print("=" * 80)

all_passed = True
for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"Input: {test['input'][:60]}{'...' if len(test['input']) > 60 else ''}")
    
    reasoning, text, has_leak = _extract_and_strip_think_tags(test['input'])
    
    passed = (
        reasoning == test['expected_reasoning'] and
        text == test['expected_text'] and
        has_leak == test['expected_leak']
    )
    
    if passed:
        print("✅ PASS")
    else:
        print("❌ FAIL")
        if reasoning != test['expected_reasoning']:
            print(f"  Expected reasoning: '{test['expected_reasoning']}'")
            print(f"  Got reasoning:      '{reasoning}'")
        if text != test['expected_text']:
            print(f"  Expected text: '{test['expected_text']}'")
            print(f"  Got text:      '{text}'")
        if has_leak != test['expected_leak']:
            print(f"  Expected leak: {test['expected_leak']}")
            print(f"  Got leak:      {has_leak}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✅ All tests passed!")
    print("\nCentralized think tag cleaning is working correctly.")
    print("This will apply to ALL clients: Ollama, Azure Foundry, OpenAI, Anthropic, etc.")
else:
    print("❌ Some tests failed - review extraction logic")

print("\nNext steps:")
print("1. Re-run experiments to get clean data")
print("2. Run: python analyze_samples.py 'results/exp_name' to verify")
print("3. Expect: 0% contamination, leak_think=False for all samples")
