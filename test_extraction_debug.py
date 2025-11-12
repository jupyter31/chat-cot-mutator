"""
Debug: Check what extract_reasoning_block sees
"""
from core.pipeline import extract_reasoning_block

# Test 1: Direct string with <think> tags
test1 = """<think>
The question raises a valid concern about the necessity of hate crime laws.
First, hate crime laws do not create new crimes but enhance penalties.
</think>
Final Answer: Sexual orientation hate crime laws are needed."""

print("=" * 60)
print("Test 1: Direct string")
print("=" * 60)
result1 = extract_reasoning_block(test1)
print(f"Input preview: {test1[:100]}...")
print(f"Extracted: {result1[:100] if result1 else '(EMPTY!)'}...")
print(f"Length: {len(result1)}")
print(f"Success: {len(result1) > 0}")

# Test 2: Dict with content (simulating message structure)
test2 = {
    "role": "assistant",
    "content": """<think>
The question raises a valid concern about the necessity of hate crime laws.
First, hate crime laws do not create new crimes but enhance penalties.
</think>
Final Answer: Sexual orientation hate crime laws are needed."""
}

print("\n" + "=" * 60)
print("Test 2: Dict with 'content' key")
print("=" * 60)
result2 = extract_reasoning_block(test2)
print(f"Input type: {type(test2)}")
print(f"Input preview: {str(test2)[:100]}...")
print(f"Extracted: {result2[:100] if result2 else '(EMPTY!)'}...")
print(f"Length: {len(result2)}")
print(f"Success: {len(result2) > 0}")

# Test 3: What if content is missing?
test3 = {
    "role": "assistant",
    "text": """<think>
The question raises a valid concern.
</think>
Final Answer: Laws are needed."""
}

print("\n" + "=" * 60)
print("Test 3: Dict with 'text' key (no 'content')")
print("=" * 60)
result3 = extract_reasoning_block(test3)
print(f"Input type: {type(test3)}")
print(f"Extracted: {result3[:50] if result3 else '(EMPTY!)'}...")
print(f"Length: {len(result3)}")
print(f"Success: {len(result3) > 0}")

# Test 4: Empty dict
test4 = {}

print("\n" + "=" * 60)
print("Test 4: Empty dict")
print("=" * 60)
result4 = extract_reasoning_block(test4)
print(f"Extracted: '{result4}'")
print(f"Length: {len(result4)}")
print(f"Success: {len(result4) > 0}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Test 1 (string): {'✓ PASS' if result1 else '✗ FAIL'}")
print(f"Test 2 (dict with content): {'✓ PASS' if result2 else '✗ FAIL'}")
print(f"Test 3 (dict with text): {'✓ PASS' if result3 else '✗ FAIL'}")
print(f"Test 4 (empty dict): {'✓ PASS (expected empty)' if not result4 else '✗ FAIL'}")
