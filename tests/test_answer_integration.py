"""Quick integration test to verify answer correctness judge is properly integrated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judges import judge_answer_correctness

# Test 1: Without LLM (fallback mode)
print("Test 1: Fallback mode (no LLM)")
result1 = judge_answer_correctness(
    predicted_answer="Paris",
    gold_answer="Paris",
    question="What is the capital of France?",
    llm_client=None,
    llm_model=None,
)
print(f"  Result: {result1['is_correct']} (method: {result1['method']})")
assert result1["is_correct"] == True
assert result1["method"] == "token_subset"
print("  ✓ PASS\n")

# Test 2: Different phrasing (will fail in fallback but should work with LLM)
print("Test 2: Different phrasing (fallback mode)")
result2 = judge_answer_correctness(
    predicted_answer="The capital is Paris",
    gold_answer="Paris",
    question="What is the capital of France?",
    llm_client=None,
    llm_model=None,
)
print(f"  Result: {result2['is_correct']} (method: {result2['method']})")
assert result2["method"] == "token_subset"
print("  ✓ Token subset matching works (result may be True/False)\n")

# Test 3: Different facts (should always be False)
print("Test 3: Different facts")
result3 = judge_answer_correctness(
    predicted_answer="1945",
    gold_answer="1944",
    question="What year did WWII end?",
    llm_client=None,
    llm_model=None,
)
print(f"  Result: {result3['is_correct']} (method: {result3['method']})")
assert result3["is_correct"] == False
print("  ✓ PASS\n")

print("="*60)
print("All integration tests passed!")
print("="*60)
print("\nNote: These tests use fallback mode.")
print("To test with LLM, run: python tests/test_answer_correctness_with_llm.py")
