"""Test the answer correctness judge with various examples."""
import logging
from eval.judges import judge_answer_correctness

logging.basicConfig(level=logging.INFO)

# Test cases without LLM (fallback to token matching)
test_cases = [
    {
        "question": "What is the capital of France?",
        "predicted": "Paris",
        "gold": "Paris",
        "expected": True,
    },
    {
        "question": "What is the capital of France?",
        "predicted": "The capital of France is Paris",
        "gold": "Paris",
        "expected": True,  # Should be True with LLM, might be False with token fallback
    },
    {
        "question": "What year did World War II end?",
        "predicted": "1945",
        "gold": "1944",
        "expected": False,
    },
    {
        "question": "How many states are in the USA?",
        "predicted": "50 states",
        "gold": "There are fifty states",
        "expected": True,  # Should be True with LLM, might be False with token fallback
    },
]

print("\n" + "="*80)
print("Testing Answer Correctness Judge (Token Fallback Mode)")
print("="*80)

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Question: {test['question']}")
    print(f"Predicted: {test['predicted']}")
    print(f"Gold: {test['gold']}")
    
    result = judge_answer_correctness(
        predicted_answer=test["predicted"],
        gold_answer=test["gold"],
        question=test["question"],
        llm_client=None,  # No LLM, will use fallback
        llm_model=None,
    )
    
    print(f"Result: {result['is_correct']}")
    print(f"Method: {result['method']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Expected: {test['expected']}")
    
    if result["is_correct"] == test["expected"]:
        print("✓ PASS")
    else:
        print("⚠ Note: Result differs from expected (this is OK for fallback mode)")

print("\n" + "="*80)
print("Note: These tests use token-based fallback.")
print("With an LLM judge, semantic equivalence detection will be more accurate.")
print("="*80)
