"""Integration test for answer correctness judge with actual LLM calls.

⚠️ This is an INTEGRATION TEST, not a unit test!

This test requires:
- Ollama running locally OR
- Microsoft internal GPT-4o configured

This makes REAL LLM API calls and is:
- Slow (network latency)
- Non-deterministic (LLM outputs can vary)
- Dependent on external services

For fast, deterministic UNIT TESTS, use:
    python tests/test_answer_correctness_unit.py

Run this integration test with:
    python tests/test_answer_correctness_integration.py
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judges import judge_answer_correctness
from clients.client_factory import create_llm_client

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Test cases that benefit from semantic understanding
test_cases = [
    {
        "question": "What is the capital of France?",
        "predicted": "Paris",
        "gold": "Paris",
        "expected": True,
        "description": "Exact match",
    },
    {
        "question": "What is the capital of France?",
        "predicted": "The capital of France is Paris.",
        "gold": "Paris",
        "expected": True,
        "description": "Different phrasing, same answer",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "predicted": "William Shakespeare",
        "gold": "Shakespeare",
        "expected": True,
        "description": "Full name vs surname only",
    },
    {
        "question": "What year did World War II end?",
        "predicted": "1945",
        "gold": "1944",
        "expected": False,
        "description": "Different facts (wrong year)",
    },
    {
        "question": "How many states are in the USA?",
        "predicted": "50 states",
        "gold": "There are fifty states in the United States",
        "expected": True,
        "description": "Number format difference (50 vs fifty)",
    },
    {
        "question": "What is the chemical formula for water?",
        "predicted": "H2O",
        "gold": "Two hydrogen atoms and one oxygen atom",
        "expected": True,
        "description": "Semantic equivalence (formula vs description)",
    },
    {
        "question": "What is the tallest mountain?",
        "predicted": "Mount Everest",
        "gold": "K2",
        "expected": False,
        "description": "Different entities",
    },
    {
        "question": "When was the Declaration of Independence signed?",
        "predicted": "July 4, 1776",
        "gold": "1776",
        "expected": True,
        "description": "More specific vs general (both correct)",
    },
]

def test_with_model(model_spec: str):
    """Test with a specific model."""
    print(f"\n{'='*80}")
    print(f"Testing Answer Correctness Judge with {model_spec}")
    print('='*80)
    
    try:
        # Parse model_spec to get provider
        # Format: "provider:model" or just "provider"
        parts = model_spec.split(":", 1)
        provider = parts[0]
        client = create_llm_client(provider, endpoint=None)
        print(f"✓ Successfully created client for {model_spec}")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        print("Skipping LLM tests...")
        return False
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['description']} ---")
        print(f"Question: {test['question']}")
        print(f"Predicted: {test['predicted']}")
        print(f"Gold: {test['gold']}")
        print(f"Expected: {test['expected']}")
        
        try:
            result = judge_answer_correctness(
                predicted_answer=test["predicted"],
                gold_answer=test["gold"],
                question=test["question"],
                llm_client=client,
                llm_model=model_spec,
            )
            
            print(f"Result: {result['is_correct']}")
            print(f"Method: {result['method']}")
            print(f"Explanation: {result['explanation']}")
            
            if result["is_correct"] == test["expected"]:
                print("✓ PASS")
                passed += 1
            else:
                print("✗ FAIL - Result differs from expected")
                failed += 1
                if "raw_response" in result:
                    print(f"Raw response: {result['raw_response'][:200]}...")
                    
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print('='*80)
    
    return failed == 0


if __name__ == "__main__":
    # Try with GPT-4o first (if configured)
    print("Testing with microsoft_internal:dev-gpt-4o-gg...")
    success = test_with_model("microsoft_internal:dev-gpt-4o-gg")
    
    if not success:
        print("\n\nNote: Some tests may fail due to LLM variability.")
        print("The important thing is that semantic matching works better than token matching.")
