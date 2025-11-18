"""Test that answer correctness judge works with reasoning models (o3-mini).

This verifies the fix for reasoning models that don't support temperature parameter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judges import judge_answer_correctness
from clients.client_factory import create_llm_client

def test_with_reasoning_model():
    """Test with o3-mini (reasoning model that doesn't support temperature)."""
    print("Testing Answer Correctness Judge with Reasoning Model (o3-mini)")
    print("=" * 80)
    
    try:
        client = create_llm_client("microsoft_internal", endpoint=None)
        print("✓ Successfully created client for o3-mini")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        return False
    
    # Simple test case
    test_case = {
        "question": "What is the capital of France?",
        "predicted": "Paris",
        "gold": "Paris",
        "expected": True,
    }
    
    print(f"\nTest Case:")
    print(f"Question: {test_case['question']}")
    print(f"Predicted: {test_case['predicted']}")
    print(f"Gold: {test_case['gold']}")
    print(f"Expected: {test_case['expected']}")
    
    try:
        result = judge_answer_correctness(
            predicted_answer=test_case["predicted"],
            gold_answer=test_case["gold"],
            question=test_case["question"],
            llm_client=client,
            llm_model="microsoft_internal:dev-gpt-o3-mini",
        )
        
        print(f"\n✓ SUCCESS - No error!")
        print(f"Result: {result['is_correct']}")
        print(f"Method: {result['method']}")
        print(f"Explanation: {result['explanation']}")
        
        if result["is_correct"] == test_case["expected"]:
            print("\n✓ PASS - Result matches expected")
            return True
        else:
            print("\n✗ FAIL - Result differs from expected")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        if "temperature" in str(e).lower():
            print("❌ Still getting temperature error - fix didn't work!")
        return False

if __name__ == "__main__":
    success = test_with_reasoning_model()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ Fix verified! Reasoning models now work correctly.")
        print("The judge now skips temperature/max_completion_tokens for reasoning models.")
    else:
        print("\n" + "=" * 80)
        print("❌ Fix needs more work.")
