"""Test that the judge_answer_correctness fix works with max_completion_tokens."""
import os
import pytest
from clients.openai_client import OpenAIClient
from eval.judges import judge_answer_correctness

@pytest.mark.integration
def test_judge_with_new_api():
    """Test that judge works with newer OpenAI models using max_completion_tokens."""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping test")
        return
    
    # Create OpenAI client
    client = OpenAIClient(api_key=api_key)
    
    # Test with a simple question
    result = judge_answer_correctness(
        predicted_answer="Martin O'Malley",
        gold_answer="Martin Joseph O'Malley",
        question="Who was the 61st Governor of Maryland?",
        llm_client=client,
        llm_model="gpt-4o-mini"  # Use a newer model that requires max_completion_tokens
    )
    
    print("\n=== Judge Answer Correctness Test ===")
    print(f"Predicted: Martin O'Malley")
    print(f"Gold: Martin Joseph O'Malley")
    print(f"Is Correct: {result['is_correct']}")
    print(f"Method: {result['method']}")
    print(f"Explanation: {result['explanation']}")
    if 'error' in result:
        print(f"ERROR: {result['error']}")
    else:
        print("✓ No errors - fix is working!")
    
    return result

if __name__ == "__main__":
    test_judge_with_new_api()
