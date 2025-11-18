"""Test to reproduce the answer correctness judge issue with o3-mini."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from eval.judges import judge_answer_correctness
from clients.client_factory import create_llm_client

print("=" * 80)
print("Testing Answer Correctness Judge with o3-mini")
print("=" * 80)

try:
    # Create Microsoft client
    client = create_llm_client("microsoft_internal", endpoint=None)
    print("✓ Created microsoft_internal client")
    
    # Test with o3-mini
    print("\nCalling judge_answer_correctness with dev-gpt-o3-mini...")
    result = judge_answer_correctness(
        predicted_answer="Paris",
        gold_answer="Paris",
        question="What is the capital of France?",
        llm_client=client,
        llm_model="microsoft_internal:dev-gpt-o3-mini",
    )
    
    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"is_correct: {result['is_correct']}")
    print(f"method: {result['method']}")
    print(f"explanation: {result['explanation'][:200]}...")
    
    if result['method'] == 'token_subset_fallback':
        print("\n❌ FAILED - Judge used fallback due to error")
        if 'error' in result:
            print(f"Error: {result['error']}")
    else:
        print("\n✅ SUCCESS - Judge worked correctly")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
