"""Unit tests for answer correctness judge with mocked LLM.

This is a proper unit test that uses mocking to avoid real LLM calls.
It tests the logic of judge_answer_correctness without external dependencies.

Run with: pytest tests/test_answer_correctness_unit.py
Or: python -m pytest tests/test_answer_correctness_unit.py -v
"""
import json
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judges import judge_answer_correctness


class TestAnswerCorrectnessUnit:
    """Unit tests with mocked LLM responses."""
    
    def _create_mock_client(self, mock_response: str):
        """Create a mock LLM client that returns a specific response."""
        mock_client = Mock()
        mock_client.send_chat_request = Mock(return_value={
            "text": mock_response,
            "usage": {"total_tokens": 100},
        })
        return mock_client
    
    def test_exact_match_with_llm_yes(self):
        """Test exact match - LLM should return YES."""
        mock_response = json.dumps({
            "semantically_equivalent": True,
            "explanation": "Both answers are identical."
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="Paris",
            gold_answer="Paris",
            question="What is the capital of France?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "llm_semantic"
        assert "explanation" in result
        mock_client.send_chat_request.assert_called_once()
    
    def test_semantic_equivalence_different_phrasing(self):
        """Test semantic equivalence with different phrasing."""
        mock_response = json.dumps({
            "semantically_equivalent": True,
            "explanation": "Both answers refer to the same city, just phrased differently."
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="The capital of France is Paris.",
            gold_answer="Paris",
            question="What is the capital of France?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "llm_semantic"
    
    def test_different_facts_should_be_incorrect(self):
        """Test that different facts are marked as incorrect."""
        mock_response = json.dumps({
            "semantically_equivalent": False,
            "explanation": "1945 and 1944 are different years - factually incorrect."
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="1945",
            gold_answer="1944",
            question="What year did World War II end?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == False
        assert result["method"] == "llm_semantic"
        assert "1945" in result["explanation"] or "different" in result["explanation"].lower()
    
    def test_number_format_equivalence(self):
        """Test that different number formats (50 vs fifty) are equivalent."""
        mock_response = json.dumps({
            "semantically_equivalent": True,
            "explanation": "50 and fifty represent the same number."
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="50 states",
            gold_answer="There are fifty states in the United States",
            question="How many states are in the USA?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "llm_semantic"
    
    def test_semantic_equivalence_formula_vs_description(self):
        """Test semantic equivalence between chemical formula and description."""
        mock_response = json.dumps({
            "semantically_equivalent": True,
            "explanation": "H2O and 'two hydrogen atoms and one oxygen atom' describe the same molecule."
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="H2O",
            gold_answer="Two hydrogen atoms and one oxygen atom",
            question="What is the chemical formula for water?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "llm_semantic"
    
    def test_fallback_when_no_llm_client(self):
        """Test fallback to token-based matching when no LLM provided."""
        result = judge_answer_correctness(
            predicted_answer="Paris",
            gold_answer="Paris",
            question="What is the capital of France?",
            llm_client=None,
            llm_model=None,
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "token_subset"
        assert "Fallback" in result["explanation"]
    
    def test_non_json_response_parsing_yes(self):
        """Test parsing of non-JSON response with YES."""
        mock_response = 'The answers are semantically_equivalent": true because they mean the same thing.'
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="Paris",
            gold_answer="Paris",
            question="What is the capital of France?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True
        assert result["method"] == "llm_semantic"
    
    def test_non_json_response_parsing_no(self):
        """Test parsing of non-JSON response with NO."""
        mock_response = 'The answers are semantically_equivalent": false because they differ.'
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="1945",
            gold_answer="1944",
            question="What year did World War II end?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == False
        assert result["method"] == "llm_semantic"
    
    def test_llm_error_fallback(self):
        """Test fallback to token matching when LLM raises an error."""
        mock_client = Mock()
        mock_client.send_chat_request = Mock(side_effect=Exception("API Error"))
        
        result = judge_answer_correctness(
            predicted_answer="Paris",
            gold_answer="Paris",
            question="What is the capital of France?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == True  # Falls back to token matching
        assert result["method"] == "token_subset_fallback"
        assert "error" in result
    
    def test_unparseable_response_defaults_to_false(self):
        """Test that unparseable responses default to False."""
        mock_response = "This is some random text with no clear answer."
        
        mock_client = self._create_mock_client(mock_response)
        
        result = judge_answer_correctness(
            predicted_answer="Something",
            gold_answer="Something else",
            question="What is the question?",
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        assert result["is_correct"] == False
        assert result["method"] == "llm_semantic"
        assert "Could not parse" in result["explanation"]
    
    def test_prompt_formatting(self):
        """Test that the prompt is correctly formatted with variables."""
        mock_response = json.dumps({
            "semantically_equivalent": True,
            "explanation": "Test"
        })
        
        mock_client = self._create_mock_client(mock_response)
        
        question = "What is 2+2?"
        predicted = "4"
        gold = "four"
        
        result = judge_answer_correctness(
            predicted_answer=predicted,
            gold_answer=gold,
            question=question,
            llm_client=mock_client,
            llm_model="mock-model",
        )
        
        # Check that send_chat_request was called
        assert mock_client.send_chat_request.called
        
        # Get the actual request that was sent
        call_args = mock_client.send_chat_request.call_args
        request = call_args[0][1]  # Second argument is the request dict
        
        # Verify the messages contain our variables
        messages_str = str(request.get("messages", []))
        assert question in messages_str
        assert predicted in messages_str
        assert gold in messages_str


def run_tests():
    """Run all tests manually (without pytest)."""
    import traceback
    
    test_class = TestAnswerCorrectnessUnit()
    test_methods = [
        method for method in dir(test_class)
        if method.startswith("test_") and callable(getattr(test_class, method))
    ]
    
    passed = 0
    failed = 0
    
    print("Running Unit Tests for Answer Correctness Judge")
    print("=" * 80)
    
    for method_name in test_methods:
        print(f"\n{method_name}...", end=" ")
        try:
            method = getattr(test_class, method_name)
            method()
            print("✓ PASS")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_methods)} tests")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
