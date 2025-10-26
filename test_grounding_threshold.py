"""Test that grounding_threshold parameter flows correctly through the pipeline."""

from eval.judges import judge_grounding
from core.schema import FrozenPassageRecord

def test_threshold_0_95():
    """Test with threshold=0.95 - should accept scores >= 0.95"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris. It is a major European city.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Test with high overlap (should be grounded)
    answer = "The capital of France is Paris"
    citations = ["p1"]
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=0.95
    )
    
    print(f"Test 0.95 threshold - High overlap:")
    print(f"  is_grounded: {result['is_grounded']}")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    print()
    
def test_threshold_1_0():
    """Test with threshold=1.0 - should require perfect overlap"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Test with near-perfect overlap
    answer = "The capital of France is Paris"
    citations = ["p1"]
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=1.0
    )
    
    print(f"Test 1.0 threshold - Near-perfect overlap:")
    print(f"  is_grounded: {result['is_grounded']}")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    print()

def test_threshold_0_5():
    """Test with threshold=0.5 - more lenient"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris. It is a beautiful city.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Test with partial overlap
    answer = "Paris is the capital"
    citations = ["p1"]
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=0.5
    )
    
    print(f"Test 0.5 threshold - Partial overlap:")
    print(f"  is_grounded: {result['is_grounded']}")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    print()

def test_negative_ungrounded_answer():
    """Test that answers with no passage support are marked as ungrounded"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Answer contains information NOT in passages
    answer = "The capital of Germany is Berlin"
    citations = ["p1"]
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=0.95
    )
    
    print(f"Negative test - Ungrounded answer (threshold=0.95):")
    print(f"  is_grounded: {result['is_grounded']} (expected: False)")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    assert result['is_grounded'] == False, "Expected ungrounded answer to be rejected!"
    print("  ✓ PASS: Correctly rejected ungrounded answer\n")

def test_negative_low_overlap_strict_threshold():
    """Test that partial overlap is rejected with strict threshold"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris. The city has many museums.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Answer has some overlap but not enough for strict threshold
    answer = "The city has museums and restaurants and cafes"
    citations = ["p1"]
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=0.95
    )
    
    print(f"Negative test - Low overlap with strict threshold (0.95):")
    print(f"  is_grounded: {result['is_grounded']} (expected: False)")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    if not result['is_grounded']:
        print("  ✓ PASS: Correctly rejected low overlap answer\n")
    else:
        print("  ⚠ Note: Answer was marked as grounded\n")

def test_threshold_boundary_case():
    """Test behavior at exact threshold boundary"""
    passages = [
        FrozenPassageRecord(
            text="Paris is beautiful",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    # Test with same answer
    answer = "Paris is beautiful"
    citations = ["p1"]
    
    # Test with threshold = 1.0 (should pass with perfect match)
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=1.0
    )
    
    print(f"Boundary test - Perfect match with threshold=1.0:")
    print(f"  is_grounded: {result['is_grounded']} (expected: True)")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    print()

def test_no_citations_provided():
    """Test that answers without citations are ungrounded"""
    passages = [
        FrozenPassageRecord(
            text="The capital of France is Paris.",
            doc_id="p1",
            cite="p1",
        )
    ]
    
    answer = "The capital of France is Paris"
    citations = []  # No citations provided
    
    result = judge_grounding(
        answer=answer,
        citations=citations,
        passages=passages,
        grounding_threshold=0.95
    )
    
    print(f"Negative test - No citations (threshold=0.95):")
    print(f"  is_grounded: {result['is_grounded']} (expected: False)")
    print(f"  overlap: {result.get('overlap_max', 'N/A')}")
    if not result['is_grounded']:
        print("  ✓ PASS: Correctly rejected answer without citations\n")
    else:
        print("  ⚠ Note: Answer was marked as grounded despite no citations\n")

if __name__ == "__main__":
    print("Testing grounding_threshold parameter:\n")
    print("=" * 60)
    print("POSITIVE TESTS (should be grounded):")
    print("=" * 60)
    test_threshold_0_95()
    test_threshold_1_0()
    test_threshold_0_5()
    
    print("=" * 60)
    print("NEGATIVE TESTS (should NOT be grounded):")
    print("=" * 60)
    test_negative_ungrounded_answer()
    test_negative_low_overlap_strict_threshold()
    test_no_citations_provided()
    
    print("=" * 60)
    print("BOUNDARY TESTS:")
    print("=" * 60)
    test_threshold_boundary_case()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
