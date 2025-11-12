from core.pipeline import extract_reasoning_block

# Test <think> tag extraction
response_with_think = """<think>
The question raises a valid concern about the necessity of hate crime laws.
First, hate crime laws enhance penalties when bias is proven.
Second, these laws protect all individuals equally.
Third, they address historical enforcement failures.
</think>
Final Answer: Hate crime laws enhance penalties for bias-motivated crimes."""

# Test old Reasoning: format
response_with_reasoning = """Reasoning: The question raises concerns.
Hate crime laws enhance penalties.
Final Answer: They address bias-motivated violence."""

# Test extraction
think_result = extract_reasoning_block(response_with_think)
reasoning_result = extract_reasoning_block(response_with_reasoning)

print("=== <think> tag extraction ===")
print(f"Extracted: {think_result[:100]}...")
print(f"Length: {len(think_result)} chars")
print(f"Success: {len(think_result) > 0}")

print("\n=== Reasoning: format extraction ===")
print(f"Extracted: {reasoning_result[:100]}...")
print(f"Length: {len(reasoning_result)} chars")
print(f"Success: {len(reasoning_result) > 0}")

# Test with actual response structure
actual_response = {
    "role": "assistant",
    "content": response_with_think
}
actual_result = extract_reasoning_block(actual_response)
print("\n=== Dict input extraction ===")
print(f"Extracted: {actual_result[:100]}...")
print(f"Length: {len(actual_result)} chars")
print(f"Success: {len(actual_result) > 0}")
