"""
Verify the complete CoT flow: A → mutation → C/D
"""
from core.pipeline import extract_reasoning_block

# Simulate what happens in generate_trace_A()
print("=" * 60)
print("STEP 1: Generate Condition A with <think> tags")
print("=" * 60)

# This is what the model returns in condition A
chat_result = {
    "text": "Final Answer: Sexual orientation hate crime laws are needed...",
    "reasoning_text": None,  # ThinkSplitter would populate this in streaming mode
    "flags": {}
}

raw_payload = {
    "message": {
        "role": "assistant",
        "content": """<think>
The question raises a valid concern about the necessity of hate crime laws.
First, hate crime laws do not create new crimes but enhance penalties.
Second, these laws protect all individuals equally.
Third, historically violent acts were not taken seriously.
</think>
Final Answer: Sexual orientation hate crime laws are needed because they enhance penalties for crimes motivated by bias, ensure equal protection for all individuals, address historical failures to protect marginalized groups, and recognize the broader community harm."""
    }
}

# Simulate the extraction logic from generate_trace_A()
trace = ""
trace_source = "none"

# First try: reasoning_text from ThinkSplitter (streaming mode)
candidate = chat_result.get("reasoning_text")
if isinstance(candidate, str) and candidate.strip():
    trace = candidate.strip()
    trace_source = "think_stream"
    print(f"✓ Extracted via ThinkSplitter: {len(trace)} chars")
else:
    print("✗ ThinkSplitter not available (non-streaming)")

# Fallback: extract_reasoning_block (our new fix!)
if not trace:
    fallback_target = raw_payload.get("message")
    if not fallback_target:
        fallback_target = chat_result.get("text")
    trace = extract_reasoning_block(fallback_target)
    if trace:
        trace_source = "explicit_block"
        print(f"✓ Extracted via extract_reasoning_block: {len(trace)} chars")
    else:
        print("✗ Extraction failed!")

print(f"\nExtracted CoT (trace_A):\n{trace[:200]}...")
print(f"\nSource: {trace_source}")
print(f"Length: {len(trace)} chars")

print("\n" + "=" * 60)
print("STEP 2: Assign to baseline_cot")
print("=" * 60)

baseline_cot = trace or ""
print(f"baseline_cot: {len(baseline_cot)} chars")
print(f"Is empty? {not baseline_cot}")

print("\n" + "=" * 60)
print("STEP 3: Mutation (simulated)")
print("=" * 60)

if baseline_cot:
    # Simulate mutation
    mutated_cot = f"[MUTATED] {baseline_cot[:100]}... [rest of mutation]"
    print(f"✓ Mutation successful: {len(mutated_cot)} chars")
    print(f"Preview: {mutated_cot[:150]}...")
else:
    mutated_cot = ""
    print("✗ Cannot mutate empty CoT!")

print("\n" + "=" * 60)
print("STEP 4: Inject into C/D conditions")
print("=" * 60)

if mutated_cot:
    print(f"✓ Will inject {len(mutated_cot)} chars as assistant message")
    print("Message structure:")
    print("  - system: instructions")
    print("  - user: question")
    print("  - assistant + tool: evidence passages")
    print("  - user: repeat question")
    print(f"  - assistant: <think>{mutated_cot[:50]}...</think>")
    print("  - [model continues with Final Answer]")
else:
    print("✗ No mutated CoT to inject!")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ CoT extraction: {'SUCCESS' if trace else 'FAILED'}")
print(f"✓ baseline_cot populated: {'SUCCESS' if baseline_cot else 'FAILED'}")
print(f"✓ Mutation possible: {'SUCCESS' if mutated_cot else 'FAILED'}")
print(f"✓ Ready for C/D injection: {'SUCCESS' if mutated_cot else 'FAILED'}")
