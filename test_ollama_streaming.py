"""
Test the Ollama streaming client with real-time progress feedback.

Demonstrates:
- Streaming tokens as they're generated
- Real-time thinking/reasoning capture
- Progress feedback for long-running requests

Make sure Ollama is running!
"""

from clients.client_factory import create_llm_client
import sys

def test_streaming_with_progress():
    """Test streaming with visual progress feedback."""
    
    print("=" * 70)
    print("Testing Ollama Streaming Client")
    print("=" * 70)
    
    # Create client with longer timeout for reasoning models
    print("\n1. Creating Ollama client (phi4-reasoning)...")
    client = create_llm_client(
        "ollama",
        base_url="http://localhost:11434",
        model_id="phi4-reasoning:latest",
        timeout_s=300  # 5 minutes for complex reasoning
    )
    print("✅ Client created!")
    
    # Progress callback
    thinking_shown = False
    content_shown = False
    
    def on_progress(channel: str, delta: str):
        """Display streaming progress in real-time."""
        nonlocal thinking_shown, content_shown
        
        if channel == "thinking":
            if not thinking_shown:
                print("\n\n" + "=" * 70)
                print("🧠 REASONING (streaming):")
                print("=" * 70)
                thinking_shown = True
            print(delta, end="", flush=True)
        
        elif channel == "content":
            if not content_shown:
                if thinking_shown:
                    print("\n")  # Separator after thinking
                print("\n" + "=" * 70)
                print("💬 ANSWER (streaming):")
                print("=" * 70)
                content_shown = True
            print(delta, end="", flush=True)
    
    # Test with a reasoning question
    print("\n2. Sending test request with streaming...")
    print("\n📝 Question: 'If I have 3 apples and buy 2 more, then give away 1, how many do I have? Show your reasoning.'")
    
    request = {
        "messages": [
            {
                "role": "user",
                "content": "If I have 3 apples and buy 2 more, then give away 1, how many do I have? Show your reasoning step by step."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "on_delta": on_progress  # Enable streaming progress
    }
    
    result = client.send_chat_request(
        model_name="phi4-reasoning:latest",
        request=request
    )
    
    # Display metadata
    print("\n\n" + "=" * 70)
    print("📊 METADATA:")
    print("=" * 70)
    usage = result.get("usage", {})
    print(f"Tokens: {usage.get('total_tokens', 'N/A')} total "
          f"({usage.get('prompt_tokens', 'N/A')} prompt + "
          f"{usage.get('completion_tokens', 'N/A')} completion)")
    print(f"Time: {usage.get('elapsed_s', 0):.2f}s")
    print(f"Speed: {usage.get('completion_tokens', 0) / max(usage.get('elapsed_s', 1), 0.1):.1f} tokens/sec")
    
    flags = result.get("flags", {})
    print(f"\nFlags:")
    print(f"  - Has reasoning: {flags.get('has_reasoning', False)}")
    print(f"  - Reasoning leaked: {flags.get('leak_think', False)}")
    
    if result.get("reasoning_text"):
        print(f"\nReasoning captured: {len(result['reasoning_text'])} chars")
    
    print("\n✅ Test complete!")
    return result


def test_deepseek_thinking():
    """Test DeepSeek R1 with thinking mode."""
    
    print("\n\n" + "=" * 70)
    print("Testing DeepSeek R1 Thinking Mode")
    print("=" * 70)
    
    try:
        client = create_llm_client(
            "ollama",
            base_url="http://localhost:11434",
            model_id="deepseek-r1:8b",
            timeout_s=300
        )
        
        thinking_shown = False
        answer_shown = False
        
        def on_progress(channel: str, delta: str):
            nonlocal thinking_shown, answer_shown
            
            if channel == "thinking":
                if not thinking_shown:
                    print("\n🔬 HIDDEN THINKING (DeepSeek):")
                    print("-" * 70)
                    thinking_shown = True
                print(delta, end="", flush=True)
            elif channel == "content":
                if not answer_shown:
                    if thinking_shown:
                        print("\n")
                    print("\n💡 ANSWER:")
                    print("-" * 70)
                    answer_shown = True
                print(delta, end="", flush=True)
        
        print("\n📝 Question: 'Solve: 2x + 5 = 13'")
        
        request = {
            "messages": [
                {"role": "user", "content": "Solve: 2x + 5 = 13. Show your work."}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
            "think": True,  # Enable DeepSeek thinking mode
            "on_delta": on_progress
        }
        
        result = client.send_chat_request(
            model_name="deepseek-r1:8b",
            request=request
        )
        
        print("\n\n✅ DeepSeek test complete!")
        
        if result.get("reasoning_text"):
            print(f"Thinking captured: {len(result['reasoning_text'])} chars")
        
        return result
        
    except Exception as e:
        print(f"\n⚠️  DeepSeek test skipped: {e}")
        print("   (deepseek-r1:8b not available)")
        return None


def test_generator_streaming():
    """Test generator-based streaming API."""
    
    print("\n\n" + "=" * 70)
    print("Testing Generator-Based Streaming (send_stream_chat_completion_request)")
    print("=" * 70)
    
    try:
        client = create_llm_client(
            "ollama",
            base_url="http://localhost:11434",
            model_id="phi4-reasoning:latest",
            timeout_s=300
        )
        
        print("\n📝 Question: 'What is 7 * 8? Show your reasoning.'")
        print("\n🔄 Streaming tokens as they arrive...\n")
        
        request = {
            "messages": [
                {"role": "user", "content": "What is 7 * 8? Show your reasoning step by step."}
            ],
            "temperature": 0.7,
            "max_tokens": 400
        }
        
        thinking_buffer = []
        content_buffer = []
        result = None
        
        # Use the generator
        for chunk in client.send_stream_chat_completion_request(
            model_name="phi4-reasoning:latest",
            request_data=request
        ):
            if "delta" in chunk:
                channel = chunk.get("channel", "content")
                delta = chunk["delta"]
                
                if channel == "thinking":
                    if not thinking_buffer:
                        print("\n" + "=" * 70)
                        print("🧠 REASONING (generator stream):")
                        print("=" * 70)
                    thinking_buffer.append(delta)
                    print(delta, end="", flush=True)
                
                elif channel == "content":
                    if not content_buffer:
                        if thinking_buffer:
                            print("\n")
                        print("\n" + "=" * 70)
                        print("💬 ANSWER (generator stream):")
                        print("=" * 70)
                    content_buffer.append(delta)
                    print(delta, end="", flush=True)
            
            elif chunk.get("done"):
                result = chunk["result"]
        
        # Display metadata
        if result:
            print("\n\n" + "=" * 70)
            print("📊 METADATA:")
            print("=" * 70)
            usage = result.get("usage", {})
            print(f"Tokens: {usage.get('total_tokens', 'N/A')} total")
            print(f"Time: {usage.get('elapsed_s', 0):.2f}s")
            print(f"Speed: {usage.get('completion_tokens', 0) / max(usage.get('elapsed_s', 1), 0.1):.1f} tokens/sec")
            
            flags = result.get("flags", {})
            print(f"\nFlags:")
            print(f"  - Has reasoning: {flags.get('has_reasoning', False)}")
            print(f"  - Reasoning leaked: {flags.get('leak_think', False)}")
        
        print("\n✅ Generator streaming test complete!")
        return result
        
    except Exception as e:
        print(f"\n⚠️  Generator streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all streaming tests."""
    
    try:
        # Test 1: Phi-4-reasoning with streaming
        result1 = test_streaming_with_progress()
        
        # Test 2: DeepSeek R1 with thinking mode (if available)
        result2 = test_deepseek_thinking()
        
        # Test 3: Generator-based streaming
        test_generator_streaming()
        
        print("\n\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETE!")
        print("=" * 70)
        print("\nStreaming benefits:")
        print("  ✓ Real-time feedback (no waiting for full response)")
        print("  ✓ Better UX for slow reasoning models")
        print("  ✓ Progress visibility during long generations")
        print("  ✓ Early termination detection")
        
        print("\nNext steps:")
        print("  • Run full A/B/C/D experiment: python -m tools.runner --config configs/exp_ollama_test_3.yaml")
        print("  • Check results in: results/exp_ollama_test_3/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? It should start automatically")
        print("2. Have you pulled phi4-reasoning? Run: ollama pull phi4-reasoning")
        print("3. Check Ollama API: Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -Method Get")
        print("4. Check Ollama process: Get-Process ollama")
        sys.exit(1)


if __name__ == "__main__":
    main()
