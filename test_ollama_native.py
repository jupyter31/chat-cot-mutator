"""
Test the Ollama native client.

Make sure Ollama is running with phi3:medium pulled!
Run: ollama pull phi3:medium
"""

from clients.client_factory import create_llm_client

def main():
    print("="*70)
    print("Testing Ollama Native Client")
    print("="*70)
    
    # Create Ollama client
    print("\n1. Creating Ollama client...")
    client = create_llm_client(
        "ollama",
        base_url="http://localhost:11434",
        model_id="deepseek-r1:8b",  # Using your installed model
        timeout_s=120
    )
    print("✅ Client created!")
    
    # Test simple request
    print("\n2. Sending test request...")
    request = {
        "messages": [
            {"role": "user", "content": "What is 2+2? Explain step by step."}
        ],
        "temperature": 0.7,
        # "max_tokens": 200
    }
    
    result = client.send_chat_request(
        model_name="deepseek-r1:8b",  # Using your installed model
        request=request
    )
    
    print("\n" + "="*70)
    print("RESPONSE:")
    print("="*70)
    print(result.get("text", ""))
    
    if result.get("reasoning_text"):
        print("\n" + "="*70)
        print("REASONING:")
        print("="*70)
        print(result["reasoning_text"])
    
    print("\n" + "="*70)
    print("METADATA:")
    print("="*70)
    print(f"Usage: {result.get('usage', {})}")
    print(f"Flags: {result.get('flags', {})}")
    print(f"Has reasoning: {bool(result.get('reasoning_text'))}")
    
    print("\n✅ Test complete!")
    
    # Test with DeepSeek thinking mode
    print("\n" + "="*70)
    print("Testing DeepSeek R1 Thinking Mode (if available)")
    print("="*70)
    
    try:
        deepseek_client = create_llm_client(
            "ollama",
            base_url="http://localhost:11434",
            model_id="deepseek-r1:8b",
            timeout_s=60
        )
        
        request_with_thinking = {
            "messages": [
                {"role": "user", "content": "Solve: If x+3=7, what is x?"}
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "think": True  # Enable DeepSeek thinking mode
        }
        
        result = deepseek_client.send_chat_request(
            model_name="deepseek-r1:8b",
            request=request_with_thinking
        )
        
        print("\nResponse:", result.get("text", ""))
        if result.get("reasoning_text"):
            print("\nThinking process:")
            print(result["reasoning_text"][:200], "...")
        
        print("\n✅ DeepSeek test complete!")
        
    except Exception as e:
        print(f"\n⚠️  DeepSeek test skipped: {e}")
        print("   (deepseek-r1:8b not available - that's OK)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama installed? Visit: https://ollama.com/download")
        print("2. Is Ollama running? It starts automatically after installation")
        print("3. Have you pulled phi3:medium? Run: ollama pull phi3:medium")
        print("4. Check if Ollama is accessible:")
        print("   curl http://localhost:11434/api/tags")
