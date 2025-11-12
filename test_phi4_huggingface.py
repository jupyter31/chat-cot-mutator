"""
Test Phi-4-reasoning using the HuggingFace client (works immediately on Windows).
No vLLM or Ollama setup needed!
"""

from clients.client_factory import create_llm_client

def main():
    print("Creating HuggingFace client for Phi-4-reasoning...")
    print("(This will download the model on first run - may take several minutes)")
    print()
    
    # Use HuggingFace client - works on Windows, no additional setup
    client = create_llm_client(
        "huggingface",
        model_name="microsoft/Phi-4-reasoning"
    )
    
    print("\nSending test request...")
    # HuggingFace client expects a dictionary, not a ChatRequest object
    request = {
        "messages": [
            {
                "role": "user",
                "content": "If a train travels 120 km in 2 hours, what is its average speed? Think step by step."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    result = client.send_chat_request(
        model_name="microsoft/Phi-4-reasoning",
        request=request
    )
    
    print("\n" + "="*70)
    print("RESPONSE:")
    print("="*70)
    print(result.get("text", ""))
    
    if result.get("reasoning_text"):
        print("\n" + "="*70)
        print("REASONING TRACE:")
        print("="*70)
        print(result["reasoning_text"])
    
    print("\n" + "="*70)
    print("METADATA:")
    print("="*70)
    print(f"Usage: {result.get('usage', {})}")
    print(f"Model: microsoft/Phi-4-reasoning")
    print(f"Has reasoning: {bool(result.get('reasoning_text'))}")
    
    return result

if __name__ == "__main__":
    result = main()
