"""
Test Ollama with the vLLM client.
Make sure Ollama is installed and running with a model pulled.

Install Ollama: https://ollama.com/download
Pull a model: ollama pull phi3:medium
"""

from clients.client_factory import create_llm_client

def main():
    print("Creating vLLM client to connect to Ollama...")
    
    # Create client pointing to Ollama
    client = create_llm_client(
        "vllm",
        base_url="http://localhost:11434/v1",  # Ollama's default port
        api_key="EMPTY"  # Ollama doesn't need an API key
    )
    
    print("Sending test request...")
    
    # Send a request (use dictionary format)
    request = {
        "messages": [
            {"role": "user", "content": "What is 2+2? Explain step by step."}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    result = client.send_chat_request(
        model_name="phi4-reasoning",  # Use the model name you pulled
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
    
    return result

if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama installed? Visit: https://ollama.com/download")
        print("2. Is Ollama running? Run: ollama serve")
        print("3. Have you pulled a model? Run: ollama pull phi3:medium")
        print("4. Check if Ollama is accessible: curl http://localhost:11434/api/tags")