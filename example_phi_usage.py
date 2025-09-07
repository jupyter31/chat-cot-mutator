#!/usr/bin/env python3
"""
Example script demonstrating how to use Phi models with the chat-dsat-mutator.

This script shows how to:
1. Initialize a Phi client
2. Test basic chat functionality
3. Run a simple mutation example

Requirements:
    pip install torch transformers

Usage:
    python example_phi_usage.py
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clients.client_factory import create_llm_client

def test_phi_client():
    """Test basic Phi client functionality"""
    
    print("🤖 Initializing Phi-2 client...")
    print("Note: First run will download the model (~6GB)")
    
    try:
        # Create Phi-2 client
        client = create_llm_client(
            "phi",
            model_name="phi-2",
            device="auto"  # Will use GPU if available
        )
        
        print("✅ Phi-2 client initialized successfully!")
        
        # Test simple chat request
        print("\n📝 Testing simple chat request...")
        
        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = client.send_chat_request("phi-2", request)
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        # Test batch request with mutation-like examples
        print("\n🔄 Testing batch request (simulating mutation workflow)...")
        
        batch_requests = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that explains concepts clearly."},
                    {"role": "user", "content": "Explain photosynthesis in one sentence."}
                ],
                "max_tokens": 50,
                "temperature": 0.5
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that explains concepts clearly."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 20,
                "temperature": 0.3
            }
        ]
        
        batch_responses = client.send_batch_chat_request("phi-2", batch_requests, batch_size=2)
        
        for i, response in enumerate(batch_responses):
            print(f"Batch response {i+1}: {response}")
        
        print("\n✅ All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages:")
        print("    pip install torch transformers")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have sufficient memory (6GB+ RAM or GPU memory)")


def test_phi3_client():
    """Test Phi-3 client functionality"""
    
    print("\n🚀 Testing Phi-3 Mini client...")
    print("Note: Phi-3 is larger and more capable than Phi-2")
    
    try:
        # Create Phi-3 client
        client = create_llm_client(
            "phi",
            model_name="phi-3",
            model_size="mini",  # Options: "mini", "small", "medium"
            device="auto"
        )
        
        print("✅ Phi-3 Mini client initialized successfully!")
        
        # Test with a more complex prompt
        request = {
            "messages": [
                {"role": "system", "content": "You are an expert software engineer."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        response = client.send_chat_request("phi-3", request)
        print(f"Phi-3 Response:\n{response['choices'][0]['message']['content']}")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
    except Exception as e:
        print(f"❌ Error with Phi-3: {e}")
        print("Phi-3 requires more memory than Phi-2. Consider using Phi-2 instead.")


if __name__ == "__main__":
    print("=== Phi Model Testing ===")
    
    # Test Phi-2 (smaller, faster)
    test_phi_client()
    
    # Uncomment to test Phi-3 (larger, more capable)
    # test_phi3_client()
    
    print("\n=== Testing Complete ===")
    print("\nTo use Phi models in your chat-dsat-mutator:")
    print("1. Install dependencies: pip install torch transformers")
    print("2. Modify chat_mutator_controller.py:")
    print("   from client_config import get_phi2_client")
    print("   llm_client = get_phi2_client()")
    print("3. Run your application normally!")
