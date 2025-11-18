"""Test Ollama client directly."""
# Use absolute imports when running as a script
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clients.ollama_client import OllamaClient

def test_ollama_client(use_streaming=True):
    model = "deepseek-r1:8b"
    llm_client = OllamaClient(model_id=model)
    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a careful assistant who cites evidence accurately.\n\nYou are an expert analyst. Use only the evidence_passage tool messages to answer the question.\n\nCite supporting passages by their identifiers. Then provide your final answer.\n\nUse citation markers like [[CITE]] that match the cite or doc_id values provided in the tool messages.\n\nQuestion:\n\n\nRespond with your final answer and citations."
            },
            {
                "role": "user",
                "content": "Question: Which dog is based in Switzerland, Appenzeller Sennenhund or Drentse Patrijshond?"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_call_0",
                        "type": "function",
                        "function": {
                            "name": "evidence_passage",
                            "arguments": {
                                "type": "passage",
                                "index": 0,
                                "label": "Appenzeller Sennenhund",
                                "text": "The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \"Senn\", herders in the Appenzell region of Switzerland.",
                                "display": "[Appenzeller Sennenhund] The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \"Senn\", herders in the Appenzell region of Switzerland.",
                                "doc_id": "en:Appenzeller_Sennenhund",
                                "cite": "Appenzeller Sennenhund"
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "evidence_passage",
                "content": "{\"type\": \"passage\", \"index\": 0, \"label\": \"Appenzeller Sennenhund\", \"text\": \"The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \\\"Senn\\\", herders in the Appenzell region of Switzerland.\", \"display\": \"[Appenzeller Sennenhund] The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \\\"Senn\\\", herders in the Appenzell region of Switzerland.\", \"doc_id\": \"en:Appenzeller_Sennenhund\", \"cite\": \"Appenzeller Sennenhund\"}",
                "tool_call_id": "tool_call_0"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_call_1",
                        "type": "function",
                        "function": {
                            "name": "evidence_passage",
                            "arguments": {
                                "type": "passage",
                                "index": 1,
                                "label": "Drentse Patrijshond",
                                "text": "The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \"Drent\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.",
                                "display": "[Drentse Patrijshond] The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \"Drent\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.",
                                "doc_id": "en:Drentse_Patrijshond",
                                "cite": "Drentse Patrijshond"
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "evidence_passage",
                "content": "{\"type\": \"passage\", \"index\": 1, \"label\": \"Drentse Patrijshond\", \"text\": \"The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \\\"Drent\\\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.\", \"display\": \"[Drentse Patrijshond] The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \\\"Drent\\\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.\", \"doc_id\": \"en:Drentse_Patrijshond\", \"cite\": \"Drentse Patrijshond\"}",
                "tool_call_id": "tool_call_1"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_call_2",
                        "type": "function",
                        "function": {
                            "name": "evidence_passage",
                            "arguments": {
                                "type": "tool_output",
                                "index": 0,
                                "label": "tool_1",
                                "tool": "wiki_search",
                                "input": "Appenzeller Sennenhund",
                                "output": "The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \"Senn\", herders in the Appenzell region of Switzerland.",
                                "text": "- wiki_search | input=Appenzeller Sennenhund | output=The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \"Senn\", herders in the Appenzell region of Switzerland.",
                                "display": "- wiki_search | input=Appenzeller Sennenhund | output=The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \"Senn\", herders in the Appenzell region of Switzerland."
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "evidence_passage",
                "content": "{\"type\": \"tool_output\", \"index\": 0, \"label\": \"tool_1\", \"tool\": \"wiki_search\", \"input\": \"Appenzeller Sennenhund\", \"output\": \"The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \\\"Senn\\\", herders in the Appenzell region of Switzerland.\", \"text\": \"- wiki_search | input=Appenzeller Sennenhund | output=The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \\\"Senn\\\", herders in the Appenzell region of Switzerland.\", \"display\": \"- wiki_search | input=Appenzeller Sennenhund | output=The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps. The name Sennenhund refers to people called \\\"Senn\\\", herders in the Appenzell region of Switzerland.\"}",
                "tool_call_id": "tool_call_2"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_call_3",
                        "type": "function",
                        "function": {
                            "name": "evidence_passage",
                            "arguments": {
                                "type": "tool_output",
                                "index": 1,
                                "label": "tool_2",
                                "tool": "wiki_search",
                                "input": "Drentse Patrijshond",
                                "output": "The Drentsche Patријshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \"Drent\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.",
                                "text": "- wiki_search | input=Drentse Patrijshond | output=The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \"Drent\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.",
                                "display": "- wiki_search | input=Drentse Patrijshond | output=The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \"Drent\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes."
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "evidence_passage",
                "content": "{\"type\": \"tool_output\", \"index\": 1, \"label\": \"tool_2\", \"tool\": \"wiki_search\", \"input\": \"Drentse Patrijshond\", \"output\": \"The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \\\"Drent\\\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.\", \"text\": \"- wiki_search | input=Drentse Patријshond | output=The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \\\"Drent\\\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.\", \"display\": \"- wiki_search | input=Drentse Patrijshond | output=The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe. Called the Dutch Partridge Dog (or \\\"Drent\\\" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed clubs operate in Belgium, Denmark, Scandinavia and North America. The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog. An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.\"}",
                "tool_call_id": "tool_call_3"
            },
            {
                "role": "user",
                "content": "Question: Which dog is based in Switzerland, Appenzeller Sennenhund or Drentse Patrijshond?"
            }
        ],
        "temperature": 0.2,
        "stream": use_streaming,  # Configurable streaming mode
        "think": True  # Enable reasoning mode for DeepSeek-R1
    }

    print(f"Testing Ollama client with model: {model}")
    print(f"Streaming mode: {use_streaming}")
    print(f"Temperature: {request_data['temperature']}")
    print(f"Think mode: {request_data['think']}")
    print("-" * 80)
    
    response = llm_client.send_chat_request(model, request_data)
    
    print("\n" + "=" * 80)
    print("RESPONSE ANALYSIS:")
    print("=" * 80)
    
    # Check for empty response
    content = response.get("text", "")
    reasoning = response.get("reasoning_text", "")
    
    print(f"\nContent length: {len(content)} chars")
    print(f"Reasoning length: {len(reasoning) if reasoning else 0} chars")
    
    if content:
        print(f"\nVisible content:\n{content[:500]}...")
    else:
        print("\n⚠️  WARNING: Empty content!")
    
    if reasoning:
        print(f"\nReasoning (first 500 chars):\n{reasoning[:500]}...")
    else:
        print("\n⚠️  WARNING: No reasoning captured!")
    
    # Show usage stats
    usage = response.get("usage", {})
    if usage:
        print(f"\nToken usage:")
        print(f"  Prompt: {usage.get('prompt_tokens', 0)}")
        print(f"  Completion: {usage.get('completion_tokens', 0)}")
        print(f"  Total: {usage.get('total_tokens', 0)}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING WITH STREAMING (default)")
    print("=" * 80)
    test_ollama_client(use_streaming=False)
    
    print("\n\n")
    print("=" * 80)
    print("TESTING WITHOUT STREAMING")
    print("=" * 80)
    test_ollama_client(use_streaming=False)
