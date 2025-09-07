import anthropic
from typing import List, Dict, Any, Iterator
from .base_llm_client import BaseLLMClient
import time

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API implementation of LLM client"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Anthropic client
        
        Args:
            api_key: Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def send_chat_request(self, model_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single chat completion request"""
        # Convert OpenAI-style request to Anthropic format
        messages = request.get("messages", [])
        system_message = None
        
        # Extract system message if present
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        
        anthropic_request = {
            "model": model_name,
            "messages": messages,
            "max_tokens": request.get("max_tokens", 1000),
        }
        
        if system_message:
            anthropic_request["system"] = system_message
        
        if "temperature" in request:
            anthropic_request["temperature"] = request["temperature"]
        
        response = self.client.messages.create(**anthropic_request)
        
        # Convert to OpenAI-style response format
        return {
            "choices": [{
                "message": {
                    "content": response.content[0].text,
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    def send_batch_chat_request(self, model_name: str, batch_requests: List[Dict[str, Any]], batch_size: int = 5) -> List[str]:
        """Send multiple chat completion requests as a batch"""
        results = []
        total_batches = len(batch_requests) // batch_size + (1 if len(batch_requests) % batch_size else 0)
        
        for i in range(0, len(batch_requests), batch_size):
            print(f"Processing batch {i // batch_size + 1} / {total_batches} with {len(batch_requests[i:i+batch_size])} requests")
            start_time = time.time()
            current_batch = batch_requests[i:i+batch_size]
            
            batch_responses = []
            for request in current_batch:
                try:
                    response = self.send_chat_request(model_name, request)
                    batch_responses.append(response)
                except Exception as e:
                    raise e
            
            results.extend(batch_responses)
            end_time = time.time()
            print(f" --> Batch {i // batch_size + 1} processed in {end_time - start_time:.2f} seconds")
            
            # Sleep briefly between batches to avoid rate limiting
            if i + batch_size < len(batch_requests):
                time.sleep(1)
        
        result_contents = [r["choices"][0]["message"]["content"] for r in results]
        return result_contents
    
    def send_stream_chat_completion_request(self, model_name: str, request_data: Dict[str, Any]) -> Iterator[Any]:
        """Send a streaming chat completion request"""
        # Convert OpenAI-style request to Anthropic format
        messages = request_data.get("messages", [])
        system_message = None
        
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        
        anthropic_request = {
            "model": model_name,
            "messages": messages,
            "max_tokens": request_data.get("max_tokens", 1000),
            "stream": True
        }
        
        if system_message:
            anthropic_request["system"] = system_message
        
        if "temperature" in request_data:
            anthropic_request["temperature"] = request_data["temperature"]
        
        final_content = ""
        token_count = 0
        
        with self.client.messages.stream(**anthropic_request) as stream:
            for text in stream.text_stream:
                final_content += text
                token_count += 1
                yield text
        
        # Create final response structure
        final_response = {
            "choices": [{
                "message": {
                    "content": final_content,
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": -1,  # Not easily available in streaming
                "completion_tokens": token_count,
                "total_tokens": 0,
            }
        }
        
        yield final_response
