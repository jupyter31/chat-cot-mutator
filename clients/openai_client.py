import openai
from typing import List, Dict, Any, Iterator
from .base_llm_client import BaseLLMClient
import json
import time

class OpenAIClient(BaseLLMClient):
    """OpenAI API implementation of LLM client"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            base_url: Optional base URL for API (useful for compatible APIs like Azure OpenAI)
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def send_chat_request(self, model_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single chat completion request"""
        response = self.client.chat.completions.create(
            model=model_name,
            **request
        )
        return response.model_dump()
    
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
                    response = self.client.chat.completions.create(
                        model=model_name,
                        **request
                    )
                    batch_responses.append(response.model_dump())
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
        request_data["stream"] = True
        
        stream = self.client.chat.completions.create(
            model=model_name,
            **request_data
        )
        
        final_content = ""
        final_funcs = []
        token_count = 0
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                final_content += content
                token_count += 1
                yield content
            
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if len(final_funcs) > tool_call.index:
                        final_funcs[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                    else:
                        final_funcs.append({
                            "index": tool_call.index,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                yield ""
        
        # Create final response structure
        final_response = {
            "choices": [{
                "message": {
                    "content": final_content,
                    "tool_calls": final_funcs if final_funcs else None
                }
            }],
            "usage": {
                "prompt_tokens": -1,  # Not tracked in this example
                "completion_tokens": token_count,
                "total_tokens": 0,
            }
        }
        
        yield final_response
