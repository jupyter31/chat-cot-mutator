import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Iterator
from .base_llm_client import BaseLLMClient
import json
import time

class HuggingFaceClient(BaseLLMClient):
    """Hugging Face Transformers implementation of LLM client"""
    
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "auto", torch_dtype=torch.float16):
        """
        Initialize Hugging Face client
        
        Args:
            model_name: HuggingFace model name (e.g., "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct")
            device: Device to run the model on ("auto", "cpu", "cuda", etc.)
            torch_dtype: PyTorch data type for the model
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize tokenizer and model
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        
        print(f"Model {model_name} loaded successfully!")
    
    def _format_messages_for_phi(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for Phi models using their expected format.
        Phi models typically expect a specific chat format.
        """
        if len(messages) == 0:
            return ""
        
        # For Phi-3, use the Instruct format
        if "phi-3" in self.model_name.lower():
            formatted_text = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    formatted_text += f"<|system|>\n{content}<|end|>\n"
                elif role == "user":
                    formatted_text += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    formatted_text += f"<|assistant|>\n{content}<|end|>\n"
            
            # Add assistant tag for generation
            formatted_text += "<|assistant|>\n"
            return formatted_text
        
        # For Phi-2 and other models, use a simpler format
        else:
            formatted_text = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    formatted_text += f"System: {content}\n\n"
                elif role == "user":
                    formatted_text += f"Human: {content}\n\n"
                elif role == "assistant":
                    formatted_text += f"Assistant: {content}\n\n"
            
            # Add assistant prefix for generation
            formatted_text += "Assistant:"
            return formatted_text
    
    def send_chat_request(self, model_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single chat completion request"""
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 1000)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.95)
        
        # Format messages for the model
        formatted_prompt = self._format_messages_for_phi(messages)
        
        # Generate response
        response = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,  # Only return the generated part
            clean_up_tokenization_spaces=True
        )
        
        # Extract the generated text
        generated_text = response[0]["generated_text"].strip()
        
        # Clean up the response (remove any remaining special tokens)
        if "phi-3" in self.model_name.lower():
            # Remove any remaining special tokens for Phi-3
            generated_text = generated_text.replace("<|end|>", "").strip()
        
        # Format response to match OpenAI API structure
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(self.tokenizer.encode(generated_text))
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
                    print(f"Error processing request: {e}")
                    # Create error response
                    batch_responses.append({
                        "choices": [{"message": {"role": "assistant", "content": None}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    })
            
            results.extend(batch_responses)
            end_time = time.time()
            print(f" --> Batch {i // batch_size + 1} processed in {end_time - start_time:.2f} seconds")
            
            # Clear GPU cache between batches if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Extract content from responses
        result_contents = []
        for r in results:
            try:
                content = r["choices"][0]["message"]["content"]
                result_contents.append(content)
            except:
                result_contents.append(None)
        
        return result_contents
    
    def send_stream_chat_completion_request(self, model_name: str, request_data: Dict[str, Any]) -> Iterator[Any]:
        """
        Send a streaming chat completion request
        Note: This is a simplified implementation that yields the complete response at once.
        For true streaming, you'd need to implement token-by-token generation.
        """
        messages = request_data.get("messages", [])
        max_tokens = request_data.get("max_tokens", 1000)
        temperature = request_data.get("temperature", 0.7)
        
        # Format messages for the model
        formatted_prompt = self._format_messages_for_phi(messages)
        
        # For streaming, we'll simulate by generating the full response and yielding it in chunks
        response = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        generated_text = response[0]["generated_text"].strip()
        
        if "phi-3" in self.model_name.lower():
            generated_text = generated_text.replace("<|end|>", "").strip()
        
        # Yield the text in chunks to simulate streaming
        chunk_size = 10  # characters per chunk
        for i in range(0, len(generated_text), chunk_size):
            chunk = generated_text[i:i+chunk_size]
            yield chunk
        
        # Yield final response structure
        final_response = {
            "choices": [{
                "message": {
                    "content": generated_text,
                    "role": "assistant"
                }
            }],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(self.tokenizer.encode(generated_text))
            }
        }
        
        yield final_response


# Specific implementations for different Phi models

class Phi2Client(HuggingFaceClient):
    """Phi-2 specific client"""
    def __init__(self, device: str = "auto"):
        super().__init__(
            model_name="microsoft/phi-2",
            device=device,
            torch_dtype=torch.float16
        )

class Phi3Client(HuggingFaceClient):
    """Phi-3 specific client"""
    def __init__(self, model_size: str = "mini", device: str = "auto"):
        model_map = {
            "mini": "microsoft/Phi-3-mini-4k-instruct",
            "small": "microsoft/Phi-3-small-8k-instruct",
            "medium": "microsoft/Phi-3-medium-4k-instruct"
        }
        
        model_name = model_map.get(model_size, "microsoft/Phi-3-mini-4k-instruct")
        
        super().__init__(
            model_name=model_name,
            device=device,
            torch_dtype=torch.bfloat16  # Phi-3 works better with bfloat16
        )
