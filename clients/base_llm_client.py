from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def send_chat_request(self, model_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single chat completion request"""
        pass
    
    @abstractmethod
    def send_batch_chat_request(self, model_name: str, batch_requests: List[Dict[str, Any]], batch_size: int = 5) -> List[str]:
        """Send multiple chat completion requests as a batch"""
        pass
    
    @abstractmethod
    def send_stream_chat_completion_request(self, model_name: str, request_data: Dict[str, Any]) -> Iterator[Any]:
        """Send a streaming chat completion request"""
        pass
