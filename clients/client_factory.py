"""
LLM Client Factory

This module provides a factory function to create LLM clients based on configuration.
It allows for easy switching between different LLM providers.

Usage:
    # Using OpenAI
    client = create_llm_client("openai", api_key="your-key")
    
    # Using Anthropic
    client = create_llm_client("anthropic", api_key="your-key")
    
    # Using internal Microsoft API (if available)
    client = create_llm_client("microsoft", endpoint="your-endpoint")
"""

from typing import Dict, Any
from .base_llm_client import BaseLLMClient

def create_llm_client(provider: str, **kwargs) -> BaseLLMClient:
    """
    Factory function to create LLM clients
    
    Args:
        provider: The LLM provider ("openai", "anthropic", "microsoft")
        **kwargs: Provider-specific configuration
        
    Returns:
        BaseLLMClient: An instance of the appropriate LLM client
        
    Raises:
        ValueError: If provider is not supported
        ImportError: If required dependencies are not installed
    """
    
    if provider == "openai":
        try:
            from .openai_client import OpenAIClient
            return OpenAIClient(**kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI client requires 'openai' package. Install with: pip install openai"
            )
    
    elif provider == "anthropic":
        try:
            from .anthropic_client import AnthropicClient
            return AnthropicClient(**kwargs)
        except ImportError:
            raise ImportError(
                "Anthropic client requires 'anthropic' package. Install with: pip install anthropic"
            )
    
    elif provider == "microsoft":
        try:
            from .llm_api import MicrosoftLLMClient
            return MicrosoftLLMClient(**kwargs)
        except ImportError:
            raise ImportError(
                "Microsoft client requires internal dependencies and authentication"
            )
    
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, anthropic, microsoft"
        )


# For backward compatibility, create a default client
def get_default_client() -> BaseLLMClient:
    """
    Get the default LLM client based on available providers and configuration.
    
    Priority:
    1. Microsoft client (if internal dependencies available)
    2. OpenAI client (if API key in environment)
    3. Anthropic client (if API key in environment)
    """
    import os
    
    # Try Microsoft client first (for internal use)
    try:
        return create_llm_client("microsoft")
    except ImportError:
        pass
    
    # Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            return create_llm_client("openai")
        except ImportError:
            pass
    
    # Try Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return create_llm_client("anthropic")
        except ImportError:
            pass
    
    raise RuntimeError(
        "No LLM client available. Please install required packages and set API keys:\n"
        "- OpenAI: pip install openai && export OPENAI_API_KEY=your_key\n"
        "- Anthropic: pip install anthropic && export ANTHROPIC_API_KEY=your_key"
    )
