# LLM Client Configuration
# 
# This file demonstrates how to configure different LLM providers for the chat-dsat-mutator.
# You can modify the client initialization in chat_mutator_controller.py to use these
# configurations instead of the default client.

import os
from clients.client_factory import create_llm_client

# Example configurations for different providers

def get_openai_client():
    """
    Get OpenAI client configuration
    
    Requirements:
    - pip install openai
    - Set OPENAI_API_KEY environment variable
    
    Example usage in controller:
        from client_config import get_openai_client
        llm_client = get_openai_client()
    """
    return create_llm_client(
        "openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url="https://api.openai.com/v1"  # Optional: custom endpoint
    )


def get_anthropic_client():
    """
    Get Anthropic client configuration
    
    Requirements:
    - pip install anthropic
    - Set ANTHROPIC_API_KEY environment variable
    
    Example usage in controller:
        from client_config import get_anthropic_client
        llm_client = get_anthropic_client()
    """
    return create_llm_client(
        "anthropic",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )


def get_azure_openai_client():
    """
    Get Azure OpenAI client configuration
    
    Requirements:
    - pip install openai
    - Set AZURE_OPENAI_API_KEY environment variable
    - Set AZURE_OPENAI_ENDPOINT environment variable
    
    Example usage in controller:
        from client_config import get_azure_openai_client
        llm_client = get_azure_openai_client()
    """
    return create_llm_client(
        "openai",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
    )


def get_microsoft_internal_client():
    """
    Get Microsoft internal client configuration
    
    Requirements:
    - Internal Microsoft dependencies (msal, etc.)
    - Proper authentication setup
    
    This is the current default client used in the application.
    """
    return create_llm_client(
        "microsoft",
        endpoint=None  # Uses default internal endpoint
    )


def get_phi2_client():
    """
    Get Phi-2 client configuration (runs locally)
    
    Requirements:
    - pip install torch transformers
    - ~6GB of GPU memory or CPU with sufficient RAM
    
    Example usage in controller:
        from client_config import get_phi2_client
        llm_client = get_phi2_client()
    """
    return create_llm_client(
        "phi",
        model_name="phi-2",
        device="auto"  # Will use GPU if available, otherwise CPU
    )


def get_phi3_client(model_size="mini"):
    """
    Get Phi-3 client configuration (runs locally)
    
    Args:
        model_size: "mini" (4K context), "small" (8K context), or "medium" (4K context)
    
    Requirements:
    - pip install torch transformers
    - 8-16GB+ GPU memory depending on model size
    
    Example usage in controller:
        from client_config import get_phi3_client
        llm_client = get_phi3_client("mini")  # or "small", "medium"
    """
    return create_llm_client(
        "phi",
        model_name="phi-3",
        model_size=model_size,
        device="auto"
    )


def get_custom_huggingface_client(model_name="microsoft/DialoGPT-medium"):
    """
    Get a custom Hugging Face model client
    
    Args:
        model_name: Any Hugging Face causal language model
    
    Requirements:
    - pip install torch transformers
    - Sufficient memory for the chosen model
    
    Example usage in controller:
        from client_config import get_custom_huggingface_client
        llm_client = get_custom_huggingface_client("microsoft/DialoGPT-large")
    """
    return create_llm_client(
        "huggingface",
        model_name=model_name,
        device="auto"
    )


# Default client selection priority:
# 1. Microsoft internal (for internal Microsoft users)
# 2. OpenAI (if OPENAI_API_KEY is set)
# 3. Anthropic (if ANTHROPIC_API_KEY is set)
# 
# To use a specific client, modify the import in chat_mutator_controller.py:
# 
# Replace:
#   from clients.client_factory import get_default_client
#   llm_client = get_default_client()
# 
# With one of:
#   from client_config import get_openai_client
#   llm_client = get_openai_client()
#   
#   from client_config import get_anthropic_client
#   llm_client = get_anthropic_client()
#   
#   from client_config import get_azure_openai_client
#   llm_client = get_azure_openai_client()
