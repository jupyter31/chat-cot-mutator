# LLM Client Configuration
# 
# This file demonstrates how to configure different LLM providers for the chat-dsat-mutator.
# You can modify the client initialization in chat_dsat_mutator_controller.py to use these
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


# Default client selection priority:
# 1. Microsoft internal (for internal Microsoft users)
# 2. OpenAI (if OPENAI_API_KEY is set)
# 3. Anthropic (if ANTHROPIC_API_KEY is set)
# 
# To use a specific client, modify the import in chat_dsat_mutator_controller.py:
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
