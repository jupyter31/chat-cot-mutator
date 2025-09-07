# LLM Client Configuration

This document explains how to configure different LLM providers for the chat-dsat-mutator application.

## Overview

The application supports multiple LLM providers through a unified interface. You can easily switch between:

- **Microsoft Internal API** (default for Microsoft employees)
- **OpenAI API** (including Azure OpenAI)
- **Anthropic Claude API**

## Provider Setup

### 1. OpenAI

**Installation:**
```bash
pip install openai
```

**Environment variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Configuration:**
```python
# In chat_dsat_mutator_controller.py, replace the import:
from client_config import get_openai_client
llm_client = get_openai_client()
```

### 2. Azure OpenAI

**Installation:**
```bash
pip install openai
```

**Environment variables:**
```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

**Configuration:**
```python
# In chat_dsat_mutator_controller.py, replace the import:
from client_config import get_azure_openai_client
llm_client = get_azure_openai_client()
```

### 3. Anthropic Claude

**Installation:**
```bash
pip install anthropic
```

**Environment variables:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Configuration:**
```python
# In chat_dsat_mutator_controller.py, replace the import:
from client_config import get_anthropic_client
llm_client = get_anthropic_client()
```

### 4. Microsoft Internal (Current Default)

This is the current implementation that uses Microsoft's internal LLM API. It requires:
- Internal Microsoft dependencies (`msal`, etc.)
- Proper Microsoft authentication

## Making the Repository Public-Ready

To make this repository public while hiding internal Microsoft APIs:

### Option 1: Configuration-Based Approach (Recommended)

1. **Keep the current structure** but make the Microsoft client optional
2. **Set OpenAI as the default** for public users
3. **Use environment variables** for configuration

```python
# In chat_dsat_mutator_controller.py
import os
from clients.client_factory import get_default_client, create_llm_client

# For public version, prioritize public APIs
if os.getenv("OPENAI_API_KEY"):
    llm_client = create_llm_client("openai")
elif os.getenv("ANTHROPIC_API_KEY"):
    llm_client = create_llm_client("anthropic")
else:
    # Fallback to Microsoft internal if available (for internal users)
    try:
        llm_client = create_llm_client("microsoft")
    except ImportError:
        raise RuntimeError("No LLM client available. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
```

### Option 2: Remove Internal Dependencies

1. **Move Microsoft-specific files** to a separate private repository or branch
2. **Keep only the base interface** and public provider implementations
3. **Update requirements.txt** to include only public dependencies

### Option 3: Plugin Architecture

1. **Keep the base interface** in the public repo
2. **Make the Microsoft client** a separate private plugin
3. **Load providers dynamically** based on availability

## Public Repository Checklist

Before making the repository public:

- [ ] Remove or make optional all Microsoft-specific dependencies
- [ ] Update requirements.txt with public dependencies
- [ ] Add comprehensive README with setup instructions
- [ ] Include example configuration files
- [ ] Add proper error handling for missing dependencies
- [ ] Test with public APIs (OpenAI, Anthropic)
- [ ] Remove any hardcoded internal URLs or credentials
- [ ] Add license file
- [ ] Update documentation to focus on public APIs

## Example Public Requirements.txt

```txt
# Core dependencies
streamlit>=1.28.0
deepdiff>=6.0.0

# Optional LLM providers (user chooses based on their preference)
openai>=1.3.0  # For OpenAI API
anthropic>=0.8.0  # For Anthropic API

# Note: Microsoft internal dependencies (msal, etc.) are not included
# These will only be available in the internal version
```

## Migration Guide

To migrate existing internal usage to public APIs:

1. **Choose a provider** (OpenAI recommended for compatibility)
2. **Install dependencies**: `pip install openai`
3. **Set environment variables**: `export OPENAI_API_KEY="your-key"`
4. **Update configuration** using one of the examples above
5. **Test your specific use case** as different providers may have slight differences in behavior

## Model Compatibility

Different providers use different model names. Update your model references accordingly:

- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **Anthropic**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Azure OpenAI**: Use your deployed model names
