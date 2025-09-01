"""
Event Handlers for Neatlogs Tracker
=================================

Provider-specific event handlers for different LLM services.
Each provider has its own handler to manage the specific API patterns and response formats.
Enhanced with comprehensive tracking and streaming support.
"""

from .base import BaseEventHandler
from .google_genai import GoogleGenAIHandler
from .litellm import LiteLLMHandler
from .openai import OpenAIHandler
from .anthropic import AnthropicHandler
from .azure import AzureOpenAIHandler

__all__ = [
    'BaseEventHandler',
    'GoogleGenAIHandler',
    'LiteLLMHandler',
    'OpenAIHandler',
    'AnthropicHandler',
    'AzureOpenAIHandler',
]

# Provider registry
PROVIDER_HANDLERS = {
    'google': GoogleGenAIHandler,
    'google_genai': GoogleGenAIHandler,
    'gemini': GoogleGenAIHandler,
    'litellm': LiteLLMHandler,
    'openai': OpenAIHandler,
    'gpt': OpenAIHandler,
    'anthropic': AnthropicHandler,
    'claude': AnthropicHandler,
    'azure': AzureOpenAIHandler,
    'azure_openai': AzureOpenAIHandler,
}


def get_langchain_handler(tracker):
    """Lazily import and return the LangChain handler"""
    from .langchain import NeatlogsLangchainCallbackHandler as LangChainHandler
    return LangChainHandler(tracker)


def get_handler_for_provider(provider: str, tracker) -> BaseEventHandler:
    """Get the appropriate event handler for a provider"""
    # Special handling for LangChain to avoid importing it unnecessarily
    if provider.lower() == 'langchain':
        return get_langchain_handler(tracker)

    handler_class = PROVIDER_HANDLERS.get(provider.lower())
    if handler_class:
        return handler_class(tracker)
    else:
        # Fallback to base handler
        return BaseEventHandler(tracker)
