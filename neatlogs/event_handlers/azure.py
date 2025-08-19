"""
Azure OpenAI Event Handler
=========================

Handles Azure OpenAI specific API patterns, inheriting from the OpenAI handler.
"""

from .openai import OpenAIHandler


class AzureOpenAIHandler(OpenAIHandler):
    """Event handler for Azure OpenAI, reusing OpenAI logic."""
    pass

