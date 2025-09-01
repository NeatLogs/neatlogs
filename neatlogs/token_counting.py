"""
Token usage extraction and utilities for Neatlogs.
=================================================

This module provides utility classes for extracting token usage from LLM responses
and for performing cost estimation for various models and provider APIs.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """
    Data structure for LLM token usage extraction.

    Contains prompt, completion, and total token counts extracted from
    LLM provider responses for usage/cost computation.
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, int]:
        attributes = {}
        if self.prompt_tokens is not None:
            attributes['prompt_tokens'] = self.prompt_tokens
        if self.completion_tokens is not None:
            attributes['completion_tokens'] = self.completion_tokens
        if self.total_tokens is not None:
            attributes['total_tokens'] = self.total_tokens
        return attributes


class TokenUsageExtractor:
    """
    Extracts token usage from LLM provider responses.

    Attempts to extract prompt, completion, and total token usage
    consistently across OpenAI, Anthropic, and other APIs.
    """
    @staticmethod
    def extract_from_response(response: Any) -> TokenUsage:
        # Try common usage attributes
        usage = getattr(response, 'usage', None)
        if usage:
            prompt = getattr(usage, 'prompt_tokens', None)
            completion = getattr(usage, 'completion_tokens', None)
            total = getattr(usage, 'total_tokens', None)
            return TokenUsage(prompt, completion, total)

        # Try usage_metadata (Anthropic style)
        usage_meta = getattr(response, 'usage_metadata', None)
        if usage_meta:
            prompt = getattr(usage_meta, 'prompt_tokens', None)
            completion = getattr(usage_meta, 'completion_tokens', None)
            total = getattr(usage_meta, 'total_tokens', None)
            return TokenUsage(prompt, completion, total)

        # Try direct attributes on response
        prompt = getattr(response, 'prompt_tokens', None)
        completion = getattr(response, 'completion_tokens', None)
        total = getattr(response, 'total_tokens', None)
        if prompt is not None or completion is not None or total is not None:
            return TokenUsage(prompt, completion, total)

        # Fallback: no usage info found
        return TokenUsage()


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an LLM API call based on model and token usage.

    Args:
        model (str): The model name
        prompt_tokens (int): Number of input tokens
        completion_tokens (int): Number of output tokens

    Returns:
        float: Estimated cost in USD.
    """
    # Handle None or non-string model values
    if not model or not isinstance(model, str):
        # Default estimation for unknown models
        return (prompt_tokens + completion_tokens) / 1000000 * 0.001

    cost_per_1k_tokens = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
        'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125},
        # $0.075 per 1M input, $0.30 per 1M output
        'gemini-1.5-flash': {'prompt': 0.075, 'completion': 0.30},
        # $0.50 per 1M input, $1.50 per 1M output
        'gemini-1.5-pro': {'prompt': 0.50, 'completion': 1.50},
        # Updated pricing for Gemini 2.0
        'gemini-2.0-flash': {'prompt': 0.10, 'completion': 0.40},
    }
    model_key = model.lower()
    for key in cost_per_1k_tokens:
        if key in model_key:
            pricing = cost_per_1k_tokens[key]
            # Convert per-1M pricing to per-1k
            prompt_cost = (prompt_tokens / 1000000) * pricing['prompt']
            completion_cost = (completion_tokens / 1000000) * \
                pricing['completion']
            return prompt_cost + completion_cost
    # Default estimation
    return (prompt_tokens + completion_tokens) / 1000000 * 0.001
