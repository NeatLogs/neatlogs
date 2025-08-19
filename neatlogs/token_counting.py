"""Token usage extraction and utilities for NeatLogs Tracker."""

from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class TokenUsage:
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
    cost_per_1k_tokens = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
        'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'gemini-1.5-flash': {'prompt': 0.00001, 'completion': 0.00001},
    }
    model_key = model.lower()
    for key in cost_per_1k_tokens:
        if key in model_key:
            pricing = cost_per_1k_tokens[key]
            prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
            completion_cost = (completion_tokens / 1000) * pricing['completion']
            return prompt_cost + completion_cost
    # Default estimation
    return (prompt_tokens + completion_tokens) / 1000 * 0.000001
