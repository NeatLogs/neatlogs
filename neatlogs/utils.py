"""
Utility functions for NeatLogs Tracker
=====================================

Common utilities and helper functions.
"""

import uuid
from typing import Dict


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an LLM API call based on model and token usage.
    
    Args:
        model: The model name
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    
    Returns:
        Estimated cost in USD
    """
    # Simplified cost estimation - real implementation would use actual pricing
    cost_per_1k_tokens = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
        'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125},
    }
    
    model_key = model.split('-')[0] + '-' + model.split('-')[1] if '-' in model else model
    
    if model_key in cost_per_1k_tokens:
        pricing = cost_per_1k_tokens[model_key]
        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (completion_tokens / 1000) * pricing['completion']
        return prompt_cost + completion_cost
    else:
        # Default estimation
        return (prompt_tokens + completion_tokens) / 1000 * 0.002


def format_session_stats(stats: Dict) -> str:
    """Format session statistics for display"""
    return f"""
📊 Session Statistics:
   Session ID: {stats['session_id']}
   Total Calls: {stats['total_calls']}
   Successful: {stats['successful_calls']}
   Failed: {stats['failed_calls']}
   Total Cost: ${stats['total_cost']:.6f}
   Total Tokens: {stats['total_tokens']}
   Active Spans: {stats['active_spans']}
"""
