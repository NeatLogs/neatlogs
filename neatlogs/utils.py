"""
Utility functions for Neatlogs
=============================

This module contains utility functions and helpers used throughout Neatlogs,
including session ID generation, LLM API cost estimation, and statistics formatting
for telemetry and analysis. These utilities help abstract provider-specific
details and ensure consistent metrics and reporting across the system.
"""

import uuid
from typing import Dict


def generate_session_id() -> str:
    """
    Generate a globally unique identifier for the current Neatlogs session.

    Returns:
        str: A UUID4 string representing the unique Neatlogs session.
    """
    return str(uuid.uuid4())


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Provide an approximate cost (USD) for a single LLM API call,
    based on model and token usage.

    This function uses hardcoded, commonly published rates as a rough guide,
    and is not intended for precise billing. If the model is unknown,
    a default estimate is used.

    Args:
        model (str): The full or base model name (e.g., "gpt-3.5-turbo-0613").
        prompt_tokens (int): Number of tokens input to the model (prompt).
        completion_tokens (int): Number of tokens generated as output.

    Returns:
        float: Estimated call cost in USD.

    Note:
        - Model string is normalized to its base type for lookup.
        - Always verify costs with provider documentation before relying
          on output for any accounting or billing purposes.
    """
    cost_per_1k_tokens = {
        'gpt-4': {'prompt': 0.03, 'completion': 0.06},
        'gpt-3.5-turbo': {'prompt': 0.001, 'completion': 0.002},
        'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'claude-3-haiku': {'prompt': 0.00025, 'completion': 0.00125},
    }

    model_key = model.split('-')[0] + '-' + \
        model.split('-')[1] if '-' in model else model

    if model_key in cost_per_1k_tokens:
        pricing = cost_per_1k_tokens[model_key]
        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (completion_tokens / 1000) * pricing['completion']
        return prompt_cost + completion_cost
    else:
        # Fallback flat rate (conservative).
        return (prompt_tokens + completion_tokens) / 1000 * 0.002


def format_session_stats(stats: Dict) -> str:
    """
    Format session-level summary statistics as a human-readable string,
    suitable for console or message display.

    Args:
        stats (Dict): Dictionary containing standard session metrics. Expected keys:
            - 'session_id', 'total_calls', 'successful_calls',
              'failed_calls', 'total_cost', 'total_tokens', 'active_spans'

    Returns:
        str: Multiline string representing all session metrics.

    Example:
        stats = {
            'session_id': '...',
            'total_calls': 10,
            'successful_calls': 8,
            'failed_calls': 2,
            ...
        }
        print(format_session_stats(stats))

    Output:
        📊 Session Statistics:
           Session ID: ...
           Total Calls: ...
           Successful: ...
           Failed: ...
           Total Cost: $...
           Total Tokens: ...
           Active Spans: ...
    """
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
