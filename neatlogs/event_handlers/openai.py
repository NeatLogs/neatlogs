"""
OpenAI Event Handler
===================

Handles OpenAI specific API patterns and response formats.
Enhanced with comprehensive tracking including streaming, tools, and legacy API support.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from .base import BaseEventHandler
from ..token_counting import estimate_cost
from ..stream_wrapper import NeatlogsStreamWrapper


class OpenAIHandler(BaseEventHandler):
    """Event handler for OpenAI, with streaming, tool, and legacy API support."""

    def extract_request_params(self, *args, **kwargs) -> Dict[str, Any]:
        params = super().extract_request_params(*args, **kwargs)
        # The 'parse' method has the model in the top-level kwargs
        if 'model' in kwargs:
            params['model'] = kwargs['model']
        params.update({
            'tools': kwargs.get('tools'),
            'tool_choice': kwargs.get('tool_choice'),
            'frequency_penalty': kwargs.get('frequency_penalty'),
            'presence_penalty': kwargs.get('presence_penalty'),
            'seed': kwargs.get('seed'),
        })
        return params

    def extract_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        """Extract messages from OpenAI function arguments"""
        return kwargs.get('messages', [])

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract data from OpenAI response object (both new and old SDKs)"""
        data = {
            'completion': '',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'model': None,
            'finish_reason': None,
            'tool_calls': []
        }

        try:
            if not hasattr(response, 'choices') or not response.choices:
                return data

            choice = response.choices[0]

            # Extract completion text/message
            if hasattr(choice, 'message') and choice.message:
                message = choice.message
                data['completion'] = getattr(message, 'content', '') or ''

                # Extract tool calls (new SDK)
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    data['tool_calls'] = [
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "type": "function",
                            "arguments": call.function.arguments
                        }
                        for call in message.tool_calls
                    ]
            elif hasattr(choice, 'text'):  # Legacy completion
                data['completion'] = choice.text

            # Extract usage information
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                data['prompt_tokens'] = getattr(usage, 'prompt_tokens', 0)
                data['completion_tokens'] = getattr(
                    usage, 'completion_tokens', 0)
                data['total_tokens'] = getattr(usage, 'total_tokens', 0)

            # Extract model information
            if hasattr(response, 'model'):
                data['model'] = response.model

            # Extract finish reason
            if hasattr(choice, 'finish_reason'):
                data['finish_reason'] = choice.finish_reason

        except Exception as e:
            logging.warning(f"Error extracting OpenAI response data: {e}")

        return data

    def handle_call_start(self, span: 'LLMSpan', *args, **kwargs):
        super().handle_call_start(span, *args, **kwargs)

    # --- Advanced Streaming Support ---

    def wrap_stream_method(self, original_method, provider: str):
        from functools import wraps
        from ..core import current_span_id_context

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            model = kwargs.get('model', 'unknown')
            span = self.create_span(
                model=model, provider=provider, operation="llm_stream", node_type="llm_call", node_name=model)

            self.handle_call_start(span, *args, **kwargs)

            token = current_span_id_context.set(span.span_id)
            try:
                stream = original_method(*args, **kwargs)
                # Use NeatlogsStreamWrapper for proper telemetry collection
                return NeatlogsStreamWrapper(stream, span, kwargs, context_token=token)
            except Exception as e:
                current_span_id_context.reset(token)
                self.handle_call_end(span, None, success=False, error=e)
                raise

        return wrapped

    def wrap_async_stream_method(self, original_method, provider: str):
        from functools import wraps
        from ..core import current_span_id_context

        @wraps(original_method)
        async def wrapped(*args, **kwargs):
            model = kwargs.get('model', 'unknown')
            span = self.create_span(
                model=model, provider=provider, operation="llm_stream_async", node_type="llm_call", node_name=model)

            self.handle_call_start(span, *args, **kwargs)

            token = current_span_id_context.set(span.span_id)
            try:
                stream = await original_method(*args, **kwargs)
                # Use NeatlogsStreamWrapper for proper telemetry collection
                return NeatlogsStreamWrapper(stream, span, kwargs, context_token=token)
            except Exception as e:
                current_span_id_context.reset(token)
                self.handle_call_end(span, None, success=False, error=e)
                raise

        return wrapped
