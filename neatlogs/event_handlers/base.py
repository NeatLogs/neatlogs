"""
Base Event Handler for NeatLogs Tracker
======================================

Provides the foundation for provider-specific event handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

from ..semconv import get_common_span_attributes
from ..token_counting import estimate_cost


class BaseEventHandler(ABC):
    """Base class for all LLM provider event handlers"""

    def __init__(self, tracker):
        self.tracker = tracker

    def create_span(self, model: str, provider: str, framework: str = None, operation: str = "llm_call") -> 'LLMSpan':
        """Create a new LLM span with standardized attributes"""
        from ..core import LLMSpan  # Import here to avoid circular dependency

        span = LLMSpan(
            session_id=self.tracker.session_id,
            agent_id=self.tracker.agent_id,
            thread_id=self.tracker.thread_id,
            api_key=self.tracker.api_key,
            model=model,
            provider=provider,
            framework=framework,
            tags=self.tracker.tags
        )

        # Set common attributes using semantic conventions
        common_attrs = get_common_span_attributes(
            session_id=self.tracker.session_id,
            agent_id=self.tracker.agent_id,
            thread_id=self.tracker.thread_id,
            model=model,
            provider=provider
        )

        span.start()

        return span

    def extract_request_params(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract request parameters from function arguments"""
        return {
            'temperature': kwargs.get('temperature'),
            'max_tokens': kwargs.get('max_tokens'),
            'top_p': kwargs.get('top_p'),
            'model': kwargs.get('model', 'unknown'),
        }

    @abstractmethod
    def extract_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        """Extract messages from function arguments - must be implemented by subclasses"""
        pass

    @abstractmethod
    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract data from response object - must be implemented by subclasses"""
        pass

    def handle_call_start(self, span: 'LLMSpan', *args, **kwargs):
        """Handle the start of an LLM call"""
        # Extract and set request parameters
        request_params = self.extract_request_params(*args, **kwargs)

        # Extract messages
        try:
            messages = self.extract_messages(*args, **kwargs)
            span.messages = messages
        except Exception as e:
            # Log but don't fail the entire operation
            import logging
            logging.warning(f"Failed to extract messages: {e}")

    def handle_call_end(self, span: 'LLMSpan', response: Any, success: bool = True, error: Optional[Exception] = None):
        """Handle the end of an LLM call"""
        if success and response:
            try:
                response_data = self.extract_response_data(response)

                # Update span with response data
                span.completion = response_data.get('completion', '')
                span.prompt_tokens = response_data.get('prompt_tokens', 0)
                span.completion_tokens = response_data.get(
                    'completion_tokens', 0)
                span.total_tokens = response_data.get('total_tokens', 0)
                span.cost = estimate_cost(
                    span.model, span.prompt_tokens, span.completion_tokens)

            except Exception as e:
                # Log extraction error but don't fail
                import logging
                logging.warning(f"Failed to extract response data: {e}")
                success = False
                error = e

        # End the span
        self.tracker.end_llm_span(span, success=success, error=error)

    def wrap_method(self, original_method, provider: str, framework: str = None):
        """Generic method wrapper for non-streaming LLM calls"""
        from functools import wraps
        from ..core import get_current_framework

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            # For LiteLLM, stream may be a kwarg in the main method
            if kwargs.get('stream', False):
                return self.wrap_stream_method(original_method, provider)(*args, **kwargs)

            model = kwargs.get('model', 'unknown')
            # Always use a local variable for framework to avoid UnboundLocalError
            _framework = framework
            if _framework is None:
                _framework = get_current_framework()

            span = self.create_span(
                model=model, provider=provider, framework=_framework)

            self.handle_call_start(span, *args, **kwargs)

            try:
                response = original_method(*args, **kwargs)
                self.handle_call_end(span, response, success=True)
                return response
            except Exception as e:
                self.handle_call_end(span, None, success=False, error=e)
                raise

        return wrapped

    # --- Streaming Placeholders ---
    # Subclasses should implement these if they have dedicated streaming methods

    def wrap_stream_method(self, original_method, provider: str):
        """Placeholder for wrapping a synchronous streaming method."""
        from functools import wraps
        from ..core import get_current_framework

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            # Get framework from thread-local context
            framework = get_current_framework()

            logging.warning(
                f"Streaming not implemented for {provider} in neatlogs. Calling original method.")
            return original_method(*args, **kwargs)
        return wrapped

    def wrap_async_stream_method(self, original_method, provider: str):
        """Placeholder for wrapping an asynchronous streaming method."""
        from functools import wraps
        from ..core import get_current_framework

        @wraps(original_method)
        async def wrapped(*args, **kwargs):
            # Get framework from thread-local context
            framework = get_current_framework()

            logging.warning(
                f"Async streaming not implemented for {provider} in neatlogs. Calling original method.")
            return await original_method(*args, **kwargs)
        return wrapped
