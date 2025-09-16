"""
Base Event Handler for Neatlogs
==============================
This module defines an abstract base class for LLM provider event handlers used by
Neatlogs. It provides a unified interface for extracting and managing telemetry
from various LLM APIs (OpenAI, Anthropic, Google, etc.) and frameworks, and is
intended to be subclassed for provider-specific logic.

Responsibilities:
    - Manage LLM span life cycles (start/end tracking)
    - Extract request and response data in a consistent, provider-agnostic way
    - Provide hooks for synchronous and asynchronous API calls, including streaming

All provider handler implementations in Neatlogs should inherit from BaseEventHandler.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

from ..semconv import get_common_span_attributes
from ..token_counting import estimate_cost


class BaseEventHandler(ABC):
    """
    Abstract base class for all Neatlogs LLM provider event handlers.

    Defines the public interface and common logic for extracting, normalizing,
    and tracking LLM API calls and responses. All subclasses must implement message
    and response extractors suited for their provider/API.

    Instantiated by subclasses in the patchers layer; not used directly.
    """

    def __init__(self, tracker):
        self.tracker = tracker

    def create_span(self, model: str, provider: str, framework: str = None, operation: str = "llm_call", node_type: str = "llm_call", node_name: str = None) -> 'LLMSpan':
        """
        Create a new LLMSpan object pre-filled with standard metadata for Neatlogs.

        Args:
            model (str): The model invoked (e.g., "gpt-3.5-turbo").
            provider (str): Provider name (e.g., "openai", "anthropic").
            framework (str, optional): Active agentic framework, if applicable.
            operation (str): The telemetry operation type. Defaults to "llm_call".
            node_type (str): The type of node for graph visualization.
            node_name (str): The human-readable name for the node.

        Returns:
            LLMSpan: New span instance with common attributes set for tracking.

        Note:
            This method initializes and registers the span with the tracker.
        """
        span = self.tracker.start_llm_span(
            model=model,
            provider=provider,
            framework=framework,
            node_type=node_type,
            node_name=node_name or model
        )
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

    def enrich_span(self, span: 'LLMSpan', response: Any):
        """Enriches a span with data from an LLM response, without ending it."""
        logging.debug(f"Neatlogs: enrich_span called for span {span.span_id}")
        if not response:
            return
        try:
            response_data = self.extract_response_data(response)
            logging.debug(
                f"Neatlogs: Extracted response data: {response_data}")

            # Update span with response data
            if response_data.get('model'):
                span.model = response_data.get('model')
                logging.debug(f"Neatlogs: Set span.model to {span.model}")
            span.completion = response_data.get('completion', '')
            span.prompt_tokens = response_data.get('prompt_tokens', 0)
            span.completion_tokens = response_data.get('completion_tokens', 0)
            span.total_tokens = response_data.get('total_tokens', 0)
            span.cost = estimate_cost(
                span.model, span.prompt_tokens, span.completion_tokens)

        except Exception as e:
            logging.warning(f"Neatlogs: Failed to enrich span data: {e}")

    def handle_call_end(self, span: 'LLMSpan', response: Any, success: bool = True, error: Optional[Exception] = None):
        """Handle the end of an LLM call by enriching and then ending the span."""
        if success and response:
            self.enrich_span(span, response)

        # End the span
        self.tracker.end_llm_span(span, success=success, error=error)

    def wrap_method(self, original_method, provider: str, framework: str = None):
        """Generic method wrapper for non-streaming LLM calls"""
        from functools import wraps
        from ..core import get_current_framework, is_patching_suppressed, get_active_langgraph_node_span, current_span_id_context

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            # Priority 1: Check if we are inside a LangGraph node that wants to be enriched.
            active_node_span = get_active_langgraph_node_span()
            if active_node_span:
                try:
                    response = original_method(*args, **kwargs)
                    self.enrich_span(active_node_span, response)
                    return response
                except Exception as e:
                    if active_node_span.error_report is None:
                        active_node_span.error_report = {
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    raise

            # Priority 2: Check if a framework (like LangChain's callback) wants to suppress us completely.
            if is_patching_suppressed():
                return original_method(*args, **kwargs)

            # For LiteLLM, stream may be a kwarg in the main method
            if kwargs.get('stream', False):
                return self.wrap_stream_method(original_method, provider)(*args, **kwargs)

            # Priority 3: Default behavior - create a new span for this call.
            model = kwargs.get('model', 'unknown')
            _framework = framework or get_current_framework()
            span = self.create_span(
                model=model, provider=provider, framework=_framework, node_type="llm_call", node_name=model)

            token = current_span_id_context.set(span.span_id)
            self.handle_call_start(span, *args, **kwargs)
            try:
                response = original_method(*args, **kwargs)
                self.handle_call_end(span, response, success=True)
                return response
            except Exception as e:
                self.handle_call_end(span, None, success=False, error=e)
                raise
            finally:
                current_span_id_context.reset(token)

        return wrapped

    def wrap_async_method(self, original_method, provider: str, framework: str = None):
        """Generic method wrapper for non-streaming async LLM calls"""
        from functools import wraps
        from ..core import get_current_framework, is_patching_suppressed, get_active_langgraph_node_span, current_span_id_context

        @wraps(original_method)
        async def wrapped(*args, **kwargs):
            # Priority 1: Check if we are inside a LangGraph node.
            active_node_span = get_active_langgraph_node_span()
            if active_node_span:
                try:
                    response = await original_method(*args, **kwargs)
                    self.enrich_span(active_node_span, response)
                    return response
                except Exception as e:
                    if active_node_span.error_report is None:
                        active_node_span.error_report = {
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    raise

            # Priority 2: Check for general framework suppression.
            if is_patching_suppressed():
                return await original_method(*args, **kwargs)

            if kwargs.get('stream', False):
                return self.wrap_async_stream_method(original_method, provider)(*args, **kwargs)

            # Priority 3: Default behavior.
            model = kwargs.get('model', 'unknown')
            _framework = framework or get_current_framework()
            span = self.create_span(
                model=model, provider=provider, framework=_framework, node_type="llm_call", node_name=model)

            token = current_span_id_context.set(span.span_id)
            self.handle_call_start(span, *args, **kwargs)
            try:
                response = await original_method(*args, **kwargs)
                self.handle_call_end(span, response, success=True)
                return response
            except Exception as e:
                self.handle_call_end(span, None, success=False, error=e)
                raise
            finally:
                current_span_id_context.reset(token)

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
