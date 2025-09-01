"""
LangChain event handler for Neatlogs Tracker
===========================================

This implementation uses LangChain's official callback system to provide robust
and comprehensive tracking of all LangChain operations.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..core import LLMSpan


class NeatlogsLangchainCallbackHandler(BaseCallbackHandler):
    """
    Neatlogs Callback Handler for LangChain.

    This handler captures events from LangChain and logs them to Neatlogs Tracker.
    It is designed to work when `enable_langchain=True` is set during `neatlogs.init()`.
    """

    def __init__(self, tracker=None):
        if tracker:
            self.tracker = tracker
        else:
            from .. import get_tracker
            self.tracker = get_tracker()

        if not self.tracker:
            logging.warning(
                "Neatlogs Tracker not initialized. LangChain calls will not be tracked.")

        self.active_spans: Dict[UUID, LLMSpan] = {}

    def _start_span(self, run_id: UUID, provider: str, model: str, attributes: Dict[str, Any]) -> None:
        if not self.tracker:
            return

        # Create a new span with the extracted provider and model
        span = self.tracker.start_llm_span(
            model=model, provider=provider, framework="langchain")

        for key, value in attributes.items():
            setattr(span, key, value)

        self.active_spans[run_id] = span

    def _end_span(self, run_id: UUID, success: bool = True, error: Optional[Exception] = None) -> None:
        if not self.tracker or run_id not in self.active_spans:
            return

        span = self.active_spans.pop(run_id)
        self.tracker.end_llm_span(span, success=success, error=error)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            # --- Corrected Provider and Model Extraction ---
            # Provider is the class name of the LLM
            provider_name = serialized.get('name', 'LangChainLLM')

            # Model is the specific model/deployment name from invocation parameters
            invocation_params = kwargs.get('invocation_params', {})
            model_name = invocation_params.get(
                'model') or invocation_params.get('deployment_name')

            # Fallback to other common keys if the primary ones are not found
            if not model_name:
                model_name = serialized.get(
                    'model_name', serialized.get('model', 'unknown'))

            attributes = {
                "prompts": prompts,
                "messages": prompts  # For consistency in logging
            }
            self._start_span(run_id, provider=provider_name,
                             model=model_name, attributes=attributes)
        except Exception as e:
            logging.error(
                f"Error in Neatlogs on_llm_start: {e}", exc_info=True)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            if run_id not in self.active_spans:
                return
            span = self.active_spans[run_id]

            if response.generations:
                completions = []
                for gen_list in response.generations:
                    for gen in gen_list:
                        completions.append(gen.text)
                span.completion = "\n".join(completions)

            if response.llm_output and 'token_usage' in response.llm_output:
                usage = response.llm_output['token_usage']
                span.prompt_tokens = usage.get('prompt_tokens', 0)
                span.completion_tokens = usage.get('completion_tokens', 0)
                span.total_tokens = usage.get('total_tokens', 0)

            self._end_span(run_id)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_llm_end: {e}", exc_info=True)

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self._end_span(run_id, success=False, error=error)
        except Exception as e:
            logging.error(
                f"Error in Neatlogs on_llm_error: {e}", exc_info=True)

    # --- Simplified Handlers for other events to reduce noise ---

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, action, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish, *, run_id: UUID, **kwargs: Any) -> None:
        pass
