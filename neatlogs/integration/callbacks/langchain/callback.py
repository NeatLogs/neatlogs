"""
Neatlogs LangChain Callback Handlers
====================================
This module contains both synchronous and asynchronous callback handlers for integrating
Neatlogs telemetry with LangChain workflows. Use these handlers to track LLM usage,
agent and tool activity, and cost telemetry with minimal code changes.

Handler selection:
    - Use `NeatlogsLangchainCallbackHandler` for synchronous LangChain pipelines (normal `chain.run()` etc).
    - Use `AsyncNeatlogsLangchainCallbackHandler` for asynchronous pipelines (`await chain.arun()` or other async APIs).
    - Do not mix handler types within the same call. For mixed-mode applications,
      provide the appropriate handler to each API as needed.

Both handlers ensure that Neatlogs captures all critical runtime and resource metrics,
including prompt/completion tokens, agent/tool activity, and chained workflows.

"""
import logging
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
import json

from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

from ....core import LLMSpan, suppress_patching, release_patching, get_tracker


def safe_serialize(data: Any) -> str:
    """Helper to safely serialize data that might not be JSON-serializable."""
    try:
        if isinstance(data, (str, int, float, bool, type(None))):
            return str(data)
        if hasattr(data, 'dict'):
            return json.dumps(data.dict())
        return json.dumps(str(data))
    except (TypeError, OverflowError):
        return str(data)


def get_model_info(serialized: Dict[str, Any]) -> Dict[str, str]:
    """Extract model information from serialized LangChain data."""
    if not serialized:
        return {"model_name": "unknown", "provider": "langchain"}

    model_info = {"model_name": "unknown", "provider": "langchain"}

    try:
        # Provider is usually the class name from the ID
        if 'id' in serialized and isinstance(serialized['id'], list) and serialized['id']:
            model_info["provider"] = serialized['id'][-1]

        # Model name is in invocation_params
        if 'invocation_params' in serialized:
            params = serialized['invocation_params']
            if 'model' in params:
                model_info["model_name"] = params['model']
            elif 'model_name' in params:
                model_info["model_name"] = params['model_name']
            elif 'deployment_name' in params:
                model_info["model_name"] = params['deployment_name']
            elif 'azure_deployment' in params:  # older azure versions
                model_info["model_name"] = params['azure_deployment']

        # Fallback for model name from kwargs
        if model_info["model_name"] == "unknown" and 'kwargs' in serialized and isinstance(serialized['kwargs'], dict):
            kwargs = serialized['kwargs']
            if 'model_name' in kwargs:
                model_info["model_name"] = kwargs['model_name']
            elif 'model' in kwargs:
                model_info["model_name"] = kwargs['model']

        # Fallback to top-level model_name
        if model_info["model_name"] == "unknown" and 'model_name' in serialized:
            model_info["model_name"] = serialized['model_name']

        # If model_name is still unknown, and provider was identified, use provider as model_name
        if model_info["model_name"] == "unknown" and model_info["provider"] != "langchain":
            model_info["model_name"] = model_info["provider"]

    except Exception as e:
        logging.warning(f"Error extracting model info: {e}")

    return model_info


def should_track_span(operation: str, attributes: Dict[str, Any]) -> bool:
    """Determine if a span should be tracked based on operation type and attributes."""
    # Always track LLM calls
    if operation == "llm":
        return True

    # Always track tool calls and agent actions
    if operation in ["tool", "agent_action"]:
        return True

    # For chains, be more selective about what to track
    if operation == "chain":
        chain_name = attributes.get("chain_name", "")

        # Skip these common intermediate/utility chains that don't add meaningful value
        skip_chains = [
            "PromptTemplate", "StrOutputParser", "RunnableParallel", "RunnablePassthrough",
            "RunnableSequence", "RunnableLambda", "RunnableBinding", "RunnableWithFallbacks",
            "unknown_chain", "RunnableAssign", "RunnableMap", "RunnableBranch",
            "RunnableRouter", "RunnableSwitch"
        ]

        # Skip if chain name matches any skip patterns
        if any(skip in chain_name for skip in skip_chains):
            return False

        # Also skip if chain name is generic or looks auto-generated
        if (chain_name == "unknown_chain" or
            chain_name.startswith("RunnableSequence") or
            chain_name.startswith("Runnable") or
                len(chain_name) < 3):
            return False

        # If we get here, it's likely a meaningful chain (like an Agent, custom chain, etc.)
        return True

    return True


class NeatlogsLangchainCallbackHandler(BaseCallbackHandler):
    """
    Synchronous Neatlogs callback handler for LangChain.

    Attach this handler to a LangChain workflow to enable Neatlogs instrumentation
    of LLM, chain, agent, and tool calls made via *synchronous* APIs (blocking Python).

    Use this handler if your pipeline is synchronous (using methods such as `chain.run()`).
    Instantiate and pass as a callback:
        handler = NeatlogsLangchainCallbackHandler(...)
        chain = LLMChain(..., callbacks=[handler])

    If any part of your workflow is asynchronous (i.e., uses `async def`/`await`),
    use AsyncNeatlogsLangchainCallbackHandler for those APIs instead.
    """

    def __init__(self, api_key: Optional[str] = None, tags: Optional[List[str]] = None):
        from ....core import LLMTracker
        tracker = get_tracker()
        # If there's no global tracker, create a temporary one just for this callback handler
        # This allows the callback handler to work independently without triggering automatic patching
        if not tracker and api_key:
            # Create a temporary tracker that sends data to the server
            # Since there's no global tracker, there's no conflict
            tracker = LLMTracker(api_key=api_key, tags=tags,
                                 enable_server_sending=True)
            logging.info(
                "Neatlogs: Created temporary tracker for LangChain callback handler")
        elif not tracker:
            logging.warning(
                "Neatlogs Tracker not initialized. Please call neatlogs.init(api_key=...) to enable automatic patching, "
                "or ensure a tracker is already initialized.")

        self.tracker = tracker
        self._api_key = api_key
        self._tags = tags or []

        self.active_spans: Dict[UUID, LLMSpan] = {}

    def _start_span(self, run_id: UUID, parent_run_id: Optional[UUID], operation: str, attributes: Dict[str, Any]) -> None:
        if not self.tracker or run_id in self.active_spans:
            if run_id in self.active_spans:
                logging.debug(
                    f"Span for run_id {run_id} already exists. Ignoring duplicate start event.")
            return

        if not should_track_span(operation, attributes):
            logging.debug(
                f"Skipping span for operation {operation} with attributes {attributes}")
            return

        model = attributes.get("model", f"langchain_{operation}")
        provider = attributes.get("provider", "langchain")

        span = self.tracker.start_llm_span(
            model=model, provider=provider, framework="langchain")

        for key, value in attributes.items():
            setattr(span, key, value)

        if parent_run_id and parent_run_id in self.active_spans:
            parent_span = self.active_spans[parent_run_id]
            setattr(span, 'parent_span_id', parent_span.span_id)

        self.active_spans[run_id] = span

    def _end_span(self, run_id: UUID, success: bool = True, error: Optional[Exception] = None) -> None:
        if not self.tracker or run_id not in self.active_spans:
            return

        span = self.active_spans.pop(run_id)
        self.tracker.end_llm_span(span, success=success, error=error)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        suppress_patching()
        try:
            serialized = serialized or {}
            model_info = get_model_info(serialized)
            model_name = model_info.get("model_name", "unknown")
            provider = model_info.get("provider", "langchain")

            attributes = {
                "model": model_name,
                "provider": provider,
                "prompts": prompts,
                "invocation_params": serialized.get('invocation_params', {})
            }
            self._start_span(run_id, parent_run_id, "llm", attributes)

            span = self.active_spans.get(run_id)
            if span:
                messages = []
                for prompt in prompts:
                    if isinstance(prompt, str):
                        if prompt.startswith("Human: "):
                            messages.append(
                                {"role": "user", "content": prompt[7:]})
                        elif prompt.startswith("System: "):
                            messages.append(
                                {"role": "system", "content": prompt[8:]})
                        else:
                            messages.append(
                                {"role": "user", "content": prompt})
                    else:
                        messages.append(
                            {"role": "user", "content": str(prompt)})
                span.messages = messages
        except Exception as e:
            logging.error(f"Error in Neatlogs on_llm_start: {e}")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            if run_id not in self.active_spans:
                return
            span = self.active_spans[run_id]

            if response.generations:
                completions = [
                    gen.text for gen_list in response.generations for gen in gen_list]
                span.completion = "\n".join(completions)

            if response.llm_output and 'token_usage' in response.llm_output:
                usage = response.llm_output['token_usage']
                span.prompt_tokens = usage.get('prompt_tokens', 0)
                span.completion_tokens = usage.get('completion_tokens', 0)
                span.total_tokens = usage.get('total_tokens', 0)

            self._end_span(run_id)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_llm_end: {e}")
        finally:
            release_patching()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        suppress_patching()
        try:
            serialized = serialized or {}
            model_info = get_model_info(serialized)
            model_name = model_info.get("model_name", "unknown")
            provider = model_info.get("provider", "langchain")

            attributes = {
                "model": model_name,
                "provider": provider,
                "invocation_params": serialized.get('invocation_params', {})
            }
            self._start_span(run_id, parent_run_id, "llm", attributes)

            span = self.active_spans.get(run_id)
            if span:
                # messages is a list of lists of BaseMessage. We handle the first list.
                if messages and messages[0]:
                    span.messages = [
                        {"role": msg.type, "content": msg.content}
                        for msg in messages[0]
                    ]
        except Exception as e:
            logging.error(f"Error in Neatlogs on_chat_model_start: {e}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            serialized = serialized or {}
            chain_name = serialized.get("name", "unknown_chain")
            attributes = {
                "model": f"chain_{chain_name}",
                "chain_name": chain_name,
                "inputs": safe_serialize(inputs),
            }
            if should_track_span("chain", attributes):
                self._start_span(run_id, parent_run_id, "chain", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_chain_start: {e}")

    def on_chain_end(self, outputs: Union[Dict[str, Any], str], *, run_id: UUID, **kwargs: Any) -> None:
        if run_id not in self.active_spans:
            return
        span = self.active_spans[run_id]
        setattr(span, 'outputs', safe_serialize(outputs))
        self._end_span(run_id)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            serialized = serialized or {}
            attributes = {
                "model": f"tool_{serialized.get('name', 'unknown_tool')}",
                "tool_name": serialized.get("name", "unknown_tool"),
                "tool_input": input_str,
            }
            self._start_span(run_id, parent_run_id, "tool", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_tool_start: {e}")

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self.active_spans:
            span = self.active_spans[run_id]
            span.completion = str(output)
            setattr(span, 'tool_output', str(output))
        self._end_span(run_id)

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            attributes = {
                "model": f"agent_action_{action.tool}",
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
            self._start_span(run_id, parent_run_id, "agent_action", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_agent_action: {e}")

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self.active_spans:
            span = self.active_spans[run_id]
            span.completion = safe_serialize(finish.return_values)
            setattr(span, 'return_values', finish.return_values)
            setattr(span, 'log', finish.log)
        self._end_span(run_id)

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self._end_span(run_id, success=False, error=error)
        finally:
            release_patching()

    def on_chain_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_span(run_id, success=False, error=error)

    def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_span(run_id, success=False, error=error)


class AsyncNeatlogsLangchainCallbackHandler(AsyncCallbackHandler):
    """
    Asynchronous Neatlogs callback handler for LangChain.

    Use this handler for *asynchronous* LangChain pipelines and agents (i.e., those
    that use `async def`/`await`, e.g. `await chain.arun()`, async agents/tools).

    This ensures Neatlogs telemetry is tracked correctly for concurrent, non-blocking
    workflows. Do not use this handler for normal blocking pipelines (prefer
    NeatlogsLangchainCallbackHandler there).

    Example:
        handler = AsyncNeatlogsLangchainCallbackHandler(...)
        result = await chain.arun(..., callbacks=[handler])
    """

    def __init__(self, api_key: Optional[str] = None, tags: Optional[List[str]] = None):
        from ....core import LLMTracker
        tracker = get_tracker()
        # If there's no global tracker, create a temporary one just for this callback handler
        # This allows the callback handler to work independently without triggering automatic patching
        if not tracker and api_key:
            # Create a temporary tracker that sends data to the server
            # Since there's no global tracker, there's no conflict
            tracker = LLMTracker(api_key=api_key, tags=tags,
                                 enable_server_sending=True)
            logging.info(
                "Neatlogs: Created temporary tracker for LangChain callback handler")
        elif not tracker:
            logging.warning(
                "Neatlogs Tracker not initialized. Please call neatlogs.init(api_key=...) to enable automatic patching, "
                "or ensure a tracker is already initialized.")

        self.tracker = tracker
        self._api_key = api_key
        self._tags = tags or []

        self.active_spans: Dict[UUID, LLMSpan] = {}

    def _start_span(self, run_id: UUID, parent_run_id: Optional[UUID], operation: str, attributes: Dict[str, Any]) -> None:
        if not self.tracker or run_id in self.active_spans:
            if run_id in self.active_spans:
                logging.debug(
                    f"Span for run_id {run_id} already exists. Ignoring duplicate start event.")
            return

        if not should_track_span(operation, attributes):
            logging.debug(
                f"Skipping span for operation {operation} with attributes {attributes}")
            return

        model = attributes.get("model", f"langchain_{operation}")
        provider = attributes.get("provider", "langchain")

        span = self.tracker.start_llm_span(
            model=model, provider=provider, framework="langchain")

        for key, value in attributes.items():
            setattr(span, key, value)

        if parent_run_id and parent_run_id in self.active_spans:
            parent_span = self.active_spans[parent_run_id]
            setattr(span, 'parent_span_id', parent_span.span_id)

        self.active_spans[run_id] = span

    def _end_span(self, run_id: UUID, success: bool = True, error: Optional[Exception] = None) -> None:
        if not self.tracker or run_id not in self.active_spans:
            return

        span = self.active_spans.pop(run_id)
        self.tracker.end_llm_span(span, success=success, error=error)

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        suppress_patching()
        try:
            serialized = serialized or {}
            model_info = get_model_info(serialized)
            model_name = model_info.get("model_name", "unknown")
            provider = model_info.get("provider", "langchain")

            attributes = {
                "model": model_name,
                "provider": provider,
                "prompts": prompts,
                "invocation_params": serialized.get('invocation_params', {})
            }
            self._start_span(run_id, parent_run_id, "llm", attributes)

            span = self.active_spans.get(run_id)
            if span:
                messages = []
                for prompt in prompts:
                    if isinstance(prompt, str):
                        if prompt.startswith("Human: "):
                            messages.append(
                                {"role": "user", "content": prompt[7:]})
                        elif prompt.startswith("System: "):
                            messages.append(
                                {"role": "system", "content": prompt[8:]})
                        else:
                            messages.append(
                                {"role": "user", "content": prompt})
                    else:
                        messages.append(
                            {"role": "user", "content": str(prompt)})
                span.messages = messages
        except Exception as e:
            logging.error(f"Error in Neatlogs on_llm_start: {e}")

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            if run_id not in self.active_spans:
                return
            span = self.active_spans[run_id]

            if response.generations:
                completions = [
                    gen.text for gen_list in response.generations for gen in gen_list]
                span.completion = "\n".join(completions)

            if response.llm_output and 'token_usage' in response.llm_output:
                usage = response.llm_output['token_usage']
                span.prompt_tokens = usage.get('prompt_tokens', 0)
                span.completion_tokens = usage.get('completion_tokens', 0)
                span.total_tokens = usage.get('total_tokens', 0)

            self._end_span(run_id)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_llm_end: {e}")
        finally:
            release_patching()

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        suppress_patching()
        try:
            serialized = serialized or {}
            model_info = get_model_info(serialized)
            model_name = model_info.get("model_name", "unknown")
            provider = model_info.get("provider", "langchain")

            attributes = {
                "model": model_name,
                "provider": provider,
                "invocation_params": serialized.get('invocation_params', {})
            }
            self._start_span(run_id, parent_run_id, "llm", attributes)

            span = self.active_spans.get(run_id)
            if span:
                # messages is a list of lists of BaseMessage. We handle the first list.
                if messages and messages[0]:
                    span.messages = [
                        {"role": msg.type, "content": msg.content}
                        for msg in messages[0]
                    ]
        except Exception as e:
            logging.error(f"Error in Neatlogs on_chat_model_start: {e}")

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            serialized = serialized or {}
            chain_name = serialized.get("name", "unknown_chain")
            attributes = {
                "model": f"chain_{chain_name}",
                "chain_name": chain_name,
                "inputs": safe_serialize(inputs),
            }
            if should_track_span("chain", attributes):
                self._start_span(run_id, parent_run_id, "chain", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_chain_start: {e}")

    async def on_chain_end(self, outputs: Union[Dict[str, Any], str], *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self.active_spans:
            span = self.active_spans[run_id]
            setattr(span, 'outputs', safe_serialize(outputs))
            self._end_span(run_id)

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            serialized = serialized or {}
            attributes = {
                "model": f"tool_{serialized.get('name', 'unknown_tool')}",
                "tool_name": serialized.get("name", "unknown_tool"),
                "tool_input": input_str,
            }
            self._start_span(run_id, parent_run_id, "tool", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_tool_start: {e}")

    async def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self.active_spans:
            span = self.active_spans[run_id]
            span.completion = str(output)
            setattr(span, 'tool_output', str(output))
        self._end_span(run_id)

    async def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            attributes = {
                "model": f"agent_action_{action.tool}",
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
            self._start_span(run_id, parent_run_id, "agent_action", attributes)
        except Exception as e:
            logging.error(f"Error in Neatlogs on_agent_action: {e}")

    async def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self.active_spans:
            span = self.active_spans[run_id]
            span.completion = safe_serialize(finish.return_values)
            setattr(span, 'return_values', finish.return_values)
            setattr(span, 'log', finish.log)
        self._end_span(run_id)

    async def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            self._end_span(run_id, success=False, error=error)
        finally:
            release_patching()

    async def on_chain_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_span(run_id, success=False, error=error)

    async def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_span(run_id, success=False, error=error)
