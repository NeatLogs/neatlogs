""" 
Langgraph event handler for Neatlogs.
"""

import logging
import json
import inspect
import contextvars
from functools import wraps
from typing import Any, Callable, Dict, List

from ..core import LLMSpan, set_current_framework, clear_current_framework
from .base import BaseEventHandler

# Helper functions to extract data


def _extract_messages_from_io(data: Any) -> list:
    if isinstance(data, dict) and "messages" in data:
        return data["messages"]
    return []


def _get_message_content(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content)
    return ""


class LangGraphHandler(BaseEventHandler):
    """
      Event handler for tracking LangGraph workflows in Neatlogs.
      This class is responsible for wrapping core LangGraph methods 
      (e.g., add_node, compile, invoke) to insert tracking spans.
    """

    def __init__(self, tracker):
        """ Initialize the handler with a Neatlogs LLMTracker. """
        super().__init__(tracker)
        self._graph_execution_context = contextvars.ContextVar(
            'neatlogs_langgraph_execution', default=None)
        # Smart filtering to reduce noise - but keep workflow tracking for message capture
        self._enable_smart_filtering = True

    def configure_smart_filtering(self, enabled: bool = True):
        """Configure smart filtering to show/hide framework spans.

        Args:
            enabled (bool): If True, hides empty framework spans and non-LLM nodes.
                           If False, shows all spans for complete debugging visibility.
        """
        self._enable_smart_filtering = enabled
        logging.info(
            f"Neatlogs: LangGraph smart filtering {'enabled' if enabled else 'disabled'}")

    def _safe_serialize(self, obj: Any) -> str:
        """Safely serialize objects that might not be JSON serializable."""
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            # Handle non-serializable objects like AIMessage
            return str(obj)

    def _format_message(self, msg: Any) -> Dict[str, Any]:
        """Format a single message properly based on its type."""
        if hasattr(msg, 'role') or hasattr(msg, 'type'):
            role = getattr(msg, 'role', getattr(msg, 'type', 'unknown'))
            content = _get_message_content(msg)

            formatted_msg = {
                'role': role,
                'content': content
            }

            # Add tool call information for AI messages with tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                formatted_msg['tool_calls'] = []
                for tool_call in msg.tool_calls:
                    tool_call_info = {
                        'id': getattr(tool_call, 'id', ''),
                        'name': getattr(tool_call, 'name', ''),
                        'args': getattr(tool_call, 'args', {})
                    }
                    # Handle dict-style tool calls
                    if isinstance(tool_call, dict):
                        tool_call_info = {
                            'id': tool_call.get('id', ''),
                            'name': tool_call.get('name', ''),
                            'args': tool_call.get('args', {})
                        }
                    formatted_msg['tool_calls'].append(tool_call_info)

            # Add tool call ID for tool messages
            if hasattr(msg, 'tool_call_id'):
                formatted_msg['tool_call_id'] = msg.tool_call_id

            return formatted_msg
        return None

    def extract_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        """Extracts messages from LangGraph input arguments.

        Supports multiple LangGraph state patterns:
        - TypedDict with messages field
        - Pydantic BaseModel with messages field
        - Dataclass with messages field
        - Custom state fields (query, input, prompt, etc.)
        - Private state patterns
        """
        input_data = args[0] if args else kwargs.get("input", {})
        messages = _extract_messages_from_io(input_data)
        formatted_messages = []

        # Handle standard LangChain messages
        if messages:
            for msg in messages:
                formatted_msg = self._format_message(msg)
                if formatted_msg:
                    formatted_messages.append(formatted_msg)

        # Handle different state patterns when no messages field found
        elif isinstance(input_data, dict):
            # Pattern 1: Customer support style (query field)
            if 'query' in input_data:
                formatted_messages.append({
                    'role': 'user',
                    'content': input_data['query']
                })
            # Pattern 2: Generic input field
            elif 'input' in input_data:
                formatted_messages.append({
                    'role': 'user',
                    'content': str(input_data['input'])
                })
            # Pattern 3: Prompt field
            elif 'prompt' in input_data:
                formatted_messages.append({
                    'role': 'user',
                    'content': input_data['prompt']
                })
            # Pattern 4: Question field (Q&A patterns)
            elif 'question' in input_data:
                formatted_messages.append({
                    'role': 'user',
                    'content': input_data['question']
                })
            # Pattern 5: Text field (generic text processing)
            elif 'text' in input_data:
                formatted_messages.append({
                    'role': 'user',
                    'content': input_data['text']
                })
            # Pattern 6: Private data patterns
            elif 'private_data' in input_data:
                formatted_messages.append({
                    'role': 'system',
                    'content': str(input_data['private_data'])
                })

        # Handle Pydantic BaseModel instances
        elif hasattr(input_data, '__dict__'):
            # Extract from Pydantic model or dataclass
            data_dict = input_data.__dict__ if hasattr(
                input_data, '__dict__') else {}

            # Check for messages first
            if 'messages' in data_dict:
                messages = data_dict['messages']
                for msg in messages:
                    formatted_msg = self._format_message(msg)
                    if formatted_msg:
                        formatted_messages.append(formatted_msg)
            # Check for other common fields
            elif 'query' in data_dict:
                formatted_messages.append({
                    'role': 'user',
                    'content': data_dict['query']
                })
            elif 'input' in data_dict:
                formatted_messages.append({
                    'role': 'user',
                    'content': str(data_dict['input'])
                })
        
        return formatted_messages



    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extracts completion data from a LangGraph result with proper message formatting.

        Supports multiple LangGraph output patterns:
        - Standard message-based responses
        - Tool call responses  
        - State field responses (category, sentiment, response, etc.)
        - Pydantic model responses
        - Dataclass responses
        - Private state responses
        """
        if not isinstance(response, dict) and not hasattr(response, '__dict__'):
            return {'completion': str(response), 'messages': []}

        # Convert Pydantic/dataclass to dict if needed
        if hasattr(response, '__dict__') and not isinstance(response, dict):
            response_dict = response.__dict__
        else:
            response_dict = response

        # Extract messages for proper formatting
        messages = _extract_messages_from_io(response_dict)
        formatted_messages = []
        final_response_content = ""

        if messages:
            for msg in messages:
                # Format each message properly using the new method
                formatted_msg = self._format_message(msg)
                if formatted_msg:
                    formatted_messages.append(formatted_msg)

                    # Get final response from last non-tool message with actual content
                    if formatted_msg.get('role') not in ['tool'] and formatted_msg.get('content'):
                        final_response_content = formatted_msg['content']

        # Handle tool responses in dict format (like from tools node)
        if isinstance(response_dict, dict) and 'messages' in response_dict and isinstance(response_dict['messages'], list):
            for msg_dict in response_dict['messages']:
                if isinstance(msg_dict, dict):
                    if msg_dict.get('role') == 'tool':
                        tool_msg = {
                            'role': 'tool',
                            'content': msg_dict.get('content', '')
                        }
                        if 'tool_call_id' in msg_dict:
                            tool_msg['tool_call_id'] = msg_dict['tool_call_id']
                        formatted_messages.append(tool_msg)
                    # Handle any other message types in the response
                    else:
                        formatted_msg = self._format_message(msg_dict)
                        if formatted_msg:
                            formatted_messages.append(formatted_msg)
                            if formatted_msg.get('role') not in ['tool'] and formatted_msg.get('content'):
                                final_response_content = formatted_msg['content']

        # Handle structured state responses (various patterns)
        elif isinstance(response_dict, dict):
            # Check for various response patterns
            if 'category' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': response_dict['category']
                })
            elif 'sentiment' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': response_dict['sentiment']
                })
            elif 'response' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': response_dict['response']
                })
                final_response_content = response_dict['response']
            # Pattern: Answer field (Q&A systems)
            elif 'answer' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': response_dict['answer']
                })
                final_response_content = response_dict['answer']
            # Pattern: Output field (generic processing)
            elif 'output' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': str(response_dict['output'])
                })
                final_response_content = str(response_dict['output'])
            # Pattern: Result field
            elif 'result' in response_dict:
                formatted_messages.append({
                    'role': 'assistant',
                    'content': str(response_dict['result'])
                })
                final_response_content = str(response_dict['result'])
            # Pattern: Private data responses
            elif 'private_data' in response_dict:
                formatted_messages.append({
                    'role': 'system',
                    'content': str(response_dict['private_data'])
                })
                final_response_content = str(response_dict['private_data'])

        return {
            'completion': final_response_content or self._safe_serialize(response_dict if isinstance(response_dict, dict) else response),
            'messages': formatted_messages
        }

    def _detect_llm_node(self, func: Callable, node_name: str) -> bool:
        """Detect if a node function contains LLM calls detection."""
        try:
            # Get the source code of the function
            source = inspect.getsource(func)

            # Check for common LLM patterns
            llm_patterns = [
                "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI",
                ".invoke(", ".ainvoke(", ".stream(", ".astream(",
                "llm.", "model.", "chat.", "openai.", "anthropic.",
                "bind_tools", "with_structured_output", "tool_calls"
            ]

            for pattern in llm_patterns:
                if pattern in source:
                    logging.debug(
                        f"Neatlogs: LLM pattern '{pattern}' detected in node '{node_name}'")
                    return True

            # Check function parameter names and local variables
            if hasattr(func, "__code__"):
                local_vars = func.__code__.co_varnames
                llm_var_patterns = ["llm", "model", "chat",
                                    "client", "openai", "anthropic"]
                if any(var in llm_var_patterns for var in local_vars):
                    logging.debug(
                        f"Neatlogs: LLM variable detected in node '{node_name}'")
                    return True

            # Check if node name suggests LLM usage
            llm_node_names = ["agent", "chat", "generate",
                              "llm", "model", "query", "ask"]
            if any(name in node_name.lower() for name in llm_node_names):
                logging.debug(
                    f"Neatlogs: LLM node name pattern detected: '{node_name}'")
                return True

        except Exception as e:
            logging.debug(
                f"Neatlogs: Could not inspect node '{node_name}': {e}")
            # If we can't inspect, assume it might be an LLM node for safety
            return True

        return False

    def _should_create_node_span(self, node_name: str, func: Callable) -> bool:
        """Determine if we should create a span for this node based on smart filtering."""
        if not self._enable_smart_filtering:
            return True

        # Always skip these framework-only nodes
        framework_only_nodes = ["__start__", "__end__"]
        if node_name in framework_only_nodes:
            return False

        # Tool message nodes should not be treated as LLM calls - they're just state updates
        if "tool_message" in node_name.lower() or "add_tool" in node_name.lower():
            return False

        # Always create spans for tool nodes and likely LLM nodes
        tool_or_llm_nodes = ["tools", "call_tool"]
        if node_name in tool_or_llm_nodes:
            return True

        # Check if it's an LLM node
        return self._detect_llm_node(func, node_name)

    def _is_actual_llm_operation(self, node_name: str, func: Callable) -> bool:
        """Determine if this is an actual LLM operation that deserves tracking."""

        # Definitely not LLM operations - these create the "empty logs"
        non_llm_operations = [
            "add_tool_message", "tool_message", "compile", "stream",
            "__start__", "__end__", "routing", "conditional"
        ]

        if any(op in node_name.lower() for op in non_llm_operations):
            return False

        # Check if it's a simple state update function
        if hasattr(func, '__name__') and func.__name__ in ['add_tool_message']:
            return False

        # Tool nodes are important operations but should be properly categorized
        if hasattr(func, 'tools_by_name') or str(type(func).__name__) == 'ToolNode':
            return True  # Tool execution is worth tracking

        # Use our existing LLM detection logic for actual model calls
        return self._detect_llm_node(func, node_name)

    def _track_node_execution(self, node_name: str):
        """Adds a node to the list of executed nodes in the current context."""
        execution_state = self._graph_execution_context.get()
        if execution_state and node_name not in execution_state["executed_nodes"]:
            execution_state["executed_nodes"].append(node_name)

    def wrap_node_action(self, node_name: str, original_action: Callable) -> Callable:
        """Wraps an individual node's callable to trace its execution with smart filtering."""
        from ..core import (
            set_active_langgraph_node_span,
            clear_active_langgraph_node_span,
            suppress_patching,
            release_patching,
            current_span_id_context
        )

        # Check if we should create a span for this node
        should_create_span = self._should_create_node_span(
            node_name, original_action)

        # Additionally check if this is an actual LLM operation
        is_actual_llm = self._is_actual_llm_operation(
            node_name, original_action)

        # Only create spans for actual LLM operations to avoid empty logs
        should_create_span = should_create_span and is_actual_llm

        # Determine if this is a tool operation vs LLM operation
        is_tool_operation = (
            "tool" in node_name.lower() or
            hasattr(original_action, 'tools_by_name') or
            str(type(original_action).__name__) == 'ToolNode'
        )
        node_type = "tool_call" if is_tool_operation else "framework_node"

        # Handle callable objects (like ToolNode) differently from functions
        # ToolNode extends RunnableCallable but doesn't expose __call__ directly
        # Check for both callable and specific class types
        is_callable_object = (
            (hasattr(original_action, '__call__') and not inspect.isfunction(
                original_action) and not inspect.ismethod(original_action))
            # RunnableCallable has _func attribute
            or hasattr(original_action, '_func')
            # ToolNode specific check
            or hasattr(original_action, 'tools_by_name')
            or str(type(original_action).__name__) in ['ToolNode', 'RunnableCallable']
        )

        if inspect.iscoroutinefunction(original_action):
                @wraps(original_action)
                async def async_tracked_action(*args, **kwargs):
                    self._track_node_execution(node_name)
                    span = None
                    token = None
                    try:
                        if should_create_span:
                            logging.debug(
                                f"Neatlogs: Creating span for LLM node: {node_name}")
                            provider_name = f"langgraph.node.{node_name}"
                            span = self.tracker.start_llm_span(
                                model=f"node/{node_name}", provider=provider_name, framework="langgraph", node_type=node_type, node_name=node_name)
                            setattr(span, 'node_name', node_name)
                            set_active_langgraph_node_span(span)
                            token = current_span_id_context.set(span.span_id)
                            release_patching()  # Allow provider patchers to run
                        else:
                            logging.debug(
                                f"Neatlogs: Skipping span for non-LLM node: {node_name}")

                        input_state_messages = self.extract_messages(*args, **kwargs)
                        result = await original_action(*args, **kwargs)

                        if span:
                            response_data = self.extract_response_data(result)
                            output_state_messages = response_data.get('messages', [])
                            num_input_messages = len(input_state_messages)
                            new_messages = output_state_messages[num_input_messages:]
                            new_completion = ""
                            if new_messages:
                                for msg in reversed(new_messages):
                                    if msg.get('role') not in ['tool'] and msg.get('content'):
                                        new_completion = msg.get('content', '')
                                        break
                            if not new_completion:
                                new_completion = response_data.get('completion', str(result))
                            span.messages = input_state_messages
                            span.completion = new_completion
                            self.tracker.end_llm_span(span, success=True)
                        return result
                    except Exception as e:
                        if span:
                            self.tracker.end_llm_span(
                                span, success=False, error=e)
                        raise
                    finally:
                        if token:
                            current_span_id_context.reset(token)
                        if span:
                            suppress_patching()  # Restore suppression
                            clear_active_langgraph_node_span()
                return async_tracked_action
        else:
                @wraps(original_action)
                def sync_tracked_action(*args, **kwargs):
                    self._track_node_execution(node_name)
                    span = None
                    token = None
                    try:
                        if should_create_span:
                            logging.debug(
                                f"Neatlogs: Creating span for LLM node: {node_name}")
                            provider_name = f"langgraph.node.{node_name}"
                            span = self.tracker.start_llm_span(
                                model=f"node/{node_name}", provider=provider_name, framework="langgraph", node_type=node_type, node_name=node_name)
                            setattr(span, 'node_name', node_name)
                            set_active_langgraph_node_span(span)
                            token = current_span_id_context.set(span.span_id)
                            release_patching()  # Allow provider patchers to run
                        else:
                            logging.debug(
                                f"Neatlogs: Skipping span for non-LLM node: {node_name}")

                        input_state_messages = self.extract_messages(*args, **kwargs)
                        result = original_action(*args, **kwargs)

                        if span:
                            response_data = self.extract_response_data(result)
                            output_state_messages = response_data.get('messages', [])
                            num_input_messages = len(input_state_messages)
                            new_messages = output_state_messages[num_input_messages:]
                            new_completion = ""
                            if new_messages:
                                for msg in reversed(new_messages):
                                    if msg.get('role') not in ['tool'] and msg.get('content'):
                                        new_completion = msg.get('content', '')
                                        break
                            if not new_completion:
                                new_completion = response_data.get('completion', str(result))
                            span.messages = input_state_messages
                            span.completion = new_completion
                            self.tracker.end_llm_span(span, success=True)
                        return result
                    except Exception as e:
                        if span:
                            self.tracker.end_llm_span(
                                span, success=False, error=e)
                        raise
                    finally:
                        if token:
                            current_span_id_context.reset(token)
                        if span:
                            suppress_patching()  # Restore suppression
                            clear_active_langgraph_node_span()
                return sync_tracked_action

    def wrap_compile(self, original_compile: Callable) -> Callable:
        """Wraps the `compile` method to capture graph structure."""
        @wraps(original_compile)
        def wrapped_compile(graph_instance, *args, **kwargs):
            # Skip graph compilation tracking - it's not an LLM operation
            # Graph compilation is just framework setup, not meaningful for LLM tracking
            return original_compile(graph_instance, *args, **kwargs)

        return wrapped_compile

    def _create_workflow_wrapper(self, original_method: Callable, is_async: bool, is_stream: bool):
        """Factory for creating invoke/stream wrappers."""
        from ..core import current_span_id_context

        async def astream_wrapper_gen(stream_gen, span, token):
            try:
                async for chunk in stream_gen:
                    self._process_chunk(chunk, span)
                    yield chunk
            finally:
                self._finalize_workflow_span(span)
                self._graph_execution_context.reset(token)

        def stream_wrapper_gen(stream_gen, span, token):
            try:
                for chunk in stream_gen:
                    self._process_chunk(chunk, span)
                    yield chunk
            finally:
                self._finalize_workflow_span(span)
                self._graph_execution_context.reset(token)

        if is_async:
            if is_stream:
                @wraps(original_method)
                async def async_stream_wrapper(*args, **kwargs):
                    if self._graph_execution_context.get() is not None:
                        async for chunk in original_method(*args, **kwargs):
                            yield chunk
                        return
                    span, graph_token = self._start_workflow_span(
                        is_stream, *args, **kwargs)
                    span_token = current_span_id_context.set(span.span_id)
                    try:
                        stream_gen = original_method(*args, **kwargs)
                        async for chunk in astream_wrapper_gen(stream_gen, span, graph_token):
                            yield chunk
                    except Exception as e:
                        self.tracker.end_llm_span(
                            span, success=False, error=e)
                        self._graph_execution_context.reset(graph_token)
                        raise
                    finally:
                        current_span_id_context.reset(span_token)
                return async_stream_wrapper
            else:
                @wraps(original_method)
                async def async_wrapper(*args, **kwargs):
                    if self._graph_execution_context.get() is not None:
                        return await original_method(*args, **kwargs)
                    span, graph_token = self._start_workflow_span(
                        is_stream, *args, **kwargs)
                    span_token = current_span_id_context.set(span.span_id)
                    try:
                        result = await original_method(*args, **kwargs)
                        self._finalize_workflow_span(span, result)
                        self._graph_execution_context.reset(graph_token)
                        return result
                    except Exception as e:
                        self.tracker.end_llm_span(
                            span, success=False, error=e)
                        self._graph_execution_context.reset(graph_token)
                        raise
                    finally:
                        current_span_id_context.reset(span_token)
                return async_wrapper
        else:
            if is_stream:
                @wraps(original_method)
                def sync_stream_wrapper(*args, **kwargs):
                    if self._graph_execution_context.get() is not None:
                        for chunk in original_method(*args, **kwargs):
                            yield chunk
                        return

                    span, graph_token = self._start_workflow_span(
                        is_stream, *args, **kwargs)
                    span_token = current_span_id_context.set(span.span_id)
                    try:
                        stream_gen = original_method(*args, **kwargs)
                        for chunk in stream_wrapper_gen(stream_gen, span, graph_token):
                            yield chunk
                    except Exception as e:
                        if hasattr(span, 'end'):
                            self.tracker.end_llm_span(
                                span, success=False, error=e)
                        self._graph_execution_context.reset(graph_token)
                        raise
                    finally:
                        current_span_id_context.reset(span_token)
                return sync_stream_wrapper
            else:
                @wraps(original_method)
                def sync_invoke_wrapper(*args, **kwargs):
                    if self._graph_execution_context.get() is not None:
                        return original_method(*args, **kwargs)

                    span, graph_token = self._start_workflow_span(
                        is_stream, *args, **kwargs)
                    span_token = current_span_id_context.set(span.span_id)
                    try:
                        result = original_method(*args, **kwargs)
                        self._finalize_workflow_span(span, result)
                        self._graph_execution_context.reset(graph_token)
                        return result
                    except Exception as e:
                        if hasattr(span, 'end'):
                            self.tracker.end_llm_span(
                                span, success=False, error=e)
                        self._graph_execution_context.reset(graph_token)
                        raise
                    finally:
                        current_span_id_context.reset(span_token)
                return sync_invoke_wrapper

    def _start_workflow_span(self, is_stream: bool, *args, **kwargs) -> tuple[LLMSpan, Any]:
        """Starts a span for a workflow execution."""
        operation = "stream" if is_stream else "invoke"

        # Create a real span for the workflow to act as the root of the graph.
        span = self.tracker.start_llm_span(
            model=f"langgraph_workflow:{operation}",
            provider="langgraph",
            framework="langgraph",
            node_type="framework_node",
            node_name=f"LangGraph Workflow ({operation})"
        )

        execution_state = {
            "executed_nodes": [],
            "final_response": "",
            "accumulated_messages": []  # Track complete conversation
        }
        token = self._graph_execution_context.set(execution_state)

        # Pregel instance is the first arg, input data is usually second
        input_data = args[1] if len(args) > 1 else kwargs.get("input", {})

        # Extract input messages more comprehensively
        input_messages = []

        # Handle dict input: check for messages, or common keys like 'input'/'query'
        if isinstance(input_data, dict):
            if 'messages' in input_data and isinstance(input_data['messages'], list):
                for msg in input_data['messages']:
                    formatted_msg = self._format_message(msg)
                    if formatted_msg:
                        input_messages.append(formatted_msg)
            else:
                # Fallback to check for other common initial input keys
                for key in ['input', 'query', 'prompt', 'question', 'text']:
                    if key in input_data and input_data[key]:
                        input_messages.append({
                            'role': 'user',
                            'content': str(input_data[key])
                        })
                        break  # Stop after finding the first one

        # Handle raw string input
        elif isinstance(input_data, str):
            input_messages.append({
                'role': 'user',
                'content': input_data
            })

        # Initialize conversation state with input messages
        if input_messages:
            execution_state['accumulated_messages'] = input_messages.copy()
            span.messages = input_messages
        return span, token

    def _process_chunk(self, chunk: dict, span: LLMSpan):
        """Processes a stream chunk to update execution state."""
        execution_state = self._graph_execution_context.get()
        if not execution_state:
            return

        if isinstance(chunk, dict):
            for key, value in chunk.items():
                if key not in ['__start__', '__end__'] and key not in execution_state["executed_nodes"]:
                    self._track_node_execution(key)

                response_data = self.extract_response_data(value)
                if response_data.get('completion'):
                    execution_state['final_response'] = response_data['completion']

    def _finalize_workflow_span(self, span: LLMSpan, result: Any = None):
        """Finalizes the workflow span with collected data."""
        execution_state = self._graph_execution_context.get()
        if not execution_state:
            self.tracker.end_llm_span(
                span, success=False, error=Exception("Execution context lost"))
            return

        if result is not None:
            response_data = self.extract_response_data(result)
            span.completion = response_data.get('completion', '')
            # Update messages with final result
            if response_data.get('messages'):
                span.messages.extend(response_data['messages'])
        else:  # Streaming case
            span.completion = execution_state.get('final_response', '')

        setattr(span, 'executed_nodes', json.dumps(
            execution_state['executed_nodes']))
        self.tracker.end_llm_span(span, success=True)

    def wrap_method(self, original_method, provider_name="langgraph", method_type=None):
        """
        Returns a wrapper for the given LangGraph method.
        """
        is_async = inspect.iscoroutinefunction(
            original_method) or inspect.isasyncgenfunction(original_method)

        if method_type in ["invoke", "ainvoke"]:
            return self._create_workflow_wrapper(original_method, is_async=is_async, is_stream=False)
        if method_type in ["stream", "astream"]:
            return self._create_workflow_wrapper(original_method, is_async=is_async, is_stream=True)
        if method_type == "compile":
            return self.wrap_compile(original_method)

        # Default case, though should not be hit with the current patcher design
        return super().wrap_method(original_method, provider_name)
