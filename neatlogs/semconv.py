"""
Semantic Conventions for Neatlogs
=======================================

Defines standardized attribute names and values for LLM operations,
providing a consistent interface for span attributes and structured logs.
This module streamlines semantic data capture and aids future integration.
"""

from typing import Dict, Any
from openinference.semconv.trace import (
    SpanAttributes,
    OpenInferenceSpanKindValues,
    MessageAttributes as OIMessageAttributes,
    ToolCallAttributes as OIToolCallAttributes,
)

# LLM-specific semantic conventions


class LLMAttributes:
    """Semantic conventions for LLM operations.

    This class defines a set of constants representing standardized semantic keys
    for LLM tracking, spanning prompt, completion, error, tool usage, and more.
    """

    # Core LLM attributes
    LLM_SYSTEM = SpanAttributes.LLM_SYSTEM  # e.g., "openai", "anthropic", "google"
    # e.g., "gpt-4", "claude-3-sonnet"
    LLM_REQUEST_MODEL = SpanAttributes.LLM_MODEL_NAME
    # "chat", "completion", "embedding"
    LLM_REQUEST_TYPE = SpanAttributes.OPENINFERENCE_SPAN_KIND

    # Request Parameters
    # We will store params as JSON here
    LLM_REQUEST_TEMPERATURE = SpanAttributes.LLM_INVOCATION_PARAMETERS
    # But for individual OTel GenAI attributes we keep these keys available for mapping
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

    # Token usage
    LLM_USAGE_PROMPT_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
    LLM_USAGE_COMPLETION_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    LLM_USAGE_TOTAL_TOKENS = SpanAttributes.LLM_TOKEN_COUNT_TOTAL

    # OTel GenAI Token Usage
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Response attributes
    # In response we confirm the model
    LLM_RESPONSE_MODEL = SpanAttributes.LLM_MODEL_NAME
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Cost (OpenInference specific)
    LLM_COST_TOTAL = "llm.cost.total"
    LLM_COST_PROMPT = "llm.cost.prompt"
    LLM_COST_COMPLETION = "llm.cost.completion"

    # Agent-specific attributes
    AGENT_ID = "agent.id"
    AGENT_SESSION_ID = "agent.session.id"
    AGENT_THREAD_ID = "agent.thread.id"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK_TRACE = "error.stack_trace"


class MessageAttributes:
    """Structured message attributes for detailed tracking.

    Uses OpenInference MessageAttributes where possible.
    """
    # We use the OpenInference convention: llm.input_messages.{i}.message.role
    # These are helper constants for the structure
    MESSAGE_ROLE = OIMessageAttributes.MESSAGE_ROLE
    MESSAGE_CONTENT = OIMessageAttributes.MESSAGE_CONTENT
    MESSAGE_TOOL_CALLS = OIMessageAttributes.MESSAGE_TOOL_CALLS
    MESSAGE_FUNCTION_NAME = OIMessageAttributes.MESSAGE_FUNCTION_CALL_NAME
    MESSAGE_FUNCTION_ARGUMENTS = OIMessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON

    TOOL_CALL_ID = OIToolCallAttributes.TOOL_CALL_ID
    TOOL_CALL_FUNCTION_NAME = OIToolCallAttributes.TOOL_CALL_FUNCTION_NAME
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON = OIToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON


class LLMRequestTypeValues:
    """Standard values for LLM request types.

    Contains enumerated constants for categorizing different types of LLM requests.
    """
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    TOOL = "tool"
    CHAIN = "chain"
    AGENT = "agent"


class LLMEvents:
    """Standard event names for LLM operations.

    Supplies event name constants (start, end, error, streaming, etc.) for compliant tracking.
    """

    # Core events
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    LLM_CALL_ERROR = "llm.call.error"
    LLM_STREAM_START = "llm.stream.start"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_STREAM_END = "llm.stream.end"

    # Agent events
    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    AGENT_ERROR = "agent.error"

    # Session events
    SESSION_START = "session.start"
    SESSION_END = "session.end"

    # Tool events
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_CALL_ERROR = "tool.call.error"


def get_provider_system_name(provider: str) -> str:
    """Map provider names to standardized system names"""
    provider_mapping = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "gemini": "gemini",
        "azure": "azure_openai",
        "azure_openai": "azure_openai",
        "litellm": "litellm",
        "mistral": "mistral",
        "cohere": "cohere",
    }
    return provider_mapping.get(provider.lower(), provider.lower())


def format_messages_for_attribute(messages: list) -> str:
    """Format messages for OpenTelemetry attribute storage"""
    import json
    try:
        # Remove any non-serializable data and limit size
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                clean_msg = {
                    "role": msg.get("role", "unknown"),
                    # Limit content length
                    "content": str(msg.get("content", ""))[:2000]
                }
                # Include tool calls if present
                if "tool_calls" in msg:
                    clean_msg["tool_calls"] = msg["tool_calls"]
                clean_messages.append(clean_msg)
        return json.dumps(clean_messages)
    except Exception:
        return json.dumps([{"role": "unknown", "content": "serialization_error"}])


def format_tools_for_attribute(tools: list) -> str:
    """Format tool definitions for OpenTelemetry attribute storage"""
    import json
    try:
        clean_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                clean_tool = {
                    "name": tool.get("name", "unknown"),
                    "type": tool.get("type", "function"),
                    "description": str(tool.get("description", ""))[:500]
                }
                clean_tools.append(clean_tool)
        return json.dumps(clean_tools)
    except Exception:
        return json.dumps([{"name": "unknown", "type": "function", "description": "serialization_error"}])


def extract_tool_calls_data(content_blocks) -> list:
    """Extract tool calls from content blocks (Anthropic format)"""
    tool_calls = []
    try:
        for block in content_blocks:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_call = {
                    "id": getattr(block, 'id', 'unknown'),
                    "name": getattr(block, 'name', 'unknown'),
                    "type": "function",
                    "arguments": getattr(block, 'input', {})
                }
                tool_calls.append(tool_call)
            elif isinstance(block, dict) and block.get('type') == 'tool_use':
                tool_call = {
                    "id": block.get('id', 'unknown'),
                    "name": block.get('name', 'unknown'),
                    "type": "function",
                    "arguments": block.get('input', {})
                }
                tool_calls.append(tool_call)
    except Exception:
        pass
    return tool_calls


def get_common_span_attributes(session_id: str, agent_id: str, thread_id: str,
                               model: str, provider: str) -> Dict[str, Any]:
    """Get common attributes that should be set on all LLM spans"""
    return {
        LLMAttributes.AGENT_SESSION_ID: session_id,
        LLMAttributes.AGENT_ID: agent_id,
        LLMAttributes.AGENT_THREAD_ID: thread_id,
        LLMAttributes.LLM_REQUEST_MODEL: model or "unknown",
        LLMAttributes.LLM_SYSTEM: get_provider_system_name(provider or "unknown"),
    }
