"""
Semantic Conventions for Neatlogs
=======================================

Defines standardized attribute names and values for LLM operations,
providing a consistent interface for span attributes and structured logs.
This module streamlines semantic data capture and aids future integration.
"""

from typing import Dict, Any

# LLM-specific semantic conventions


class LLMAttributes:
    """Semantic conventions for LLM operations.

    This class defines a set of constants representing standardized semantic keys
    for LLM tracking, spanning prompt, completion, error, tool usage, and more.
    """

    # Core LLM attributes
    LLM_SYSTEM = "llm.system"  # e.g., "openai", "anthropic", "google"
    LLM_REQUEST_MODEL = "llm.request.model"  # e.g., "gpt-4", "claude-3-sonnet"
    LLM_REQUEST_TYPE = "llm.request.type"  # "chat", "completion", "embedding"
    LLM_REQUEST_TEMPERATURE = "llm.request.temperature"
    LLM_REQUEST_MAX_TOKENS = "llm.request.max_tokens"
    LLM_REQUEST_TOP_P = "llm.request.top_p"
    LLM_REQUEST_TOP_K = "llm.request.top_k"
    LLM_REQUEST_FREQUENCY_PENALTY = "llm.request.frequency_penalty"
    LLM_REQUEST_PRESENCE_PENALTY = "llm.request.presence_penalty"
    LLM_REQUEST_STREAMING = "llm.request.streaming"
    LLM_REQUEST_SEED = "llm.request.seed"

    # Token usage
    LLM_USAGE_PROMPT_TOKENS = "llm.usage.prompt_tokens"
    LLM_USAGE_COMPLETION_TOKENS = "llm.usage.completion_tokens"
    LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
    LLM_USAGE_STREAMING_TOKENS = "llm.usage.streaming_tokens"

    # Response attributes
    LLM_RESPONSE_MODEL = "llm.response.model"
    LLM_RESPONSE_ID = "llm.response.id"
    LLM_RESPONSE_FINISH_REASON = "llm.response.finish_reason"
    LLM_RESPONSE_STOP_REASON = "llm.response.stop_reason"
    LLM_RESPONSE_CHOICES = "llm.response.choices"

    # Cost and performance
    LLM_COST_TOTAL = "llm.cost.total"
    LLM_COST_PROMPT = "llm.cost.prompt"
    LLM_COST_COMPLETION = "llm.cost.completion"

    # Content attributes
    LLM_PROMPTS = "llm.prompts"  # JSON array of prompt messages
    LLM_COMPLETIONS = "llm.completions"  # JSON array of completion messages
    LLM_CONTENT_COMPLETION_CHUNK = "llm.content.completion.chunk"

    # Agent-specific attributes
    AGENT_ID = "agent.id"
    AGENT_SESSION_ID = "agent.session.id"
    AGENT_THREAD_ID = "agent.thread.id"
    AGENT_OPERATION = "agent.operation"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK_TRACE = "error.stack_trace"

    # Tool attributes
    LLM_TOOLS = "llm.tools"  # JSON array of available tools
    LLM_TOOL_CALLS = "llm.tool_calls"  # JSON array of tool calls made

    # Safety attributes
    LLM_SAFETY_RATINGS = "llm.safety.ratings"
    LLM_SAFETY_BLOCKED = "llm.safety.blocked"


class MessageAttributes:
    """Structured message attributes for detailed tracking.

    Specifies schema for indexed prompt and completion messages, including tool calls,
    useful for transforming LLM interaction history into semantic logs/attributes.
    """

    # Prompt attributes (indexed)
    PROMPT_ROLE = "llm.prompt.{i}.role"
    PROMPT_CONTENT = "llm.prompt.{i}.content"
    PROMPT_TYPE = "llm.prompt.{i}.type"
    PROMPT_TOOL_CALLS = "llm.prompt.{i}.tool_calls"

    # Completion attributes (indexed)
    COMPLETION_ID = "llm.completion.{i}.id"
    COMPLETION_ROLE = "llm.completion.{i}.role"
    COMPLETION_CONTENT = "llm.completion.{i}.content"
    COMPLETION_TYPE = "llm.completion.{i}.type"
    COMPLETION_FINISH_REASON = "llm.completion.{i}.finish_reason"
    COMPLETION_TOOL_CALLS = "llm.completion.{i}.tool_calls"

    # Tool call attributes (doubly indexed)
    COMPLETION_TOOL_CALL_ID = "llm.completion.{i}.tool_call.{j}.id"
    COMPLETION_TOOL_CALL_NAME = "llm.completion.{i}.tool_call.{j}.name"
    COMPLETION_TOOL_CALL_TYPE = "llm.completion.{i}.tool_call.{j}.type"
    COMPLETION_TOOL_CALL_ARGUMENTS = "llm.completion.{i}.tool_call.{j}.arguments"


class LLMRequestTypeValues:
    """Standard values for LLM request types.

    Contains enumerated constants for categorizing different types of LLM requests.
    """
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_TRANSLATION = "audio_translation"


class CoreAttributes:
    """Core system attributes.

    Provides attribute names used for system- and error-level reporting.
    """
    ERROR_MESSAGE = "error.message"
    ERROR_TYPE = "error.type"
    ERROR_STACK_TRACE = "error.stack_trace"


class InstrumentationAttributes:
    """Instrumentation metadata attributes.

    Used for capturing metadata about the underlying instrumentation library/version.
    """
    LIBRARY_NAME = "instrumentation.library.name"
    LIBRARY_VERSION = "instrumentation.library.version"


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
        "google": "google_genai",
        "google_genai": "google_genai",
        "gemini": "google_genai",
        "azure": "azure_openai",
        "azure_openai": "azure_openai",
        "litellm": "litellm",
        # "cohere": "cohere",
        # "huggingface": "huggingface",
        # "ollama": "ollama",
        # "claude": "anthropic",
        # "gpt": "openai",
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
