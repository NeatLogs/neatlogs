"""
Neatlogs stream wrapper implementation.
=======================================

This module provides wrappers for LLM/AI streaming responses.
Streams are instrumented to collect telemetry and chunked content for monitoring,
enabling accurate and detailed logging of streaming LLM API usage.
"""

import time
from typing import Any,  Iterator
from .core import LLMSpan
from .semconv import MessageAttributes


class NeatlogsStreamWrapper:
    """
    Wrapper for streaming responses.

    This class provides an iterator interface and batch/finish management for streamed
    LLM API responses. It captures chunk content, tracks metadata, and signals span
    completion after the stream ends for comprehensive telemetry.
    """

    def __init__(self, stream: Any, span: LLMSpan, request_kwargs: dict, context_token: Any = None):
        self._stream = stream
        self._span = span
        self._request_kwargs = request_kwargs
        self._context_token = context_token
        self._start_time = time.time()
        self._first_token_time = None
        self._chunk_count = 0
        self._content_chunks = []
        self._finish_reason = None
        self._model = None
        self._response_id = None
        self._usage = None
        self._tool_calls = {}

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize_stream()
            raise

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self._finalize_stream()
        # Re-raise any exception that occurred within the stream.
        return False

    def _process_chunk(self, chunk: Any) -> None:
        self._chunk_count += 1
        if self._first_token_time is None and hasattr(chunk, 'choices') and chunk.choices:
            if any(choice.delta.content for choice in chunk.choices if hasattr(choice.delta, 'content')):
                self._first_token_time = time.time()

        if hasattr(chunk, 'id') and chunk.id and not self._response_id:
            self._response_id = chunk.id

        if hasattr(chunk, 'model') and chunk.model and not self._model:
            self._model = chunk.model

        if hasattr(chunk, 'choices') and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        self._content_chunks.append(choice.delta.content)
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            idx = tool_call.index
                            if idx not in self._tool_calls:
                                self._tool_calls[idx] = {
                                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            if tool_call.id:
                                self._tool_calls[idx]['id'] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    self._tool_calls[idx]['function']['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    self._tool_calls[idx]['function']['arguments'] += tool_call.function.arguments
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    self._finish_reason = choice.finish_reason

    def _finalize_stream(self) -> None:
        from .core import current_span_id_context
        full_content = "".join(self._content_chunks)
        self._span.completion = full_content
        if self._usage:
            self._span.prompt_tokens = self._usage.prompt_tokens
            self._span.completion_tokens = self._usage.completion_tokens
            self._span.total_tokens = self._usage.total_tokens
        self._span.end()
        if self._context_token:
            current_span_id_context.reset(self._context_token)
