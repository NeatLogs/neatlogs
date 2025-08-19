"""NeatLogs Tracker stream wrapper implementation.

This module provides wrappers for streaming functionality,
It instruments streams to collect telemetry data for monitoring and analysis.
"""

import time
from typing import Any, AsyncIterator, Iterator
from .core import LLMSpan
from .semconv import SpanAttributes, MessageAttributes

class NeatLogsStreamWrapper:
    """Wrapper for streaming responses."""

    def __init__(self, stream: Any, span: LLMSpan, request_kwargs: dict):
        self._stream = stream
        self._span = span
        self._request_kwargs = request_kwargs
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
                                self._tool_calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
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
        full_content = "".join(self._content_chunks)
        self._span.completion = full_content
        if self._usage:
            self._span.prompt_tokens = self._usage.prompt_tokens
            self._span.completion_tokens = self._usage.completion_tokens
            self._span.total_tokens = self._usage.total_tokens
        self._span.end()
