"""
Neatlogs Span Processor
=======================

Implements an OpenTelemetry SpanProcessor that captures finished spans,
extracts relevant LLM data, and sends it to the Neatlogs backend.
"""

import logging
import threading
import traceback
from typing import Optional

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import Span

from ..core import LLMTracker, LLMCallData
from ..semconv import LLMAttributes, MessageAttributes, LLMEvents

logger = logging.getLogger(__name__)


class NeatlogsSpanProcessor(SpanProcessor):
    """
    Captures OpenTelemetry spans and reports them to the Neatlogs backend.

    This processor listens for finished spans, identifies those containing
    LLM-related attributes (based on OpenInference semantic conventions),
    converts them into the standardized `LLMCallData` format, and dispatches
    them via the `LLMTracker` for ingestion by the Neatlogs server.
    """

    def __init__(self, tracker: LLMTracker):
        self.tracker = tracker

    def on_start(self, span: Span, parent_context: Optional[object] = None) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended."""
        logger.error(
            f"DEBUG: on_end called for span: {span.name if span else 'None'}")
        if not span:
            return

        # We only care about LLM spans (or chains/agents if we decide to track those too)
        # For now, let's focus on spans that have LLM attributes
        attributes = span.attributes or {}

        # Check if it's an LLM span
        # OpenInference uses 'openinference.span.kind' = 'LLM'
        # OTel GenAI uses 'gen_ai.system' or similar
        is_llm = (
            attributes.get("openinference.span.kind") == "LLM" or
            "llm.model_name" in attributes or
            "gen_ai.system" in attributes
        )

        if not is_llm:
            return

        try:
            # print(f"DEBUG: NeatlogsSpanProcessor received LLM span: {span.name}")
            self._process_llm_span(span)
        except Exception as e:
            print(f"DEBUG: Error processing span: {e}")
            logger.error(f"Neatlogs: Failed to process span {span.name}: {e}")
            logger.debug(traceback.format_exc())

    def shutdown(self) -> None:
        """Called when the tracer provider is shut down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Called when the tracer provider is force flushed."""
        return True

    def _process_llm_span(self, span: ReadableSpan):
        """Extract data from span and send to Neatlogs."""
        attributes = span.attributes or {}

        # Extract core fields
        span_id = format(span.context.span_id, "016x")
        trace_id = format(span.context.trace_id, "032x")

        # print(f"DEBUG: Span Attributes Keys: {list(attributes.keys())}")
        print(f"DEBUG: Input Value: {attributes.get('input.value')}")
        parent_span_id = format(span.parent.span_id,
                                "016x") if span.parent else None

        # Extract LLM fields using semantic conventions
        model = (
            attributes.get(LLMAttributes.LLM_REQUEST_MODEL) or
            attributes.get(LLMAttributes.LLM_RESPONSE_MODEL) or
            attributes.get("gen_ai.request.model") or
            "unknown"
        )

        provider = (
            attributes.get(LLMAttributes.LLM_SYSTEM) or
            attributes.get("gen_ai.system") or
            "unknown"
        )

        # Token usage
        prompt_tokens = (
            attributes.get(LLMAttributes.LLM_USAGE_PROMPT_TOKENS) or
            attributes.get(LLMAttributes.GEN_AI_USAGE_INPUT_TOKENS) or
            0
        )
        completion_tokens = (
            attributes.get(LLMAttributes.LLM_USAGE_COMPLETION_TOKENS) or
            attributes.get(LLMAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) or
            0
        )
        total_tokens = (
            attributes.get(LLMAttributes.LLM_USAGE_TOTAL_TOKENS) or
            (prompt_tokens + completion_tokens)
        )

        # Cost
        cost = attributes.get(LLMAttributes.LLM_COST_TOTAL) or 0.0

        # Messages (Input)
        messages = []

        # 1. Try to get messages from standard OpenInference attributes (llm.input_messages.x.message.content)
        # This requires iterating through indexed attributes which is tricky with flat dict.
        # However, OpenInference usually populates `input.value` as a fallback or primary for some providers.

        # 2. Try to parse `input.value` if it's JSON
        input_value = attributes.get("input.value")
        input_mime_type = attributes.get("input.mime_type")

        if input_value:
            import json
            try:
                if input_mime_type == "application/json":
                    input_data = json.loads(input_value)
                    # Handle Google GenAI format: {"contents": [{"role": "user", "parts": [{"text": "..."}]}]}
                    if isinstance(input_data, dict):
                        if "contents" in input_data:
                            for content in input_data["contents"]:
                                role = content.get("role", "user")
                                parts = content.get("parts", [])
                                text = ""
                                for part in parts:
                                    if isinstance(part, dict) and "text" in part:
                                        text += part["text"]
                                messages.append(
                                    {"role": role, "content": text})
                        # Handle OpenAI format: {"messages": [{"role": "user", "content": "..."}]}
                        elif "messages" in input_data:
                            messages = input_data["messages"]
                        # Handle simple dict if it looks like a message
                        elif "role" in input_data and "content" in input_data:
                            messages.append(input_data)
                else:
                    # Treat as raw string input
                    messages.append(
                        {"role": "user", "content": str(input_value)})
            except Exception:
                # Fallback: just use raw value
                messages.append({"role": "user", "content": str(input_value)})

        completion = (
            attributes.get("llm.output_messages.0.message.content") or
            attributes.get("output.value") or  # Fallback to output.value
            attributes.get("gen_ai.output.messages") or
            ""
        )

        # If completion is still empty but we have output.value as JSON, try to parse it
        if not completion and attributes.get("output.value") and attributes.get("output.mime_type") == "application/json":
            import json
            try:
                output_data = json.loads(attributes.get("output.value"))
                # Google GenAI response format
                if isinstance(output_data, dict) and "candidates" in output_data:
                    candidates = output_data["candidates"]
                    if candidates and len(candidates) > 0:
                        parts = candidates[0].get(
                            "content", {}).get("parts", [])
                        text = ""
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                text += part["text"]
                        completion = text
            except Exception:
                pass

        # Timestamps
        start_time = span.start_time / 1e9  # nanoseconds to seconds
        end_time = span.end_time / 1e9
        duration = end_time - start_time

        # Status
        status = "SUCCESS"
        error_report = None
        if not span.status.is_ok and span.status.description:
            status = "FAILURE"
            error_report = {
                "message": span.status.description,
                "type": "SpanError"
            }

        # Create LLMCallData
        call_data = LLMCallData(
            session_id=self.tracker.session_id,
            agent_id=self.tracker.agent_id,
            thread_id=self.tracker.thread_id,  # Or use trace_id if we want to align
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            node_type="llm_call",
            node_name=model,
            model=model,
            provider=provider,
            framework=None,  # Hard to get from span unless added as attribute
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            messages=messages,  # We need a robust way to get this
            completion=str(completion),
            timestamp=None,  # Will be set in LLMCallData
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            tags=self.tracker.tags,
            error_report=error_report,
            status=status,
            api_key=self.tracker.api_key
        )

        # Send to server
        if self.tracker.enable_server_sending and not self.tracker.dry_run:
            self.tracker._send_data_to_server(call_data)
        elif self.tracker.dry_run:
            msg = (
                f"Neatlogs [Dry Run]: Processed span {span_id} for {model}. "
                f"Input Messages: {len(messages)}, Completion Length: {len(str(completion))}"
            )
            print(msg)  # Ensure visibility
            logger.info(msg)
            if messages:
                print(f"Neatlogs [Dry Run] Messages: {messages}")
                logger.debug(f"Neatlogs [Dry Run] Messages: {messages}")
