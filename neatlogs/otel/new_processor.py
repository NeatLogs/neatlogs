"""
Neatlogs Span Processor
=======================

Implements an OpenTelemetry SpanProcessor that captures finished spans,
extracts relevant LLM data, and sends it to the Neatlogs backend.
"""

import uuid
import logging
import traceback
from typing import Optional

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import Span

from ..new_core import LLMTracker, NewLLMCallData
from ..semconv import LLMAttributes

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
        logger.debug(f"on_end called for span: {span.name if span else 'None'}")
        if not span:
            return

        attributes = span.attributes or {}
        span_kind = attributes.get("openinference.span.kind")

        logger.debug(f"Processing span '{span.name}' with span_kind: {span_kind}")

        # Process all OpenInference spans (LLM, TOOL, AGENT, CHAIN, etc.)
        # Skip spans without openinference.span.kind as they're likely infrastructure spans
        if not span_kind:
            return

        try:
            self._process_span(span)
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

    def _process_span(self, span: ReadableSpan):
        """Extract data from span and send to Neatlogs."""

        new_call_data = NewLLMCallData(
            span=span, trace_id=str(uuid.uuid4()), api_key=self.tracker.api_key
        )

        # Send to server
        if self.tracker.enable_server_sending and not self.tracker.dry_run:
            self.tracker._send_data_to_server(new_call_data)
        elif self.tracker.dry_run:
            message: str = f"Neatlogs [Dry Run]: Processed span"
            print(message)  # Ensure visibility
            logger.info(message)
