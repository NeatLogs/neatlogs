"""
Core tracking functionality for Neatlogs Tracker

This module provides the core LLM tracking functionality with OpenTelemetry
and OpenInference integration for standardized observability.
"""

import os
import json
import queue
import threading
import logging
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from opentelemetry.trace import Span
from opentelemetry.sdk.trace import ReadableSpan

import contextvars

# Context variable for agentic framework
_current_framework_ctx = contextvars.ContextVar("current_framework", default=None)

# Context variable for parent span
current_span_id_context = contextvars.ContextVar("current_span_id", default=None)


# Context variable to suppress low-level patching
_suppress_patching_ctx = contextvars.ContextVar("suppress_patching", default=False)


def set_current_framework(framework: str):
    """Set the current framework context for the current async task."""
    _current_framework_ctx.set(framework)


def get_current_framework() -> Optional[str]:
    """Get the current framework from the current async task."""
    return _current_framework_ctx.get()


def clear_current_framework():
    """Clear the current framework context."""
    _current_framework_ctx.set(None)


def suppress_patching():
    """Sets a flag to suppress low-level patching for the current async task."""
    _suppress_patching_ctx.set(True)


def release_patching():
    """Releases the suppression flag for low-level patching."""
    _suppress_patching_ctx.set(False)


def is_patching_suppressed() -> bool:
    """Checks if low-level patching is currently suppressed."""
    return _suppress_patching_ctx.get()


# Context variable for passing LangGraph node spans to provider handlers
_active_langgraph_node_span_ctx = contextvars.ContextVar(
    "active_langgraph_node_span", default=None
)


def set_active_langgraph_node_span(span: Span):
    """Set the active LangGraph node span in the current context."""
    _active_langgraph_node_span_ctx.set(span)


def get_active_langgraph_node_span() -> Optional[Span]:
    """Get the active LangGraph node span from the current context."""
    return _active_langgraph_node_span_ctx.get()


def clear_active_langgraph_node_span():
    """Clear the active LangGraph node span from the current context."""
    _active_langgraph_node_span_ctx.set(None)


# Sentinel object for queue lifecycle management
_STOP = object()


@dataclass
class NewLLMCallData:
    """Data structure for LLM call information"""

    trace_id: str
    span: dict
    api_key: Optional[str] = None


class LLMTracker:
    """
    Main orchestrator for LLM tracking, logging, and reporting.

    The LLMTracker manages the lifecycle of LLM operations, from span creation
    to data collection and reporting. It handles both file-based logging and
    server-side telemetry transmission, with optional OpenTelemetry integration.

    Key Responsibilities:
    - Managing active spans and completed calls
    - Coordinating background threads for server communication
    - OpenTelemetry span creation and attribute population
    - Handling graceful shutdown procedures
    - Providing thread-safe operations for concurrent environments
    """

    def __init__(
        self,
        api_key,
        session_id=None,
        agent_id=None,
        thread_id=None,
        tags=None,
        enable_server_sending=True,
        # OpenTelemetry options
        enable_otel: bool = True,  # Default to True now as it's the core engine
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
        otel_console_export: bool = False,
        dry_run: bool = False,
    ):
        self.session_id = session_id or str(uuid4())
        self.agent_id = agent_id or "default-agent"
        self.thread_id = thread_id or str(uuid4())
        self.tags = tags or []
        self.api_key = api_key

        # Dry run configuration overrides
        self.dry_run = dry_run
        if self.dry_run:
            logging.info(
                "Neatlogs: Dry run mode enabled. Data will NOT be sent to server."
            )
            self.enable_server_sending = False
            # Force enable OTel console export for visibility
            self.enable_otel = True
            otel_console_export = True
        else:
            self.enable_server_sending = enable_server_sending
            self.enable_otel = enable_otel

        self._threads = []

        # Queue-based sender setup
        self._send_queue = queue.Queue()
        self._sender_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._sender_thread.start()

        self.setup_logging()
        self._lock = threading.Lock()

        # OpenTelemetry configuration
        self._tracer = None
        self._tracer_provider = None

        if self.enable_otel:
            self._setup_otel(otlp_endpoint, otlp_headers, otel_console_export)

        logging.info(
            f"LLMTracker initialized - Session: {self.session_id}, "
            f"Agent: {self.agent_id}, Thread: {self.thread_id}, OTel: {self.enable_otel}, DryRun: {self.dry_run}"
        )

    def _setup_otel(
        self,
        otlp_endpoint: Optional[str],
        otlp_headers: Optional[Dict[str, str]],
        console_export: bool,
    ):
        """
        Configure OpenTelemetry.

        This method attempts to attach the NeatlogsSpanProcessor to the current
        global TracerProvider. If no provider is configured (i.e., it's the default
        ProxyTracerProvider), it initializes a new SDK TracerProvider and sets it
        as global.
        """

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from .otel.new_processor import NeatlogsSpanProcessor

            # Check if a global provider is already set
            current_provider = trace.get_tracer_provider()

            # If it's a ProxyTracerProvider, it means it hasn't been configured yet.
            # (Note: We check class name to avoid importing ProxyTracerProvider which might be internal)
            is_proxy = current_provider.__class__.__name__ == "ProxyTracerProvider"

            if is_proxy:
                logging.info(
                    "Neatlogs: No global TracerProvider detected. Initializing new SDK TracerProvider."
                )

                # Create Resource
                resource = Resource.create(
                    {
                        "service.name": "neatlogs",
                        "service.version": "1.1.7",
                        "neatlogs.session_id": self.session_id,
                        "neatlogs.agent_id": self.agent_id,
                        "neatlogs.thread_id": self.thread_id,
                    }
                )

                # Initialize new TracerProvider
                self._tracer_provider = TracerProvider(resource=resource)
                self._owns_tracer_provider = True

                # Set as global
                trace.set_tracer_provider(self._tracer_provider)
            else:
                logging.info(
                    "Neatlogs: Detected existing global TracerProvider. Attaching to it."
                )
                self._tracer_provider = current_provider
                self._owns_tracer_provider = False

            # Add Neatlogs Span Processor
            # This captures data for the Neatlogs backend
            if hasattr(self._tracer_provider, "add_span_processor"):
                self._tracer_provider.add_span_processor(NeatlogsSpanProcessor(self))
            else:
                logging.warning(
                    "Neatlogs: Current TracerProvider does not support adding span processors. Neatlogs data capture may fail."
                )

            # Configure Exporters (only if we created the provider OR if user explicitly asked for them)
            # If we attached to an existing provider, we generally assume the user configured their own exporters.
            # BUT, if the user passed `otlp_endpoint` to neatlogs.init(), they probably expect us to configure it.

            if otlp_endpoint:
                if hasattr(self._tracer_provider, "add_span_processor"):
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=otlp_endpoint, headers=otlp_headers or {}
                    )
                    self._tracer_provider.add_span_processor(
                        BatchSpanProcessor(otlp_exporter)
                    )
                    logging.info(
                        f"Neatlogs OTel: Added OTLP exporter for {otlp_endpoint}"
                    )

            if console_export:
                if hasattr(self._tracer_provider, "add_span_processor"):
                    console_exporter = ConsoleSpanExporter()
                    self._tracer_provider.add_span_processor(
                        BatchSpanProcessor(console_exporter)
                    )
                    logging.info("Neatlogs OTel: Added Console exporter")

            self._tracer = trace.get_tracer("neatlogs")
            logging.info("Neatlogs: OpenTelemetry setup complete.")

        except ImportError as e:
            logging.error(f"Neatlogs: Failed to import OpenTelemetry components: {e}")
            self.enable_otel = False
        except Exception as e:
            logging.error(f"Neatlogs: Failed to configure OpenTelemetry: {e}")
            self.enable_otel = False

    def _send_to_server_sync(self, data: NewLLMCallData):
        if not self.enable_server_sending:
            return

        try:
            url = os.getenv("NEATLOGS_API_URL", "https://app.neatlogs.com/api/data/v3")

            payload = {
                "projectAPIKey": data.api_key or self.api_key,
                "externalTraceId": data.trace_id,
                "dataDump": data.span,
                "timestamp": datetime.now().timestamp(),
            }

            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
            response.raise_for_status()

        except Exception as e:
            logging.error(f"Neatlogs: Failed to send span: {e}")

    def _enqueue_span(self, call_data: NewLLMCallData):
        self._send_queue.put(call_data)

    def _send_worker(self):
        while True:
            item = self._send_queue.get()
            if item is _STOP:
                self._send_queue.task_done()
                break

            try:
                self._send_to_server_sync(item)
            except Exception as e:
                logging.error(f"Neatlogs sender worker error: {e}")
            finally:
                self._send_queue.task_done()

    def setup_logging(self):
        """
        Configure file-based logging for LLM calls.

        Sets up a dedicated logger that writes formatted LLM call data to a file.
        This ensures a local backup of all traces is available independent of
        server connectivity.
        """
        self.file_logger = logging.getLogger(f"llm_tracker_{self.session_id}")
        self.file_logger.setLevel(logging.INFO)
        # Remove existing handlers to avoid duplicates
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        # Note: We rely on the parent logger configuration or add a FileHandler if needed.
        # For now, we assume the user or environment configures the handlers for this logger name
        # or we might want to add a default FileHandler here if that was the original intent.
        # Based on previous code, it just set level and cleared handlers.
        # We'll stick to that but ensure it's clean.

    def log_llm_call(self, call_data: NewLLMCallData):
        """
        Log LLM call data to file and trigger server sending.

        Args:
            call_data (LLMCallData): The data to log and send.
        """
        # Log to file
        log_entry = {"event_type": "LLM_CALL", "data": asdict(call_data)}
        self.file_logger.info(json.dumps(log_entry, indent=2))

        # Send to server
        if self.enable_server_sending:
            self._enqueue_span(call_data)

    def add_tags(self, tags: List[str]):
        """Add tags to the tracker."""
        with self._lock:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
        logging.info(f"Added tags: {tags}")

    def shutdown(self):
        """
        Gracefully shutdown the tracker and clean up resources.

        This method ensures that all pending data is sent to the Neatlogs server
        and that the OpenTelemetry tracer provider is properly shut down (if owned).
        """
        logging.debug("Neatlogs: shutdown initiated")

        # Signal worker to stop
        self._send_queue.put(_STOP)

        # Wait until all queued spans are sent
        self._send_queue.join()

        # Wait for worker thread
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5)

        logging.debug("Neatlogs: shutdown completed")

        # Shutdown OpenTelemetry tracer to flush pending spans
        # We only shutdown if we own the TracerProvider (i.e., we created it).
        # If we attached to an existing global one, we don't shutdown it.
        if self.enable_otel and self._tracer_provider and self._owns_tracer_provider:
            try:
                self._tracer_provider.shutdown()
                logging.debug("Neatlogs: OpenTelemetry tracer shutdown complete")
            except Exception as e:
                logging.debug(f"Neatlogs: Error shutting down OTel tracer: {e}")

        logging.debug("Neatlogs: LLMTracker.shutdown() finished.")


# --- Global Tracker Instance and Initialization ---


_global_tracker: Optional[LLMTracker] = None
_init_lock = threading.Lock()


def get_tracker() -> Optional[LLMTracker]:
    """
    Get the global tracker instance.
    """
    return _global_tracker
