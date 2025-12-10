"""
Neatlogs - LLM Call Tracking Library
==========================================

A comprehensive LLM tracking system with OpenTelemetry and OpenInference support.
Automatically captures and logs all LLM API calls with detailed metrics.
"""

from .new_core import get_tracker, LLMTracker
import logging
import atexit
import threading
from typing import List, Optional, Dict

__version__ = "1.1.7"
__all__ = ["init", "get_tracker", "add_tags", "get_langchain_callback_handler"]

# --- Global Tracker Instance and Initialization ---

_global_tracker: Optional[LLMTracker] = None
_init_lock = threading.Lock()


def init(
    api_key: str,
    tags: Optional[List[str]] = None,
    debug: bool = False,
    # OpenTelemetry options
    enable_otel: bool = False,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    otel_console_export: bool = False,
    dry_run: bool = False,
):
    """
    Initialize the Neatlogs tracking system with optional OpenTelemetry support.

    Args:
        api_key (str): API key for the session. Will be persisted and logged.
        tags (List[str], optional): List of tags to associate with the tracking session.
        debug (bool): Enable debug logging. Defaults to False.
        enable_otel (bool): Enable OpenTelemetry tracing. Defaults to False.
        otlp_endpoint (str, optional): OTLP HTTP endpoint for exporting traces.
        otlp_headers (Dict[str, str], optional): Headers for OTLP exporter
            (e.g., {"Authorization": "Bearer xxx"}).
        otel_console_export (bool): Enable console export for debugging OTel spans.
        dry_run (bool): If True, disables sending data to Neatlogs server and enables console logging.
                        Useful for local testing and debugging. Defaults to False.

    Returns:
        LLMTracker: The initialized tracker instance.

    Example:
        >>> import neatlogs
        >>> # Basic usage
        >>> tracker = neatlogs.init(api_key="your_api_key")

        >>> # With OpenTelemetry (auto-instrumentation)
        >>> tracker = neatlogs.init(
        ...     api_key="your_api_key",
        ...     enable_otel=True,
        ... )

        >>> # Dry run mode (local testing)
        >>> tracker = neatlogs.init(api_key="test", dry_run=True)
    """

    session_id = None
    agent_id = None
    thread_id = None

    global _global_tracker

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    with _init_lock:
        if _global_tracker is None:
            _global_tracker = LLMTracker(
                api_key=api_key,
                session_id=session_id,
                agent_id=agent_id,
                thread_id=thread_id,
                tags=tags,
                # OpenTelemetry options
                enable_otel=enable_otel,
                otlp_endpoint=otlp_endpoint,
                otlp_headers=otlp_headers,
                otel_console_export=otel_console_export,
                dry_run=dry_run,
            )
            from .instrumentation import manager

            manager.instrument_all()

            # Log initialization info
            logging.info("🚀 Neatlogs Tracker initialized successfully!")
            logging.info(f"   📊 Session: {_global_tracker.session_id}")
            logging.info(f"   🤖 Agent: {_global_tracker.agent_id}")
            logging.info(f"   🧵 Thread: {_global_tracker.thread_id}")
            if tags:
                logging.info(f"   🏷️  Tags: {tags}")
            if enable_otel or dry_run:
                logging.info("   📡 OpenTelemetry: Enabled")
                if otlp_endpoint:
                    logging.info(f"   🔗 OTLP Endpoint: {otlp_endpoint}")
            if dry_run:
                logging.info("   🧪 Dry Run: Enabled (No data sent to server)")

    return _global_tracker


def get_langchain_callback_handler(
    api_key: Optional[str] = None, tags: Optional[List[str]] = None
):
    """
    Get the LangChain callback handler for Neatlogs tracking.


    This function lazily imports the callback handler to avoid triggering
    framework detection when it's not needed.

    Args:
        api_key (str, optional): API key for the tracker.
        tags (List[str], optional): Tags to associate with the tracking session.

    Returns:
        NeatlogsLangchainCallbackHandler: The callback handler instance.
    """
    from .integration.callbacks.langchain.callback import (
        NeatlogsLangchainCallbackHandler,
    )

    return NeatlogsLangchainCallbackHandler(api_key=api_key, tags=tags)


def add_tags(tags: List[str]):
    """
    Add tags to the current Neatlogs tracker.


    Args:
        tags (list): List of tags to add

    Example:
        >>> neatlogs.add_tags(["production", "customer-support", "v2.1"])
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError("Tracker not initialized. Call neatlogs.init() first.")

    tracker.add_tags(tags)


# --- Automatic Instrumentation Setup ---
# This is handled by instrument_all() called in init()


def _shutdown_neatlogs():
    """Shutdown the Neatlogs tracker and clean up resources on exit."""
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' called.")
    tracker = get_tracker()
    if tracker:
        tracker.shutdown()
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' finished.")


# Ensure that all data is sent and resources are cleaned up on exit.
atexit.register(_shutdown_neatlogs)


# Configure a default handler for the library's logger.
# This prevents "No handler found" warnings if the user of the library
# does not configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
