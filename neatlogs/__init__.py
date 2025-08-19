"""
NeatLogs Tracker - LLM Call Tracking Library
==========================================

A comprehensive LLM tracking system.
Automatically captures and logs all LLM API calls with detailed metrics.
"""

from .core import LLMTracker
from .patchers import ProviderPatcher
from .instrumentation import InstrumentationManager
import logging
import atexit

__version__ = "1.0.0"
__all__ = ['init', 'get_tracker', 'end_session',
           'get_session_stats', 'add_tags']

# Global tracker instance
_global_tracker = None
_instrumentation_manager = None


def init(api_key, debug=False, tags=None):
    """
    Initialize the LLM tracking system.

    Args:
        api_key (str): API key for the session. Will be persisted and logged.
        tags (list, optional): List of tags to associate with the tracking session.
        debug (bool): Enable debug logging. Defaults to False.

    Example:
        >>> import neatlogs
        >>> neatlogs.init(
        ...     api_key="project_key",
        ...     tags=["tag1", "tag2"]
        ... )
        >>> # Now all calls are automatically tracked!
    """

    session_id = None
    agent_id = None
    thread_id = None
    auto_patch = True

    global _global_tracker

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # Create the tracker
    _global_tracker = LLMTracker(
        api_key=api_key,
        session_id=session_id,
        agent_id=agent_id,
        thread_id=thread_id,
        tags=tags or [],
        enable_server_sending=True  # Enable server sending by default
    )
    atexit.register(_global_tracker.shutdown)

    enabled_providers = {}
    last_used_models = {}
    # Auto-patch providers if requested
    if auto_patch:
        patcher = ProviderPatcher(_global_tracker)
        _instrumentation_manager = InstrumentationManager(patcher)
        _instrumentation_manager.instrument_all()
        enabled_providers = {
            provider: True for provider in _instrumentation_manager.instrumented_providers}
        enabled_providers.update(
            {framework: True for framework in _instrumentation_manager.instrumented_frameworks})
        for provider in enabled_providers:
            last_used_models[provider] = None

    # Prepare summary string for enabled providers
    summary_lines = []
    for provider, enabled in enabled_providers.items():
        if enabled:
            summary_lines.append(
                f"🖇 NeatLogs Tracker: {provider.replace('_', ' ').title()} instrumentation patched")
    summary = "\\n".join(summary_lines)

    # Log the summary to console and to log file
    logging.info("🚀 NeatLogs Tracker initialized successfully!")
    logging.info(f"   📊 Session: {_global_tracker.session_id}")
    logging.info(f"   🤖 Agent: {_global_tracker.agent_id}")
    logging.info(f"   🧵 Thread: {_global_tracker.thread_id}")
    if summary:
        logging.info(summary)

    # Also write the full summary to the log file
    # try:
    #     with open(log_file, "a", encoding="utf-8") as f:
    #         f.write(f"NeatLogs Tracker Initialization Summary:\\n")
    #         f.write(f"Session: {_global_tracker.session_id}\\n")
    #         f.write(f"Agent: {_global_tracker.agent_id}\\n")
    #         f.write(f"Thread: {_global_tracker.thread_id}\\n")
    #         if _global_tracker.api_key:
    #             f.write(f"API Key: {_global_tracker.api_key}\\n")
    #         if summary:
    #             f.write(summary + "\\n")
    #         f.write("\\n")
    # except Exception as e:
    #     logging.error(
    #         f"Failed to write initialization summary to log file: {e}")

        return _global_tracker


def get_tracker():
    """
    Get the current global tracker instance.

    Returns:
        LLMTracker: Current tracker instance or None if not initialized.

    Example:
        >>> tracker = neatlogs.get_tracker()
        >>> if tracker:
        ...     stats = tracker.get_session_stats()
        ...     print(f"Total calls: {stats['total_calls']}")
    """
    return _global_tracker


def get_session_stats():
    """
    Get statistics for the current tracking session.

    Returns:
        dict: Session statistics including calls, costs, tokens, etc.
        None: If no tracker is initialized.

    Example:
        >>> stats = neatlogs.get_session_stats()
        >>> print(f"Total LLM calls: {stats['total_calls']}")
        >>> print(f"Total cost: ${stats['total_cost']:.4f}")
    """
    if _global_tracker:
        return _global_tracker.get_session_stats()
    return None


def end_session():
    """
    End the current tracking session and cleanup.

    This will restore any patched methods and cleanup resources.

    Example:
        >>> neatlogs.end_session()
        >>> # Tracking is now stopped
    """
    global _global_tracker, _instrumentation_manager

    if _instrumentation_manager:
        _instrumentation_manager.uninstrument_all()
        _instrumentation_manager = None

    if _global_tracker:
        # You could add cleanup logic here
        logging.info("🛑 NeatLogs Tracker session ended")
        _global_tracker = None
    else:
        logging.warning("No active tracking session to end")


# Convenience functions for manual tracking (advanced usage)
def start_span(model=None, provider=None, framework=None):
    """
    Manually start tracking an LLM span.

    This is for advanced usage when you want manual control.
    Most users should rely on automatic patching.

    Args:
        model (str): LLM model name
        provider (str): LLM provider name
        framework (str, optional): Framework name (e.g., 'crewai', 'langchain')

    Returns:
        LLMSpan: Active span for manual tracking
    """
    if not _global_tracker:
        raise RuntimeError(
            "Tracker not initialized. Call neatlogs.init() first.")

    return _global_tracker.start_llm_span(model=model, provider=provider, framework=framework)


def end_span(span, success=True, error=None):
    """
    Manually end tracking an LLM span.

    Args:
        span (LLMSpan): The span to end
        success (bool): Whether the operation was successful
        error (Exception, optional): Error if operation failed
    """
    if not _global_tracker:
        raise RuntimeError(
            "Tracker not initialized. Call neatlogs.init() first.")

    _global_tracker.end_llm_span(span, success=success, error=error)


def add_tags(tags):
    """
    Add tags to the current tracker.

    Args:
        tags (list): List of tags to add

    Example:
        >>> neatlogs.add_tags(["production", "customer-support", "v2.1"])
    """
    if not _global_tracker:
        raise RuntimeError(
            "Tracker not initialized. Call neatlogs.init() first.")

    _global_tracker.add_tags(tags)


# Version info
def version():
    """Get the version of neatlogs"""
    return __version__


# Health check
def is_active():
    """Check if tracking is currently active"""
    return _global_tracker is not None
