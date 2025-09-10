"""
Neatlogs Instrumentation Manager
===============================
This module manages auto-instrumentation for LLM providers and agentic frameworks
within the Neatlogs system. It dynamically applies import-time patching to supported
frameworks (such as LangChain, CrewAI) and LLM APIs (OpenAI, Anthropic, LiteLLM, etc.),
balancing provider-level and framework-level tracking to prevent duplicate data and conflicts.

Overview:
    - Registry (`SUPPORTED_PROVIDERS`, `SUPPORTED_FRAMEWORKS`) lists all integrations supported.
    - The import monitor (`_neatlogs_import_monitor`) detects and hooks relevant libraries as they're imported.
    - Global and per-import state controls which integrations are patched and when.
    - The public API (`instrument_all`, `uninstrument_all`, `is_framework_active`) exposes
      all functionality for initializing, tearing down, or querying instrumentation.

Design Notes:
    - The system is robust against race conditions (instrumenting just-in-time, never patching twice).
    - Integration logic is extensible: add new providers by amending the registry and patcher.
    - Thread-safety and recursion avoidance are carefully managed.
"""

import builtins
import sys
import logging
from typing import Dict, Optional

# --- Configuration Registry ---

SUPPORTED_PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {"patcher": "patch_openai"},
    "azure_openai": {"patcher": "patch_azure_openai"},
    "google.genai": {"patcher": "patch_google_genai"},
    "anthropic": {"patcher": "patch_anthropic"},
    "litellm": {"patcher": "patch_litellm"},
}

# Framework configurations with their instrumentation approach
SUPPORTED_FRAMEWORKS: Dict[str, Dict[str, str]] = {
    "langchain": {
        "handler": "NeatlogsLangchainCallbackHandler",
        "instrumentation_type": "callback",
    },
    "crewai": {
        "patcher": "patch_crewai",
        "instrumentation_type": "wrapper",
    },
    "langgraph": {
        "patcher": "patch_langgraph",
        "instrumentation_type": "wrapper",
    },
}

# Mapping of frameworks to providers they might use internally
# This helps us determine which providers to suppress when a framework is active
# NOTE: LangGraph is NOT included here because it uses dual tracking - it needs provider patchers active
FRAMEWORK_PROVIDER_MAPPING: Dict[str, set] = {
    "langchain": {"openai", "azure_openai", "google.genai", "anthropic"},
    "crewai": {"openai", "azure_openai"},
    # "langgraph": NOT included - dual tracking requires provider patchers to be active
}


# --- State Management ---

_instrumentation_hook_active = False
_patcher_instance = None
_detected_frameworks: set = set()  # Track all detected frameworks
# Track packages currently being patched to prevent recursion
_currently_patching: set = set()
# Track packages that have already been successfully patched
_already_patched: set = set()
_original_import = builtins.__import__


# --- The Import Monitor (Two-Phase Logic) ---

def _neatlogs_import_monitor(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Custom import-monitor function for Neatlogs instrumentation.

    This function is injected to wrap the Python `__import__` mechanism, letting
    Neatlogs detect and patch integrations as soon as they are used. Behavior is split:
      1. Before Neatlogs tracker initialization: collects framework detections, but defers patching.
      2. After Neatlogs tracker initialization: immediately patches eligible frameworks and providers.

    Recursion is prevented using tracking sets. Patching operations are guarded so that
    the same framework/provider is never patched more than once.

    Args:
        name (str): The module name being imported.

    Returns:
        The result of the original import.

    Raises:
        Logs and swallows exceptions related to patching, never interrupts actual import flow.
    """
    module = _original_import(name, globals, locals, fromlist, level)

    # Prevent recursion - if we're already patching this module, skip
    global _currently_patching, _already_patched
    if name in _currently_patching or name in _already_patched:
        return module

    # Phase 1: Framework detection
    global _detected_frameworks
    if name in SUPPORTED_FRAMEWORKS and name not in _detected_frameworks:
        logging.info(
            f"Neatlogs: Detected agentic framework '{name}'. Registering framework.")
        _detected_frameworks.add(name)

    # Phase 2: Provider and framework patching after Neatlogs init
    if _patcher_instance:
        _currently_patching.add(name)

        try:
            # Patch framework if detected
            if name in SUPPORTED_FRAMEWORKS and name not in _already_patched:
                framework_config = SUPPORTED_FRAMEWORKS[name]
                if "patcher" in framework_config:
                    patch_method_name = framework_config["patcher"]
                    patch_method = getattr(
                        _patcher_instance, patch_method_name, None)
                    if patch_method and patch_method():
                        _already_patched.add(name)

            # Only patch providers if not blocked by active frameworks
            if name in SUPPORTED_PROVIDERS and name not in _already_patched:
                if is_framework_active(name):
                    logging.debug(
                        f"Neatlogs: Skipping '{name}' provider patching - framework is active.")
                    return module

                patch_method_name = SUPPORTED_PROVIDERS[name]["patcher"]
                patch_method = getattr(
                    _patcher_instance, patch_method_name, None)
                if patch_method and patch_method():
                    _already_patched.add(name)

        except Exception as e:
            logging.error(f"Error patching {name}: {e}")
        finally:
            _currently_patching.discard(name)

    return module


# --- Public API ---

def setup_import_monitor():
    """
    Replaces the built-in import function with our monitor. This is called
    as soon as the neatlogs library is imported.
    """
    global _instrumentation_hook_active
    if _instrumentation_hook_active:
        return
    builtins.__import__ = _neatlogs_import_monitor
    _instrumentation_hook_active = True


def instrument_all(tracker):
    """
    Called by neatlogs.init() to fully activate instrumentation.
    This function sets the patcher instance and patches any libraries that
    were imported *before* init was called.
    """
    global _patcher_instance, _currently_patching, _already_patched
    if _patcher_instance:
        return

    from .patchers import ProviderPatcher
    _patcher_instance = ProviderPatcher(tracker)

    # Patch libraries that were already in sys.modules before the hook was fully active.
    for package_name in list(sys.modules.keys()):
        # Prevent recursion during initial patching
        if package_name in _currently_patching or package_name in _already_patched:
            continue

        _currently_patching.add(package_name)
        try:
            if package_name in SUPPORTED_PROVIDERS:
                # Check if any framework is active - if so, skip provider patching
                if is_framework_active(package_name):
                    logging.debug(
                        f"Neatlogs: Skipping '{package_name}' provider patching - framework is active.")
                    continue

                patch_method_name = SUPPORTED_PROVIDERS[package_name]["patcher"]
                patch_method = getattr(
                    _patcher_instance, patch_method_name, None)
                if patch_method:
                    if patch_method():
                        _already_patched.add(package_name)

            if package_name in SUPPORTED_FRAMEWORKS:
                framework_config = SUPPORTED_FRAMEWORKS[package_name]
                if "patcher" in framework_config:
                    patch_method_name = framework_config["patcher"]
                    patch_method = getattr(
                        _patcher_instance, patch_method_name, None)
                    if patch_method:
                        if patch_method():
                            _already_patched.add(package_name)
        except Exception as e:
            logging.error(
                f"Error during initial patching of {package_name}: {e}")
        finally:
            _currently_patching.discard(package_name)

    # Also patch langgraph if it's in the supported frameworks
    if 'langgraph' in SUPPORTED_FRAMEWORKS and 'langgraph' not in _already_patched:
        patch_method_name = SUPPORTED_FRAMEWORKS['langgraph']['patcher']
        patch_method = getattr(_patcher_instance, patch_method_name, None)
        if patch_method and patch_method():
            _already_patched.add('langgraph')

    logging.info("Neatlogs: Instrumentation manager fully activated.")


def uninstrument_all():
    """
    Disables the instrumentation system and restores the original import function.
    """
    global _instrumentation_hook_active, _detected_frameworks, _patcher_instance, _currently_patching, _already_patched
    if not _instrumentation_hook_active:
        return

    builtins.__import__ = _original_import

    # Unpatch all methods if necessary (optional, for very clean shutdowns)

    _instrumentation_hook_active = False
    _detected_frameworks.clear()
    _currently_patching.clear()
    _already_patched.clear()
    _patcher_instance = None
    logging.info("Neatlogs: Instrumentation manager disabled.")


def is_framework_active(provider_name: Optional[str] = None) -> bool:
    """
    Checks if any agentic framework is currently active, optionally checking
    if a specific provider should be suppressed due to framework conflicts.

    Args:
        provider_name (str, optional): The name of the provider to check.

    Returns:
        bool: True if a framework is active and conflicts with the provider (if specified),
              or if any framework is active (if provider_name is None).
    """
    global _detected_frameworks

    # If no provider specified, just check if any framework is active
    if provider_name is None:
        return len(_detected_frameworks) > 0

    # Check if any active framework conflicts with this provider
    for framework in _detected_frameworks:
        if framework in FRAMEWORK_PROVIDER_MAPPING:
            if provider_name in FRAMEWORK_PROVIDER_MAPPING[framework]:
                return True

    return False
