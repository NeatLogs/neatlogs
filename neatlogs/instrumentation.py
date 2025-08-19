"""
Instrumentation Manager for NeatLogs Tracker
==========================================

Handles the dynamic patching of libraries to enable telemetry.
"""

import sys
import builtins
import logging
from .registry import PROVIDERS, AGENTIC_LIBRARIES


class InstrumentationManager:
    def __init__(self, patcher):
        self.patcher = patcher
        self.instrumented_providers = set()
        self.instrumented_frameworks = set()
        self.original_import = builtins.__import__
        self._currently_patching = set()  # Track modules currently being patched

    def instrument_all(self):
        """Replace the built-in import and patch already-imported modules."""
        builtins.__import__ = self._import_monitor
        self._patch_existing_modules()
        # Framework detector removed: no import hook or scan needed.

    def uninstrument_all(self):
        """Restore the original import function."""
        builtins.__import__ = self.original_import
        # Framework detector removed: no stop_detection needed.

    def _patch_existing_modules(self):
        """Scan sys.modules and patch any supported libraries that are already loaded."""
        for module_name in list(sys.modules.keys()):
            self._check_and_patch(module_name)

    def _import_monitor(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Custom import function to patch modules as they are imported."""
        # If we're already patching this module, skip to prevent recursion
        if name in self._currently_patching:
            return self.original_import(name, globals, locals, fromlist, level)
        module = self.original_import(name, globals, locals, fromlist, level)
        self._check_and_patch(name)
        return module

    def _check_and_patch(self, module_name):
        """Check if a module is a target for instrumentation and patch it."""
        if module_name in self._currently_patching:
            return  # Already patching this module, prevent recursion
        self._currently_patching.add(module_name)
        try:
            # Prioritize agentic frameworks
            if module_name in AGENTIC_LIBRARIES and module_name not in self.instrumented_frameworks:
                patch_method_name = AGENTIC_LIBRARIES[module_name]
                if hasattr(self.patcher, patch_method_name):
                    getattr(self.patcher, patch_method_name)()
                    self.instrumented_frameworks.add(module_name)
                    logging.info(f"Instrumented framework: {module_name}")

            # Patch providers if no framework is handling them
            if module_name in PROVIDERS and module_name not in self.instrumented_providers:
                patch_method_name = PROVIDERS[module_name]
                if hasattr(self.patcher, patch_method_name):
                    getattr(self.patcher, patch_method_name)()
                    self.instrumented_providers.add(module_name)
                    logging.info(f"Instrumented provider: {module_name}")
        finally:
            self._currently_patching.remove(module_name)
