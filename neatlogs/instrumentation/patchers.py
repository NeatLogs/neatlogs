"""
Provider patchers for Neatlogs Tracker
=====================================

Handles automatic patching of LLM providers to enable tracking.
This version uses a context-aware patching strategy to support multiple
frameworks like LangChain and CrewAI simultaneously without conflicts.
"""
import logging
from functools import wraps
from ..event_handlers import get_handler_for_provider
from ..core import set_current_framework, clear_current_framework
from . import manager as instrumentation


class ProviderPatcher:
    def __init__(self, tracker):
        self.tracker = tracker
        self.original_methods = {}

    def patch_google_genai(self):
        """Patch Google GenAI to automatically track calls using the event handler."""
        try:
            logging.debug("Neatlogs: Attempting to patch Google GenAI")
            import google.genai
            logging.debug(
                "Neatlogs: Successfully imported google.genai module")

            # Patch the Client class
            if hasattr(google.genai, 'Client') and not hasattr(google.genai.Client, '_neatlogs_patched'):
                original_client_init = google.genai.Client.__init__

                def tracked_client_init(client_self, *args, **kwargs):
                    original_client_init(client_self, *args, **kwargs)

                    # Patch the models.generate_content method
                    if hasattr(client_self, 'models') and hasattr(client_self.models, 'generate_content') and not hasattr(client_self.models.generate_content, '_neatlogs_patched'):
                        original_generate_content = client_self.models.generate_content
                        handler = get_handler_for_provider(
                            "google_genai", self.tracker)
                        tracked_generate_content = handler.wrap_method(
                            original_generate_content, "google")
                        client_self.models.generate_content = tracked_generate_content
                        setattr(client_self.models.generate_content,
                                '_neatlogs_patched', True)
                        logging.debug(
                            "Neatlogs: Successfully patched google.genai Client models.generate_content method")

                google.genai.Client.__init__ = tracked_client_init
                setattr(google.genai.Client, '_neatlogs_patched', True)
                self.original_methods['google.genai.Client.__init__'] = original_client_init
                logging.debug(
                    "Neatlogs: Successfully patched google.genai Client")

            logging.debug("Neatlogs: Successfully patched Google GenAI")
            return True
        except ImportError as e:
            logging.debug(
                f"Neatlogs: Google GenAI not available for patching: {e}")
            return False
        except Exception as e:
            logging.error(f"Failed to patch Google GenAI: {e}", exc_info=True)
            return False

    def patch_openai(self):
        """Patch OpenAI to automatically track calls, including legacy versions."""
        try:
            import openai

            # Check if we've already patched this module
            if hasattr(openai, '_neatlogs_patched_module'):
                return True

            # Mark the module as patched to prevent recursion
            openai._neatlogs_patched_module = True

            if hasattr(openai, 'OpenAI') and not getattr(openai.OpenAI, '_neatlogs_patched', False):
                self._patch_client(openai.OpenAI, "openai")
            if hasattr(openai, 'AsyncOpenAI') and not getattr(openai.AsyncOpenAI, '_neatlogs_patched', False):
                self._patch_client(openai.AsyncOpenAI, "openai")

            if hasattr(openai, 'ChatCompletion') and not hasattr(openai.ChatCompletion, '_neatlogs_patched'):
                original_create = openai.ChatCompletion.create
                if not hasattr(original_create, '_neatlogs_patched'):
                    handler = get_handler_for_provider("openai", self.tracker)
                    tracked_create = handler.wrap_method(
                        original_create, "openai")
                    setattr(tracked_create, '_neatlogs_patched', True)
                    openai.ChatCompletion.create = tracked_create
                    openai.ChatCompletion._neatlogs_patched = True
                    self.original_methods['openai.ChatCompletion.create'] = original_create
            return True
        except ImportError:
            return False
        except Exception as e:
            logging.error(f"Failed to patch OpenAI: {e}")
            return False

    def patch_azure_openai(self):
        """Patch Azure OpenAI to automatically track calls."""
        try:
            import openai
            if hasattr(openai, 'AzureOpenAI'):
                self._patch_client(openai.AzureOpenAI, "azure")
            if hasattr(openai, 'AsyncAzureOpenAI'):
                self._patch_client(openai.AsyncAzureOpenAI, "azure")
            return True
        except ImportError:
            logging.debug("OpenAI library not installed, skipping Azure patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch Azure OpenAI: {e}")
            return False

    def _patch_client(self, client_class, provider_name):
        """A helper function to patch OpenAI-compatible client classes."""
        if hasattr(client_class, '_neatlogs_patched'):
            return

        original_init = client_class.__init__

        @wraps(original_init)
        def tracked_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            if hasattr(client_self.chat, 'completions') and not hasattr(client_self.chat.completions, '_neatlogs_patched'):
                original_create = client_self.chat.completions.create
                handler = get_handler_for_provider(provider_name, self.tracker)

                # Check if the original method is async
                import inspect
                if inspect.iscoroutinefunction(original_create):
                    tracked_create_method = handler.wrap_async_method(
                        original_create, provider_name)
                else:
                    tracked_create_method = handler.wrap_method(
                        original_create, provider_name)

                client_self.chat.completions.create = tracked_create_method
                client_self.chat.completions._neatlogs_patched = True

        client_class.__init__ = tracked_init
        client_class._neatlogs_patched = True
        self.original_methods[f'{provider_name}.{client_class.__name__}.__init__'] = original_init

    def patch_crewai(self):
        """Patch CrewAI to set the framework context before execution."""
        try:
            import crewai
            if hasattr(crewai.Crew, '_neatlogs_patched_kickoff'):
                return True

            original_kickoff = crewai.Crew.kickoff

            @wraps(original_kickoff)
            def tracked_kickoff(crew_self, *args, **kwargs):
                set_current_framework("crewai")
                try:
                    return original_kickoff(crew_self, *args, **kwargs)
                finally:
                    clear_current_framework()

            crewai.Crew.kickoff = tracked_kickoff
            crewai.Crew._neatlogs_patched_kickoff = True
            self.original_methods['crewai.Crew.kickoff'] = original_kickoff
            return True
        except ImportError:
            return False
        except Exception as e:
            logging.error(f"Failed to patch CrewAI: {e}")
            return False

    def patch_litellm(self):
        """Patch LiteLLM to automatically track calls."""
        try:
            import litellm
            if hasattr(litellm, '_neatlogs_patched'):
                return True

            original_completion = litellm.completion
            if hasattr(original_completion, '_neatlogs_patched'):
                return True

            handler = get_handler_for_provider("litellm", self.tracker)
            tracked_completion = handler.wrap_method(
                original_completion, "litellm")
            setattr(tracked_completion, '_neatlogs_patched', True)
            litellm.completion = tracked_completion
            litellm._neatlogs_patched = True
            self.original_methods['litellm.completion'] = original_completion
            return True
        except ImportError:
            return False
        except Exception as e:
            logging.error(f"Failed to patch LiteLLM: {e}")
            return False

    def patch_anthropic(self):
        """Patch Anthropic to automatically track calls."""
        try:
            import anthropic

            def _patch_anthropic_client(client_class):
                if hasattr(client_class, '_neatlogs_patched'):
                    return

                original_init = client_class.__init__

                @wraps(original_init)
                def tracked_init(client_self, *args, **kwargs):
                    original_init(client_self, *args, **kwargs)
                    handler = get_handler_for_provider(
                        "anthropic", self.tracker)

                    if hasattr(client_self.messages, 'create') and not hasattr(client_self.messages.create, '_neatlogs_patched'):
                        original_create = client_self.messages.create
                        tracked_create_method = handler.wrap_method(
                            original_create, "anthropic")
                        client_self.messages.create = tracked_create_method
                        setattr(client_self.messages.create,
                                '_neatlogs_patched', True)

                client_class.__init__ = tracked_init
                client_class._neatlogs_patched = True
                self.original_methods[f'anthropic.{client_class.__name__}.__init__'] = original_init

            if hasattr(anthropic, 'Anthropic'):
                _patch_anthropic_client(anthropic.Anthropic)
            if hasattr(anthropic, 'AsyncAnthropic'):
                _patch_anthropic_client(anthropic.AsyncAnthropic)
            return True
        except ImportError:
            return False
        except Exception as e:
            logging.error(f"Failed to patch Anthropic: {e}")
            return False
