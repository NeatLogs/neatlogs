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

    def _patch_openai_classes(self, provider_name):
        """A helper to patch the underlying classes for OpenAI and Azure OpenAI."""
        try:
            # Try the modern approach first - patch client instances
            import openai

            # Check if we have the modern OpenAI or AzureOpenAI classes
            client_classes = []
            if hasattr(openai, 'OpenAI'):
                client_classes.append(openai.OpenAI)
            if hasattr(openai, 'AzureOpenAI'):
                client_classes.append(openai.AzureOpenAI)

            if not client_classes:
                logging.debug(
                    f"Neatlogs: No OpenAI client classes found for {provider_name}")
                return False

            handler = get_handler_for_provider(provider_name, self.tracker)
            patched_any = False

            # Patch each client class
            for client_class in client_classes:
                if hasattr(client_class, '_neatlogs_patched_init'):
                    continue  # Already patched

                original_init = client_class.__init__

                @wraps(original_init)
                def patched_init(client_self, *args, **kwargs):
                    # Call original init first
                    original_init(client_self, *args, **kwargs)

                    # Now patch the client instance methods
                    try:
                        # Patch chat completions create method
                        if hasattr(client_self, 'chat') and hasattr(client_self.chat, 'completions'):
                            completions_obj = client_self.chat.completions
                            if hasattr(completions_obj, 'create') and not hasattr(completions_obj.create, '_neatlogs_patched'):
                                original_create = completions_obj.create
                                wrapped_create = handler.wrap_method(
                                    original_create, provider_name)
                                completions_obj.create = wrapped_create
                                setattr(completions_obj.create,
                                        '_neatlogs_patched', True)

                        # Patch beta chat completions parse method
                        if (hasattr(client_self, 'beta') and
                            hasattr(client_self.beta, 'chat') and
                                hasattr(client_self.beta.chat, 'completions')):
                            beta_completions_obj = client_self.beta.chat.completions
                            if hasattr(beta_completions_obj, 'parse') and not hasattr(beta_completions_obj.parse, '_neatlogs_patched'):
                                original_parse = beta_completions_obj.parse
                                wrapped_parse = handler.wrap_method(
                                    original_parse, provider_name)
                                beta_completions_obj.parse = wrapped_parse
                                setattr(beta_completions_obj.parse,
                                        '_neatlogs_patched', True)

                    except Exception as e:
                        logging.debug(
                            f"Neatlogs: Error patching {client_class.__name__} instance: {e}")

                # Replace the class __init__ method
                client_class.__init__ = patched_init
                client_class._neatlogs_patched_init = True
                self.original_methods[f'{client_class.__module__}.{client_class.__name__}.__init__'] = original_init
                patched_any = True

            if patched_any:
                logging.debug(
                    f"Neatlogs: Successfully patched OpenAI/Azure client classes for provider '{provider_name}'.")
                return True
            else:
                logging.debug(
                    f"Neatlogs: No suitable client classes found for patching {provider_name}")
                return False

        except ImportError as e:
            logging.debug(
                f"Neatlogs: Failed to import OpenAI for patching {provider_name}: {e}")
            return False
        except Exception as e:
            logging.error(
                f"Failed to patch OpenAI classes for {provider_name}: {e}", exc_info=True)
            return False

    def patch_openai(self):
        """Patch OpenAI to automatically track calls by modifying the underlying classes."""
        return self._patch_openai_classes("openai")

    def patch_azure_openai(self):
        """Patch Azure OpenAI to automatically track calls by modifying the underlying classes."""
        return self._patch_openai_classes("azure")

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

    def patch_langgraph(self):
        """Patch LangGraph to automatically track workflows and nodes."""
        try:
            import langgraph.graph.state
            import langgraph.pregel
            import inspect

            handler = get_handler_for_provider("langgraph", self.tracker)

            # --- Patch StateGraph ---
            graph_cls = langgraph.graph.state.StateGraph
            if not getattr(graph_cls, '_neatlogs_patched_add_node', False):
                original_add_node = graph_cls.add_node

                @wraps(original_add_node)
                def tracked_add_node(graph_self, key, action):
                    wrapped_action = handler.wrap_node_action(key, action)
                    return original_add_node(graph_self, key, wrapped_action)
                graph_cls.add_node = tracked_add_node
                setattr(graph_cls, '_neatlogs_patched_add_node', True)
                self.original_methods['langgraph.graph.state.StateGraph.add_node'] = original_add_node

            if not getattr(graph_cls, '_neatlogs_patched_compile', False):
                original_compile = graph_cls.compile
                graph_cls.compile = handler.wrap_compile(original_compile)
                setattr(graph_cls, '_neatlogs_patched_compile', True)
                self.original_methods['langgraph.graph.state.StateGraph.compile'] = original_compile

            # --- Patch Pregel ---
            pregel_cls = langgraph.pregel.Pregel
            methods_to_patch = ["invoke", "ainvoke", "stream", "astream"]
            for method_name in methods_to_patch:
                if hasattr(pregel_cls, method_name) and not getattr(getattr(pregel_cls, method_name), '_neatlogs_patched', False):
                    original_method = getattr(pregel_cls, method_name)

                    wrapped_method = handler.wrap_method(
                        original_method, method_type=method_name)
                    setattr(wrapped_method, '_neatlogs_patched', True)
                    setattr(pregel_cls, method_name, wrapped_method)
                    self.original_methods[f'langgraph.pregel.Pregel.{method_name}'] = original_method

            logging.debug("Neatlogs: Successfully patched LangGraph")
            return True
        except ImportError:
            logging.debug("Neatlogs: LangGraph not available for patching.")
            return False
        except Exception as e:
            logging.error(f"Failed to patch LangGraph: {e}", exc_info=True)
            return False
