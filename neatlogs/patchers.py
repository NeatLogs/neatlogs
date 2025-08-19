"""
Provider patchers for NeatLogs Tracker
=====================================
Handles automatic patching of LLM providers to enable tracking.
Enhanced with comprehensive streaming support and improved API coverage.
"""
import logging
from functools import wraps
from .event_handlers import get_handler_for_provider
from .core import set_current_framework, clear_current_framework
# from .token_counting import TokenUsageExtractor, estimate_cost


class ProviderPatcher:
    def __init__(self, tracker):
        self.tracker = tracker
        self.original_methods = {}

    def patch_google_genai(self):
        """Patch Google GenAI to automatically track calls using the event handler."""
        patcher_self = self
        try:
            from google import genai
            if hasattr(genai.Client, '_neatlogs_patched_init'):
                return True

            original_client_init = genai.Client.__init__

            @wraps(original_client_init)
            def tracked_client_init(client_self, *args, **kwargs):
                original_client_init(client_self, *args, **kwargs)

                # Ensure we don't patch the same instance multiple times
                if hasattr(client_self, '_neatlogs_patched_methods'):
                    return

                handler = get_handler_for_provider(
                    "google_genai", patcher_self.tracker)

                # Patch generate_content for regular calls
                if hasattr(client_self.models, 'generate_content'):
                    original_method = client_self.models.generate_content
                    if not hasattr(original_method, '_neatlogs_patched'):
                        tracked_method = handler.wrap_method(
                            original_method, "google")
                        client_self.models.generate_content = tracked_method
                        setattr(client_self.models.generate_content,
                                '_neatlogs_patched', True)

                # Patch generate_content with stream=True for streaming calls
                if hasattr(client_self.models, 'generate_content'):
                    original_streaming_method = client_self.models.generate_content
                    if not hasattr(original_streaming_method, '_neatlogs_patched_stream'):
                        # The original method is the same, but we wrap it for streaming
                        # The handler will check for the `stream=True` kwarg
                        # This is a simplified approach; a more robust solution might need to inspect kwargs
                        # but for Google GenAI, the handler logic should suffice.
                        pass  # The wrap_method already handles both cases by checking for a generator response

                client_self._neatlogs_patched_methods = True

            genai.Client.__init__ = tracked_client_init
            genai.Client._neatlogs_patched_init = True
            self.original_methods['genai.Client.__init__'] = original_client_init

            logging.info("Successfully patched Google GenAI")
            return True
        except ImportError:
            logging.debug("Google GenAI not installed, skipping patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch Google GenAI: {e}", exc_info=True)
            return False

    def patch_litellm(self):
        """Patch LiteLLM to automatically track calls"""
        patcher_self = self
        try:
            import litellm
            if hasattr(litellm, '_neatlogs_patched'):
                return True

            original_completion = litellm.completion
            if hasattr(original_completion, '_neatlogs_patched'):
                return True

            handler = get_handler_for_provider("litellm", patcher_self.tracker)
            tracked_completion = handler.wrap_method(
                original_completion, "litellm")

            @wraps(original_completion)
            def tracked_completion_with_model(*args, **kwargs):
                model = kwargs.get('model', 'unknown')
                if hasattr(patcher_self.tracker, 'last_used_models'):
                    patcher_self.tracker.last_used_models['litellm'] = model
                return tracked_completion(*args, **kwargs)

            setattr(tracked_completion_with_model, '_neatlogs_patched', True)
            litellm.completion = tracked_completion_with_model
            litellm._neatlogs_patched = True
            self.original_methods['litellm.completion'] = original_completion
            logging.info("Successfully patched LiteLLM")
            return True
        except ImportError:
            logging.debug("LiteLLM not installed, skipping patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch LiteLLM: {e}")
            return False

    def _patch_client(self, client_class, provider_name, framework=None):
        """A helper function to patch OpenAI-compatible client classes."""
        patcher_self = self

        if hasattr(client_class, '_neatlogs_patched'):
            return

        original_init = client_class.__init__

        @wraps(original_init)
        def tracked_init(client_self, *args, **kwargs):
            result = original_init(client_self, *args, **kwargs)
            # Patch chat.completions.create
            if hasattr(client_self.chat, 'completions') and not hasattr(client_self.chat.completions, '_neatlogs_patched'):
                original_create = client_self.chat.completions.create
                handler = get_handler_for_provider(
                    provider_name, patcher_self.tracker)

                # Wrap for regular and streaming calls
                tracked_create = handler.wrap_method(
                    original_create, provider_name, framework=framework)

                @wraps(original_create)
                def tracked_create_with_model(*args, **kwargs):
                    model = kwargs.get('model', 'unknown')
                    # Always set framework, even if None
                    fw = framework if framework is not None else None
                    if hasattr(patcher_self.tracker, 'last_used_models'):
                        patcher_self.tracker.last_used_models[provider_name] = model

                    if kwargs.get('stream'):
                        span = self.tracker.start_llm_span(
                            model=model, provider=provider_name, framework=fw)
                        from .stream_wrapper import NeatLogsStreamWrapper
                        return NeatLogsStreamWrapper(tracked_create(*args, **kwargs), span, kwargs)
                    else:
                        return tracked_create(*args, **kwargs)

                client_self.chat.completions.create = tracked_create_with_model
                client_self.chat.completions._neatlogs_patched = True

            # Patch beta.chat.completions.parse if present
            if hasattr(client_self, 'beta') and hasattr(client_self.beta, 'chat') and hasattr(client_self.beta.chat, 'completions'):
                completions_obj = client_self.beta.chat.completions
                if hasattr(completions_obj, 'parse') and not hasattr(completions_obj.parse, '_neatlogs_patched'):
                    original_parse = completions_obj.parse
                    handler = get_handler_for_provider(
                        provider_name, patcher_self.tracker)

                    tracked_parse = handler.wrap_method(
                        original_parse, provider_name, framework=framework)

                    @wraps(original_parse)
                    def tracked_parse_with_model(*args, **kwargs):
                        model = kwargs.get('model', 'unknown')
                        fw = framework if framework is not None else None
                        if hasattr(patcher_self.tracker, 'last_used_models'):
                            patcher_self.tracker.last_used_models[provider_name] = model
                        # No stream support for parse (as of now)
                        return tracked_parse(*args, **kwargs)

                    completions_obj.parse = tracked_parse_with_model
                    setattr(completions_obj.parse, '_neatlogs_patched', True)
                    logging.info(f"NeatLogs Tracker: Patched beta.chat.completions.parse for provider '{provider_name}'")

            return result

        client_class.__init__ = tracked_init
        client_class._neatlogs_patched = True
        self.original_methods[f'{provider_name}.{client_class.__name__}.__init__'] = original_init

    def patch_openai(self):
        """Patch OpenAI to automatically track calls"""
        try:
            import openai
            # Patch OpenAI client classes only if not already patched
            if hasattr(openai, 'OpenAI') and not getattr(openai.OpenAI, '_neatlogs_patched', False):
                self._patch_client(openai.OpenAI, "openai")
            if hasattr(openai, 'AsyncOpenAI') and not getattr(openai.AsyncOpenAI, '_neatlogs_patched', False):
                self._patch_client(openai.AsyncOpenAI, "openai")

            # Legacy patch
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

            logging.info("Successfully patched OpenAI")
            return True
        except ImportError:
            logging.debug("OpenAI not installed, skipping patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch OpenAI: {e}")
            return False

    def patch_azure_openai(self):
        """Patch Azure OpenAI to automatically track calls"""
        try:
            import openai
            if hasattr(openai, 'AzureOpenAI'):
                self._patch_client(openai.AzureOpenAI, "azure")
            if hasattr(openai, 'AsyncAzureOpenAI'):
                self._patch_client(openai.AsyncAzureOpenAI, "azure")
            logging.info("Successfully patched Azure OpenAI")
            return True
        except ImportError:
            logging.debug("OpenAI library not installed, skipping Azure patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch Azure OpenAI: {e}")
            return False

    def patch_anthropic(self):
        """Patch Anthropic to automatically track calls including streaming"""
        patcher_self = self
        try:
            import anthropic

            def _patch_anthropic_client(client_class):
                if hasattr(client_class, '_neatlogs_patched'):
                    return

                original_init = client_class.__init__

                @wraps(original_init)
                def tracked_init(client_self, *args, **kwargs):
                    result = original_init(client_self, *args, **kwargs)
                    handler = get_handler_for_provider(
                        "anthropic", patcher_self.tracker)

                    # Patch messages.create
                    if hasattr(client_self.messages, 'create') and not hasattr(client_self.messages.create, '_neatlogs_patched'):
                        original_create = client_self.messages.create
                        tracked_create = handler.wrap_method(
                            original_create, "anthropic")

                        @wraps(original_create)
                        def tracked_create_wrapper(*args, **kwargs):
                            model = kwargs.get('model', 'unknown')
                            if hasattr(patcher_self.tracker, 'last_used_models'):
                                patcher_self.tracker.last_used_models['anthropic'] = model

                            if kwargs.get('stream'):
                                span = self.tracker.start_llm_span(
                                    model=model, provider='anthropic', framework=None)
                                from .stream_wrapper import NeatLogsStreamWrapper
                                return NeatLogsStreamWrapper(tracked_create(*args, **kwargs), span, kwargs)
                            else:
                                return tracked_create(*args, **kwargs)

                        client_self.messages.create = tracked_create_wrapper
                        setattr(client_self.messages.create,
                                '_neatlogs_patched', True)

                    return result

                client_class.__init__ = tracked_init
                client_class._neatlogs_patched = True
                self.original_methods[f'anthropic.{client_class.__name__}.__init__'] = original_init

            if hasattr(anthropic, 'Anthropic'):
                _patch_anthropic_client(anthropic.Anthropic)
            if hasattr(anthropic, 'AsyncAnthropic'):
                _patch_anthropic_client(anthropic.AsyncAnthropic)

            logging.info(
                "Successfully patched Anthropic (including streaming)")
            return True
        except ImportError:
            logging.debug("Anthropic not installed, skipping patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch Anthropic: {e}")
            return False

    def patch_crewai(self):
        """Patch CrewAI to automatically track calls and set the framework."""
        patcher_self = self
        try:
            import crewai
            if hasattr(crewai.Crew, '_neatlogs_patched_kickoff'):
                return True

            original_kickoff = crewai.Crew.kickoff

            @wraps(original_kickoff)
            def tracked_kickoff(crew_self, *args, **kwargs):
                # Set thread-local context for CrewAI framework
                logging.debug(
                    "NeatLogs: Setting thread-local framework context to 'crewai' before Crew.kickoff")
                set_current_framework("crewai")
                try:
                    logging.debug("NeatLogs: Calling Crew.kickoff (patched)")
                    result = original_kickoff(crew_self, *args, **kwargs)
                    logging.debug("NeatLogs: Crew.kickoff completed")
                    return result
                finally:
                    logging.debug(
                        "NeatLogs: Clearing thread-local framework context after Crew.kickoff")
                    clear_current_framework()

            crewai.Crew.kickoff = tracked_kickoff
            crewai.Crew._neatlogs_patched_kickoff = True
            self.original_methods['crewai.Crew.kickoff'] = original_kickoff
            logging.info("Successfully patched CrewAI")
            return True
        except ImportError:
            logging.debug("CrewAI not installed, skipping patch")
            return False
        except Exception as e:
            logging.error(f"Failed to patch CrewAI: {e}")
            return False

    def patch_all(self):
        """Patch all available providers"""
        results = {}

        def is_package_available(pkg_name):
            try:
                import importlib.util
                return importlib.util.find_spec(pkg_name) is not None
            except ImportError:
                return False

        # The order is important here. Patch base classes before subclasses if they are in different calls.
        # However, with the new _patch_client helper, we patch the specific class, so order is less critical.

        if is_package_available("openai"):
            # Azure depends on OpenAI, but we patch them separately to track different providers.
            # The _neatlogs_patched flag prevents re-patching the same methods.
            results['azure'] = self.patch_azure_openai()
            results['openai'] = self.patch_openai()
        else:
            results['openai'] = False
            results['azure'] = False

        if is_package_available("google.genai"):
            results['google_genai'] = self.patch_google_genai()
        else:
            results['google_genai'] = False

        if is_package_available("litellm"):
            results['litellm'] = self.patch_litellm()
        else:
            results['litellm'] = False

        if is_package_available("anthropic"):
            results['anthropic'] = self.patch_anthropic()
        else:
            results['anthropic'] = False

        if is_package_available("crewai"):
            results['crewai'] = self.patch_crewai()
        else:
            results['crewai'] = False

        # Log summary
        summary_lines = []
        for provider, enabled in results.items():
            if enabled:
                summary_lines.append(
                    f"🖇 NeatLogs Tracker: {provider.replace('_', ' ').title()} instrumentation patched")

        if summary_lines:
            logging.info("\n".join(summary_lines))

        successful_patches = sum(1 for enabled in results.values() if enabled)
        logging.info(
            f"Patching complete: {successful_patches} providers families patched successfully")

        return results

    from .token_counting import TokenUsageExtractor, estimate_cost
