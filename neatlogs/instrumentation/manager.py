"""
Neatlogs Instrumentation Manager
===============================
This module manages auto-instrumentation for LLM providers and frameworks
within the Neatlogs system using OpenInference.
"""

import importlib.util
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def instrument_all(instrumentations: Optional[List[str]] = None):
    """
    Instrument all supported and available libraries.

    Args:
        instrumentations (List[str], optional): List of frameworks to instrument.
                                                If None, all available supported frameworks are instrumented.
                                                Supported: "openai", "openai-agents", "langchain", "anthropic",
                                                "google-genai", "crewai", "groq", "litellm".

    This function checks for the presence of OpenInference instrumentation libraries
    and initializes them if found.
    """

    # Helper to check if we should instrument a specific framework
    def should_instrument(name):
        if instrumentations is not None:
            return name in instrumentations
        return True

    # OpenAI Agents
    # The OpenAI Agents SDK is imported as 'agents'
    # Check for OpenAI Agents first, as it uses OpenAI SDK internally
    if should_instrument("openai-agents"):
        has_agents = importlib.util.find_spec("agents") is not None

        if has_agents:
            if importlib.util.find_spec("openinference.instrumentation.openai_agents"):
                try:
                    from openinference.instrumentation.openai_agents import (
                        OpenAIAgentsInstrumentor,
                    )

                    OpenAIAgentsInstrumentor().instrument()
                    logger.info("Neatlogs: OpenAI Agents instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument OpenAI Agents: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'agents' but 'openinference-instrumentation-openai-agents' is not installed. "
                    "Install with: uv add neatlogs[openai-agents]"
                )

    # OpenAI - instrument regardless of whether Agents is present (they can coexist)
    if should_instrument("openai"):
        if importlib.util.find_spec("openai"):
            if importlib.util.find_spec("openinference.instrumentation.openai"):
                try:
                    from openinference.instrumentation.openai import OpenAIInstrumentor

                    OpenAIInstrumentor().instrument()
                    logger.info("Neatlogs: OpenAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument OpenAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'openai' but 'openinference-instrumentation-openai' is not installed. "
                    "Install with: uv add neatlogs[openai]"
                )

    # LangChain
    if should_instrument("langchain"):
        if importlib.util.find_spec("langchain"):
            if importlib.util.find_spec("openinference.instrumentation.langchain"):
                try:
                    from openinference.instrumentation.langchain import (
                        LangChainInstrumentor,
                    )

                    LangChainInstrumentor().instrument()
                    logger.info("Neatlogs: LangChain instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument LangChain: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'langchain' but 'openinference-instrumentation-langchain' is not installed. "
                    "Install with: uv add neatlogs[langchain]"
                )

    # Anthropic
    if should_instrument("anthropic"):
        if importlib.util.find_spec("anthropic"):
            if importlib.util.find_spec("openinference.instrumentation.anthropic"):
                try:
                    from openinference.instrumentation.anthropic import (
                        AnthropicInstrumentor,
                    )

                    AnthropicInstrumentor().instrument()
                    logger.info("Neatlogs: Anthropic instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument Anthropic: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'anthropic' but 'openinference-instrumentation-anthropic' is not installed. "
                    "Install with: uv add neatlogs[anthropic]"
                )

    # Google GenAI
    if should_instrument("google-genai"):
        if importlib.util.find_spec("google.genai"):
            if importlib.util.find_spec("openinference.instrumentation.google_genai"):
                try:
                    from openinference.instrumentation.google_genai import (
                        GoogleGenAIInstrumentor,
                    )

                    GoogleGenAIInstrumentor().instrument()
                    logger.info("Neatlogs: Google GenAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument Google GenAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'google.genai' but 'openinference-instrumentation-google-genai' is not installed. "
                    "Install with: uv add neatlogs[google-genai]"
                )

    # CrewAI
    if should_instrument("crewai"):
        if importlib.util.find_spec("crewai"):
            if importlib.util.find_spec("openinference.instrumentation.crewai"):
                try:
                    from openinference.instrumentation.crewai import CrewAIInstrumentor

                    CrewAIInstrumentor().instrument()
                    logger.info("Neatlogs: CrewAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument CrewAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'crewai' but 'openinference-instrumentation-crewai' is not installed. "
                    "Install with: uv add neatlogs[crewai]"
                )

    # Groq
    if should_instrument("groq"):
        if importlib.util.find_spec("groq"):
            if importlib.util.find_spec("openinference.instrumentation.groq"):
                try:
                    from openinference.instrumentation.groq import GroqInstrumentor

                    GroqInstrumentor().instrument()
                    logger.info("Neatlogs: Groq instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument Groq: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'groq' but 'openinference-instrumentation-groq' is not installed. "
                    "Install with: uv add neatlogs[groq]"
                )

    # LiteLLM
    if should_instrument("litellm"):
        if importlib.util.find_spec("litellm"):
            if importlib.util.find_spec("openinference.instrumentation.litellm"):
                try:
                    from openinference.instrumentation.litellm import (
                        LiteLLMInstrumentor,
                    )

                    LiteLLMInstrumentor().instrument()
                    logger.info("Neatlogs: LiteLLM instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument LiteLLM: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'litellm' but 'openinference-instrumentation-litellm' is not installed. "
                    "Install with: uv add neatlogs[litellm]"
                )

    # Add other providers here as we add support (e.g., Bedrock, Mistral, etc.)

    logger.info("Neatlogs: Auto-instrumentation setup complete.")
