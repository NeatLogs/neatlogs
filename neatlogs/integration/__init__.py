"""
Integration Modules for Neatlogs
===============================
This package contains official Neatlogs integration adapters for third-party
frameworks, libraries, and tools (such as LangChain, CrewAI, etc). These modules
enable automatic and extensible telemetry for LLM- and agent-based workflows.

Usage:
    - Import the relevant integration or callback handler in your workflow.
    - See submodules under `.callbacks` for high-fidelity callback interfaces.

Extensibility:
    - To add a new integration, place your adapter in this package and document
      any custom callback or event mechanisms.

For details:
    - See `.callbacks.langchain.LANGCHAIN_README.md` for usage of the LangChain hooks.
    - Consult each submodule for provider-specific behaviors.
"""
