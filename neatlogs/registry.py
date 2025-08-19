"""
Registry for NeatLogs Tracker
=============================

Defines the LLM providers and agentic frameworks that NeatLogs can instrument.
This centralized registry makes it easy to manage and extend tracking capabilities.
"""

# A mapping of library names to the function that patches them.
# The key is the name of the library as it would be imported (e.g., 'openai').
# The value is the name of the method in ProviderPatcher to call for patching.

PROVIDERS = {
    'openai': 'patch_openai',
    'google.genai': 'patch_google_genai',
    'anthropic': 'patch_anthropic',
    'litellm': 'patch_litellm',
    'azure': 'patch_azure_openai',
    'cohere': 'patch_cohere',
    'huggingface': 'patch_huggingface',
    'ibm_watsonx_ai': 'patch_ibm_watsonx_ai',
}

AGENTIC_LIBRARIES = {
    'crewai': 'patch_crewai',
    'langchain': 'patch_langchain',
    'langgraph': 'patch_langgraph',
    'autogen': 'patch_autogen',
    'ag2': 'patch_ag2',
    'agents': 'patch_openai_agents',
    'smolagents': 'patch_smolagents',
    'agno': 'patch_agno',
    'google.adk': 'patch_google_adk',
}
