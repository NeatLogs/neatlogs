"""
LangChain integration for Neatlogs.
"""


def NeatlogsLangchainCallbackHandler(*args, **kwargs):
    """Lazily import and return the LangChain callback handler"""
    from neatlogs.integration.callbacks.langchain.callback import NeatlogsLangchainCallbackHandler as Handler
    return Handler(*args, **kwargs)


def AsyncNeatlogsLangchainCallbackHandler(*args, **kwargs):
    """Lazily import and return the async LangChain callback handler"""
    from neatlogs.integration.callbacks.langchain.callback import AsyncNeatlogsLangchainCallbackHandler as Handler
    return Handler(*args, **kwargs)


__all__ = [
    "NeatlogsLangchainCallbackHandler",
    "AsyncNeatlogsLangchainCallbackHandler",
]
