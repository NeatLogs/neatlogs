from .llm import LLMTracker

# Global instance of LLMTracker


def init(api_key):
    """
    Initialize the LLM tracker with API key.
    
    Args:
        api_key (str): The API key for authentication
    """
    
    _llm_tracker = LLMTracker(api_key=api_key)
    _llm_tracker.override_api()
    return _llm_tracker