from .llm import LLMTracker


def init(api_key):
    _llm_tracker = LLMTracker(api_key = api_key)
    _llm_tracker.override_api()