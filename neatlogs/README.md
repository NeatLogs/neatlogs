# Neatlogs - Comprehensive LLM Call Tracking Library

## Introduction

Neatlogs is a comprehensive LLM tracking system designed to automatically capture and log all LLM API calls with detailed metrics, costs, and telemetry data. The library provides seamless instrumentation for multiple LLM providers (OpenAI, Anthropic, Google GenAI, etc.) and agentic frameworks (LangChain, CrewAI, etc.) without requiring code changes in your application.

### Key Features

- **Zero-Code Instrumentation**: Automatic tracking of LLM calls through import-time patching
- **Multi-Provider Support**: Native support for OpenAI, Anthropic, Google GenAI, LiteLLM, and Azure OpenAI
- **Framework Integration**: Built-in support for LangChain, CrewAI, and other agentic frameworks
- **Streaming Support**: Full instrumentation of streaming responses with detailed chunk tracking
- **Cost Estimation**: Automatic cost calculation based on token usage and model pricing
- **Thread-Safe Operations**: Designed for concurrent environments with proper synchronization
- **Server Telemetry**: Background transmission of telemetry data to Neatlogs servers
- **Semantic Conventions**: Standardized attribute naming for consistent data structure

## Architecture Overview

Neatlogs employs a sophisticated architecture centered around import-time patching and event-driven data collection. The system is designed to balance provider-level and framework-level tracking while preventing conflicts and duplicate data collection.

### Core Design Principles

1. **Import-Time Instrumentation**: The library sets up import hooks immediately upon import, detecting and patching supported libraries as they're loaded
2. **Two-Phase Patching**: Framework detection occurs first, followed by selective provider patching to avoid conflicts
3. **Context-Aware Tracking**: Uses context variables to track the current framework context and suppress redundant patching
4. **Event Handler Pattern**: Provider-specific event handlers extract and normalize data from different API responses
5. **Span-Based Tracking**: All operations are tracked as spans with start/end timing, metadata, and error handling

### Main Components

- **LLMTracker**: Central orchestrator managing span lifecycle, data collection, and server communication
- **Instrumentation Manager**: Coordinates import hooks, framework detection, and selective patching
- **Provider Patchers**: Wrap specific provider methods to inject tracking logic
- **Event Handlers**: Extract provider-specific data (messages, tokens, costs) from API responses
- **Callback Handlers**: Framework-specific handlers for LangChain-style callback systems
- **Stream Wrapper**: Specialized handling for streaming responses with chunk-level tracking

## File Breakdown

### Root Directory Files

| File | Purpose | Key Contents |
|------|---------|-------------|
| [`__init__.py`](neatlogs/__init__.py) | Main package entry point | Initialization API (`init()`, `get_langchain_callback_handler()`), global tracker management, automatic instrumentation setup |
| [`core.py`](neatlogs/core.py) | Core tracking functionality | `LLMTracker`, `LLMSpan`, `LLMCallData` classes, context variables for framework tracking, background server communication |
| [`requirements.txt`](neatlogs/requirements.txt) | Dependencies | Minimal dependencies (requests library) |
| [`utils.py`](neatlogs/utils.py) | Utility functions | Session ID generation, cost estimation, statistics formatting |
| [`semconv.py`](neatlogs/semconv.py) | Semantic conventions | Standardized attribute names for LLM operations, provider mappings, message formatting utilities |
| [`token_counting.py`](neatlogs/token_counting.py) | Token usage utilities | Token extraction from responses, cost estimation for various models |
| [`stream_wrapper.py`](neatlogs/stream_wrapper.py) | Streaming support | `NeatlogsStreamWrapper` for instrumenting streaming responses |

### Instrumentation Directory

| File | Purpose | Key Contents |
|------|---------|-------------|
| [`instrumentation/__init__.py`](neatlogs/instrumentation/__init__.py) | Instrumentation package init | Empty package initializer |
| [`instrumentation/manager.py`](neatlogs/instrumentation/manager.py) | Patching coordination | Import hook setup, framework/provider detection, conflict resolution, patching lifecycle management |
| [`instrumentation/patchers.py`](neatlogs/instrumentation/patchers.py) | Provider-specific patching | `ProviderPatcher` class with methods for patching OpenAI, Anthropic, Google GenAI, LiteLLM, CrewAI |

### Event Handlers Directory

| File | Purpose | Key Contents |
|------|---------|-------------|
| [`event_handlers/__init__.py`](neatlogs/event_handlers/__init__.py) | Handler registry | Provider handler mappings, factory function for getting appropriate handlers |
| [`event_handlers/base.py`](neatlogs/event_handlers/base.py) | Abstract base handler | `BaseEventHandler` class defining interface for message extraction, response parsing, method wrapping |
| [`event_handlers/openai.py`](neatlogs/event_handlers/openai.py) | OpenAI handler | OpenAI-specific data extraction, streaming support, tool calls handling |
| [`event_handlers/anthropic.py`](neatlogs/event_handlers/anthropic.py) | Anthropic handler | Claude API response parsing, tool use extraction |
| [`event_handlers/google_genai.py`](neatlogs/event_handlers/google_genai.py) | Google GenAI handler | Gemini API integration, content block processing |
| [`event_handlers/litellm.py`](neatlogs/event_handlers/litellm.py) | LiteLLM handler | Universal LLM API wrapper support |
| [`event_handlers/azure.py`](neatlogs/event_handlers/azure.py) | Azure OpenAI handler | Azure-specific authentication and endpoint handling |
| [`event_handlers/langgraph.py`](neatlogs/event_handlers/langgraph.py) | LangGraph handler | LangGraph workflow instrumentation |

### Integration Directory

| File | Purpose | Key Contents |
|------|---------|-------------|
| [`integration/__init__.py`](neatlogs/integration/__init__.py) | Integration package init | Empty package initializer |
| [`integration/callbacks/__init__.py`](neatlogs/integration/callbacks/__init__.py) | Callbacks package init | Empty package initializer |
| [`integration/callbacks/langchain/__init__.py`](neatlogs/integration/callbacks/langchain/__init__.py) | LangChain callbacks init | Empty package initializer |
| [`integration/callbacks/langchain/callback.py`](neatlogs/integration/callbacks/langchain/callback.py) | LangChain callback handlers | Synchronous and asynchronous callback handlers for LangChain workflows |

## Component Interconnections

### Data Flow Architecture

1. **Initialization**: `neatlogs.init()` creates global tracker and sets up import monitoring
2. **Import Detection**: Import hook in `manager.py` detects supported frameworks/providers
3. **Selective Patching**: `ProviderPatcher` wraps methods based on detected frameworks to avoid conflicts
4. **Call Interception**: Wrapped methods create spans and call event handlers
5. **Data Extraction**: Event handlers extract provider-specific data from requests/responses
6. **Span Completion**: Tracker completes spans with timing, tokens, costs, and errors
7. **Data Transmission**: Background threads send telemetry to Neatlogs servers

### Key Design Decisions

- **Import-Time Patching**: Enables zero-code instrumentation but requires careful conflict management
- **Framework-First Detection**: Prioritizes framework-level tracking to prevent provider-level duplication
- **Context Variables**: Thread-local storage for framework context prevents race conditions
- **Event Handler Pattern**: Allows provider-specific logic while maintaining consistent interfaces
- **Background Processing**: Non-blocking server communication ensures minimal performance impact
- **Graceful Degradation**: Comprehensive error handling ensures tracking failures don't break applications

## Extension Guide

### Adding New LLM Providers

To add support for a new LLM provider:

1. **Create Event Handler**:
   ```python
   # neatlogs/event_handlers/new_provider.py
   from .base import BaseEventHandler

   class NewProviderHandler(BaseEventHandler):
       def extract_messages(self, *args, **kwargs):
           # Extract messages from provider-specific format
           return kwargs.get('messages', [])

       def extract_response_data(self, response):
           # Extract tokens, completion, etc. from response
           return {
               'completion': response.get('text', ''),
               'prompt_tokens': response.get('prompt_tokens'),
               'completion_tokens': response.get('completion_tokens'),
               'total_tokens': response.get('total_tokens')
           }
   ```

2. **Add Patcher Method**:
   ```python
   # neatlogs/instrumentation/patchers.py
   def patch_new_provider(self):
       try:
           import new_provider
           # Patch the client method
           original_method = new_provider.Client.generate
           handler = get_handler_for_provider("new_provider", self.tracker)
           new_provider.Client.generate = handler.wrap_method(original_method, "new_provider")
           return True
       except ImportError:
           return False
   ```

3. **Register Provider**:
   ```python
   # neatlogs/instrumentation/manager.py
   SUPPORTED_PROVIDERS["new_provider"] = {"patcher": "patch_new_provider"}
   ```

4. **Update Handler Registry**:
   ```python
   # neatlogs/event_handlers/__init__.py
   from .new_provider import NewProviderHandler
   PROVIDER_HANDLERS["new_provider"] = NewProviderHandler
   ```

### Adding New Frameworks

To add support for a new agentic framework:

1. **Register Framework**:
   ```python
   # neatlogs/instrumentation/manager.py
   SUPPORTED_FRAMEWORKS["new_framework"] = {
       "patcher": "patch_new_framework",
       "instrumentation_type": "wrapper"
   }
   FRAMEWORK_PROVIDER_MAPPING["new_framework"] = {"openai", "anthropic"}
   ```

2. **Add Patcher Method**:
   ```python
   # neatlogs/instrumentation/patchers.py
   def patch_new_framework(self):
       try:
           import new_framework
           # Set framework context before execution
           original_run = new_framework.Agent.run
           @wraps(original_run)
           def tracked_run(*args, **kwargs):
               set_current_framework("new_framework")
               try:
                   return original_run(*args, **kwargs)
               finally:
                   clear_current_framework()
           new_framework.Agent.run = tracked_run
           return True
       except ImportError:
           return False
   ```

### Creating Custom Callback Handlers

For frameworks using callback patterns:

1. **Inherit from BaseCallbackHandler**:
   ```python
   from langchain_core.callbacks.base import BaseCallbackHandler

   class CustomCallbackHandler(BaseCallbackHandler):
       def __init__(self, tracker):
           self.tracker = tracker
           self.active_spans = {}

       def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
           span = self.tracker.start_llm_span(model="custom", provider="custom_framework")
           span.messages = prompts
           self.active_spans[run_id] = span

       def on_llm_end(self, response, *, run_id, **kwargs):
           if run_id in self.active_spans:
               span = self.active_spans.pop(run_id)
               # Extract response data
               span.completion = response.generations[0][0].text
               self.tracker.end_llm_span(span)
   ```

2. **Handle Context Properly**:
   ```python
   def on_llm_start(self, ...):
       suppress_patching()  # Prevent double-tracking
       # ... span creation logic

   def on_llm_end(self, ...):
       # ... span completion logic
       release_patching()  # Restore normal patching
   ```

## Developer Checklist

### For New LLM Providers

- [ ] Event handler implements `BaseEventHandler` interface
- [ ] Message extraction handles provider-specific formats
- [ ] Response data extraction includes tokens, completion, and metadata
- [ ] Streaming support implemented if provider supports it
- [ ] Patcher method added to `ProviderPatcher` class
- [ ] Provider registered in `SUPPORTED_PROVIDERS` dictionary
- [ ] Handler registered in `PROVIDER_HANDLERS` dictionary
- [ ] Error handling prevents crashes on malformed responses
- [ ] Thread safety verified for concurrent usage
- [ ] Documentation updated with provider-specific details
- [ ] Unit tests added for handler functionality
- [ ] Integration tests verify end-to-end tracking

### For New Frameworks

- [ ] Framework registered in `SUPPORTED_FRAMEWORKS`
- [ ] Provider conflicts handled in `FRAMEWORK_PROVIDER_MAPPING`
- [ ] Patcher method implemented in `ProviderPatcher`
- [ ] Framework context properly set/cleared
- [ ] Callback handler created if framework uses callbacks
- [ ] Suppress patching context used to prevent conflicts
- [ ] Error handling for import failures
- [ ] Documentation includes framework-specific setup
- [ ] Tests verify framework detection and patching
- [ ] Integration tests confirm no duplicate tracking

### General Requirements

- [ ] Code follows existing patterns and conventions
- [ ] Comprehensive error handling and logging
- [ ] Thread-safe implementation verified
- [ ] Memory leaks prevented (proper cleanup)
- [ ] Performance impact minimized
- [ ] Backward compatibility maintained
- [ ] Documentation updated in all relevant places
- [ ] Changelog entry added
- [ ] Version bump considered if breaking changes
- [ ] CI/CD pipeline passes all tests
- [ ] Manual testing in realistic scenarios completed

## Usage Examples

### Basic Setup
```python
import neatlogs

# Initialize tracking
tracker = neatlogs.init(
    api_key="your_api_key",
    tags=["production", "chatbot"]
)

# All LLM calls are now automatically tracked
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### LangChain Integration
```python
import neatlogs
from neatlogs.integration.callbacks.langchain import NeatlogsLangchainCallbackHandler

# Initialize Neatlogs
tracker = neatlogs.init(api_key="your_api_key")

# Use callback handler
handler = NeatlogsLangchainCallbackHandler(tracker)
chain = LLMChain(llm=llm, callbacks=[handler])
result = chain.run("What is the capital of France?")
```

### Custom Framework Support
```python
# For frameworks not auto-detected
from neatlogs.core import set_current_framework, clear_current_framework

set_current_framework("my_custom_framework")
try:
    # Your LLM operations here
    response = llm.generate(prompt)
finally:
    clear_current_framework()
```

This comprehensive architecture ensures Neatlogs provides robust, extensible LLM tracking capabilities while maintaining simplicity for end users and flexibility for developers extending the system.