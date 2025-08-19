# NeatLogs

A comprehensive LLM tracking system that automatically captures and logs all LLM API calls with detailed metrics.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/neatlogs.svg)](https://badge.fury.io/py/neatlogs)

## Features

- 🚀 **Automatic LLM Call Tracking**: Seamlessly tracks all LLM API calls without code changes
- 📊 **Comprehensive Metrics**: Token usage, costs, response times, and more
- 🔌 **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, and LiteLLM
- 🧵 **Session Management**: Track conversations across multiple threads and agents
- 📝 **Structured Logging**: Detailed logs with OpenTelemetry support
- 🎯 **Easy Integration**: Simple one-line initialization
- 🔍 **Real-time Monitoring**: Live tracking and statistics

## Quick Start

### Installation

**Basic installation (no LLM libraries):**

```bash
pip install neatlogs
```

### Basic Usage

```python
import neatlogs

# Initialize tracking with your API key
neatlogs.init(
    api_key="your-api-key-here"
)

# Now all LLM calls are automatically tracked!
# Use any supported LLM library normally

# Get session statistics
stats = neatlogs.get_session_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Total cost: ${stats['total_cost']:.4f}")
```

## Supported Providers

- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Google Gemini** (Gemini models)
- **Azure OpenAI**
- **LiteLLM** (unified interface)
- **CrewAI**

### Configuration Options

```python
neatlogs.init(
    api_key="your-api-key",
    tags=["tag1", "tag2"],
    debug=False,       # Enable debug logging
)
```

## Session Statistics

Get comprehensive insights into your LLM usage:

```python
stats = neatlogs.get_session_stats()

# Available metrics:
# - total_calls: Number of LLM API calls
# - total_tokens_input: Total input tokens
# - total_tokens_output: Total output tokens
# - total_cost: Total cost in USD
# - average_response_time: Average response time
# - provider_breakdown: Usage by provider
# - model_breakdown: Usage by model
```
