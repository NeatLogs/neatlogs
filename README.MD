# Neatlogs

A comprehensive LLM tracking system that automatically captures and logs all LLM API calls with detailed metrics.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/neatlogs.svg)](https://badge.fury.io/py/neatlogs)

## Features

- 🚀 **Automatic LLM Call Tracking**: Seamlessly tracks all LLM API calls without code changes
- 📊 **Comprehensive Metrics**: Token usage, costs, response times, and more
- 🔌 **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, and LiteLLM
- 🔗 **LangChain Integration**: Seamless tracking for LangChain chains, agents, and tools
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

# add tags
neatlogs.add_tags(["neatlogs"])
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

## Framework

Neatlogs provides comprehensive support for various AI frameworks and models:

- [LangChain Integration](#langchain-integration)
- [CrewAI Integration](#crewai-integration)
- [LangGraph Integration](#langgraph-integration)

### LangChain Integration
Neatlogs provides comprehensive tracking for all LangChain components and workflows:

- **LLM & Chat Models**: Track all LLM calls, token usage, costs, and response times
- **Chains**: Monitor chain execution, inputs, outputs, and performance metrics
- **Agents**: Capture agent actions, tool calls, decision-making processes, and reasoning
- **Tools**: Record tool usage, inputs, outputs, and execution times
- **RAG Systems**: Track retrieval-augmented generation workflows including vector searches and document retrieval
- **Async Workflows**: Full support for asynchronous LangChain pipelines and concurrent operations
- **Error Handling**: Capture and log errors across all LangChain components
- **Model Detection**: Automatic identification of underlying LLM models and providers.

#### LangChain Callback Handler

Neatlogs provides a dedicated callback handler for LangChain to enable detailed tracking of your LangChain applications without modifying your existing code.

#### Usage

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import neatlogs

# Get the callback handler
handler = neatlogs.get_langchain_callback_handler(api_key="your-api-key")

# Use it with your LangChain components
llm = OpenAI()
chain = LLMChain(llm=llm, callbacks=[handler])

# Your chain calls will now be tracked automatically
result = chain.run("Hello world")
```

#### Features

- **LLM Tracking**: Captures all LLM calls with token usage, costs, and response times
- **Chain Monitoring**: Tracks chain executions, inputs, and outputs
- **Tool Call Tracking**: Monitors tool usage and performance
- **Agent Monitoring**: Records agent actions and decision processes
- **Automatic Detection**: Automatically detects model types and providers
- **Async Support**: Full support for both synchronous and asynchronous workflows

#### Asynchronous Usage

For asynchronous LangChain workflows:

```python
from neatlogs.integration.callbacks.langchain.callback import AsyncNeatlogsLangchainCallbackHandler

# Use the async handler for async workflows
async_handler = AsyncNeatlogsLangchainCallbackHandler(api_key="your-api-key")

# Use with async chains
result = await async_chain.arun(..., callbacks=[async_handler])
```

### CrewAI Integration
CrewAI is a framework for orchestrating role-playing AI agents. Neatlogs provides seamless integration with CrewAI through automatic instrumentation:

- **Agent Tracking**: Monitor all agent activities, tasks, and interactions
- **Crew Orchestration**: Track crew-level operations and agent coordination
- **Task Monitoring**: Capture task execution, delegation, and completion
- **Automatic Setup**: No code changes required - just initialize with `neatlogs.init()`

```python
import neatlogs
from crewai import Agent, Task, Crew

# Initialize Neatlogs (that's all you need!)
neatlogs.init(api_key="your-api-key")

# Your CrewAI code works normally and gets tracked automatically
agent = Agent(role="Researcher", goal="Research AI trends")
task = Task(description="Research latest AI developments")
crew = Crew(agents=[agent], tasks=[task])

result = crew.kickoff()
```

### LangGraph Integration
LangGraph is a library for building stateful, multi-actor applications with LLMs, using graphs to define the flow of execution.

Neatlogs provides seamless integration with LangGraph through automatic instrumentation:

- **Graph Execution Tracking**: Monitor graph execution, node transitions, and state changes
- **Node Monitoring**: Track individual node executions, inputs, outputs, and performance
- **Edge Tracking**: Capture edge traversals and conditional logic
- **Automatic Setup**: No code changes required - just initialize with `neatlogs.init()`

```python
import neatlogs
from langgraph import StateGraph

# Initialize Neatlogs (that's all you need!)
neatlogs.init(api_key="your-api-key")

# Your LangGraph code works normally and gets tracked automatically
graph = StateGraph(...)
# ... define your graph

result = graph.invoke(...)
```

### Configuration Options

```python
neatlogs.init(
    api_key="your-api-key",
    tags=["tag1", "tag2"],
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
