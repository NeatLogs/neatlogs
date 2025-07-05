# Neatlogs Python SDK

[![PyPI version](https://badge.fury.io/py/neatlogs.svg)](https://badge.fury.io/py/neatlogs)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Python SDK for [Neatlogs](https://neatlogs.com) - a collaborative debugging platform for AI agents and LLM applications.

## What is Neatlogs?

Neatlogs helps your team debug AI agents faster and more effectively by bringing developers and domain experts together for collaborative debugging. Unlike traditional observability tools, Neatlogs is designed from the ground up for team collaboration on AI agent debugging.

### Key Features

- **🔍 Agent Trace Debugging**: Crystal-clear visualization of agent thoughts, tool calls, and decisions
- **💬 Collaborative Comments**: Team members can comment directly on specific parts of agent traces
- **📊 Smart Evaluations**: Quick feedback system with actionable insights
- **📋 AI-Powered Task Management**: Convert discussions into trackable tasks automatically
- **🔧 Easy Integration**: Just 2 lines of code to get started

## Why Choose Neatlogs?

### The Problem with AI Agent Debugging

Unlike regular code where errors scream at you, AI agents are mysterious. When your agent gives puzzling results, you need to:

1. **Analyze the output** - What did it actually produce?
2. **Examine the trace** - Step-by-step log of thoughts and actions
3. **Find the root cause** - Usually issues with prompts or tools

### How Neatlogs Solves This

- **❌ No More Vague Feedback**: Skip the spreadsheet chaos - collaborate directly on agent traces
- **✅ Actionable Fixes**: Turn evaluations and discussions into clear, actionable tasks
- **🎯 Everything in One Place**: No more hunting for context across Slack, email, and docs
- **🚀 2x Faster Development**: Teams often ship reliable AI agents twice as fast

### Neatlogs vs. Traditional Observability

| Traditional Tools | Neatlogs |
|-------------------|----------|
| Developer-only focus | Built for team collaboration |
| Technical UI only | Non-technical friendly |
| Show what happened | Help understand why + how to fix |
| Scattered feedback | Centralized discussions |

## Quick Start

### Installation

```bash
pip install neatlogs
```

### Basic Usage

```python
import neatlogs

# Initialize with your Project API Key
neatlogs.init("your-project-api-key-here")

# Your existing AI agent code works unchanged!
# Neatlogs automatically captures LLM calls
```

### Get Your Project API Key

1. Visit [app.neatlogs.com](https://app.neatlogs.com)
2. Create a new project
3. Copy your Project API Key
4. Use it in the `neatlogs.init()` call

## Supported Frameworks

### LiteLLM (Primary Support)

Neatlogs seamlessly integrates with [LiteLLM](https://github.com/BerriAI/litellm) v1.3.1+:

```python
import neatlogs
import litellm

# Initialize Neatlogs
neatlogs.init("your-project-api-key")

# Use LiteLLM normally - Neatlogs captures everything automatically
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)
```

### CrewAI Integration

Neatlogs offers seamless integration with CrewAI for multi-agent applications:

```python
import neatlogs
from crewai import Agent, Task, Crew

# Initialize Neatlogs before creating your crew
neatlogs.init("your-project-api-key")

# Your CrewAI code works unchanged
agent = Agent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="You are an expert data analyst..."
)

# All agent interactions are automatically traced
```

## Advanced Features

### Adding Tags

Organize your traces with custom tags:

```python
import neatlogs

# Initialize tracker
tracker = neatlogs.init("your-project-api-key")

# Add tags to categorize traces
tracker.add_tags(["production", "customer-support", "v2.1"])

# All subsequent LLM calls will be tagged
```

### Stop Tracking

Temporarily disable tracking when needed:

```python
# Stop instrumenting LLM calls
tracker.stop_instrumenting()

# Your LLM calls won't be tracked until you reinitialize
```

## What Gets Tracked?

Neatlogs automatically captures:

- **💭 Agent Thoughts**: The reasoning process before actions
- **🛠️ Tool Calls**: API calls, function executions, and their results
- **🔄 Multi-Agent Interactions**: Task delegations and handoffs
- **📝 Prompts & Responses**: Complete conversation history
- **⚠️ Errors**: Detailed error information and stack traces
- **📊 Usage Statistics**: Token counts and performance metrics

## Platform Features

Once your traces are captured, use the Neatlogs platform to:

### 🔍 Trace Analysis
- **Clean UI**: No more endless scrolling through logs
- **Error Highlighting**: Automatically spot issues
- **AI Summaries**: Get quick overviews of complex traces
- **Step Navigation**: Jump between agent actions effortlessly

### 💬 Team Collaboration
- **Pinpoint Comments**: Comment on specific trace steps
- **Context Preservation**: Full conversation history in one place
- **Real-time Notifications**: Stay updated on team feedback
- **Domain Expert Friendly**: Non-technical team members can contribute

### 📊 Smart Evaluations
- **Quick Feedback**: Mark responses as Correct/Incorrect/Mixed
- **Bulk Operations**: Evaluate multiple traces efficiently
- **Filter & Search**: Find specific traces instantly
- **Export Options**: Download evaluation data

### 📋 Task Management
- **AI-Generated Tasks**: Convert comment threads to actionable items
- **Kanban Board**: Visual task tracking
- **Smart Suggestions**: AI recommends solutions based on context
- **Team Assignment**: Distribute work effectively

## Requirements

- Python 3.8+
- LiteLLM 1.3.1+ (for LLM call tracking)
- Active internet connection (for trace upload)

## Links

- **Website**: [neatlogs.com](https://neatlogs.com)
- **Documentation**: [docs.neatlogs.com](https://docs.neatlogs.com)
- **Dashboard**: [app.neatlogs.com](https://app.neatlogs.com)
- **Support**: [GitHub Issues](https://github.com/NeatLogs/neatlogs/issues)

## FAQ

**Q: Do I need to change my existing code?**
A: No! Just add `neatlogs.init()` and your existing LLM calls are automatically tracked.

**Q: What about data privacy?**
A: Traces are sent securely to Neatlogs servers. Check our privacy policy for details.

**Q: Can I use this in production?**
A: Yes! Neatlogs is designed for production use with minimal performance impact.

**Q: What LLM providers are supported?**
A: Any provider supported by LiteLLM (OpenAI, Anthropic, Azure, local models, etc.)

**Q: How much does it cost?**
A: Visit [neatlogs.com](https://neatlogs.com) for current pricing information.

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://github.com/NeatLogs/neatlogs/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

Having issues? We're here to help!

- 📧 Email: hello@neatlogs.com
- 💬 Discord: [Join our community](https://discord.gg/neatlogs)
- 📖 Docs: [docs.neatlogs.com](https://docs.neatlogs.com)
- 🐛 Issues: [GitHub Issues](https://github.com/NeatLogs/neatlogs/issues)

---

**Ready to debug your AI agents like a pro?** 

Start with just 2 lines of code:

```python
import neatlogs
neatlogs.init("your-project-api-key")
```

[Get started at app.neatlogs.com →](https://app.neatlogs.com)
