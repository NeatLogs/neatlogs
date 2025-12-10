"""
Neatlogs OpenAI Agents SDK Example
==================================
This example demonstrates a simple flow using the OpenAI Agents SDK.
"""

import os
from neatlogs import init

# Initialize neatlogs
init(api_key="test-key", debug=True, enable_otel=True)

print("=" * 60)
print("Neatlogs OpenAI Agents SDK Example")
print("=" * 60)

try:
    from agents import Agent, Runner

    # Create a simple agent
    # This matches the "Hello World" example from the docs
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    print("\nRunning agent...")

    # Run the agent synchronously
    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")

    print(f"\nResponse: {result.final_output}")
    print("\n✓ Success!")
    print("=" * 60)

    import time

    time.sleep(1)

except ImportError:
    print("\n⚠ Error: OpenAI Agents library not installed")
    print("  Install with: uv add openai-agents")
    print("  or: pip install openai-agents")
    print("=" * 60)

except Exception as e:
    print(f"\n⚠ Error executing agent: {e}")
    print("  Make sure OPENAI_API_KEY is set in your environment.")
    print("=" * 60)
