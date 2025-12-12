"""
Neatlogs Google GenAI Example
=============================
This example demonstrates how to use Neatlogs with Google Generative AI API calls.
Traces will be written to a local file (neatlogs.jsonl).
"""

import neatlogs
import os
import time

# Initialize neatlogs to write traces to a local file.
# dry_run=True prevents data from being sent to the remote server.
neatlogs.init(
    api_key="test-key",
)

print("=" * 60)
print("Neatlogs Google GenAI Example")
print("=" * 60)

try:
    import google.generativeai as genai

    # Configure Google GenAI with API key from environment
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    print("\nMaking Google GenAI API calls...")

    # Create a model instance
    model = genai.GenerativeModel('gemini-2.5-flash')

    # First API call - Simple chat
    print("\n1. Making first API call...")
    response1 = model.generate_content(
        "Explain quantum computing in 3 sentences or less."
    )
    print(f"Response 1: {response1.text.strip()}")

    # Second API call - Code generation
    print("\n2. Making second API call...")
    response2 = model.generate_content(
        "Write a Python function to calculate fibonacci numbers recursively."
    )
    print(f"Response 2: {response2.text.strip()}")

    # Third API call - Creative writing
    print("\n3. Making third API call...")
    response3 = model.generate_content(
        "Write a haiku about artificial intelligence."
    )
    print(f"Response 3: {response3.text.strip()}")

    print("\n[SUCCESS] All Google GenAI calls completed successfully")

    print("\n" + "=" * 60)

    # Wait for background threads to complete sending data
    print("\nWaiting for data to be sent to server...")
    time.sleep(5)  # Give background thread time to complete HTTP request

except ImportError:
    print("\n[ERROR] Google GenAI library not installed")
    print("  Install with: uv add google-generativeai")
    print("  or: pip install google-generativeai")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error making Google GenAI call: {e}")
    print("  Make sure GEMINI_API_KEY is set in your environment:")
    print("  export GEMINI_API_KEY='your-api-key-here'")
    print("=" * 60)
