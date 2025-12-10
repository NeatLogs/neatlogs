"""
Neatlogs OpenAI Example
========================
This example demonstrates how to use Neatlogs with OpenAI API calls.
Traces will be sent to the local dev server.
"""

from neatlogs import init

# Initialize neatlogs with debug mode and OpenTelemetry enabled
init(api_key="test-key", debug=True, enable_otel=True)

print("=" * 60)
print("Neatlogs OpenAI Example")
print("=" * 60)

try:
    from openai import OpenAI

    # Create OpenAI client (uses OPENAI_API_KEY from environment)
    client = OpenAI()

    print("\nMaking OpenAI API call...")

    # Make a simple chat completion request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in 5 words or less"},
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print(f"\nResponse: {response.choices[0].message.content}")
    print("\n✓ Success! Trace sent successfully")
    print("\n" + "=" * 60)

    import time

    time.sleep(1)

except ImportError:
    print("\n⚠ Error: OpenAI library not installed")
    print("  Install with: uv add openai")
    print("  or: pip install openai")
    print("=" * 60)

except Exception as e:
    print(f"\n⚠ Error making OpenAI call: {e}")
    print("  Make sure OPENAI_API_KEY is set in your environment:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("=" * 60)
