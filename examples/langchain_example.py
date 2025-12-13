"""
Neatlogs LangChain Example
==========================
This example demonstrates how to use Neatlogs with LangChain to track LLM calls.
"""

import neatlogs
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Neatlogs
neatlogs.init(api_key=os.getenv("NEATLOGS_API_KEY", "test-key"))

# Get the Neatlogs callback handler for LangChain
neatlogs_callback = neatlogs.get_langchain_callback_handler()

print("=" * 60)
print("Neatlogs LangChain Example")
print("=" * 60)

try:
    # Create a LangChain chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    print("\nInvoking LangChain chain with Neatlogs callback...")

    # Invoke the chain with the Neatlogs callback handler
    response = chain.invoke(
        {"input": "Say hello in 5 words or less"},
        config={"callbacks": [neatlogs_callback]}
    )

    print(f"\nResponse: {response}")
    print("\n[SUCCESS] LangChain call tracked successfully by Neatlogs.")
    print("\n" + "=" * 60)


except ImportError:
    print("\n⚠ Error: LangChain or OpenAI library not installed")
    print("  Install with: uv add langchain langchain_openai")
    print("  or: pip install langchain langchain_openai")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error making LangChain call: {e}")
    print("  Make sure OPENAI_API_KEY is set in your environment:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("=" * 60)
