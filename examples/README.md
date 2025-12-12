# Neatlogs Examples

This folder contains examples demonstrating how to use Neatlogs to track LLM API calls with different providers.

## Available Examples

- **openai_sdk.py** - OpenAI SDK integration example
- **azure_openai.py** - Azure OpenAI integration example
- **azure_openai_agents.py** - Azure OpenAI Agents integration example
- **google_genai.py** - Google Generative AI integration example
- **anthropic.py** - Anthropic Claude integration example
- **crewai.py** - CrewAI integration example

## Running the Examples

### 1. Start the Dev Server

First, start the local development server to receive traces:

```bash
cd tools
uvicorn dev_server:app --reload --port 8000
```

The server will listen for traces at `http://localhost:8000/api/data/v2`

### 2. Set Your API Keys

Set the API key for the provider you want to test:

```bash
# For OpenAI examples
export OPENAI_API_KEY='your-openai-key-here'

# For Azure OpenAI examples
export AZURE_OPENAI_API_KEY='your-azure-openai-key-here'
export AZURE_OPENAI_ENDPOINT='your-azure-endpoint-here'
export AZURE_OPENAI_API_VERSION='2024-08-01-preview'
export AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment-name-here'

# For Google GenAI examples
export GOOGLE_API_KEY='your-google-api-key-here'

# For Anthropic examples
export ANTHROPIC_API_KEY='your-anthropic-key-here'
```

### 3. Install Provider Libraries

Install the library for the provider you want to test:

```bash
# For OpenAI
uv add openai
# or: pip install openai

# For Azure OpenAI (same as OpenAI)
uv add openai
# or: pip install openai

# For Google GenAI
uv add google-generativeai
# or: pip install google-generativeai

# For Anthropic
uv add anthropic
# or: pip install anthropic

# For CrewAI
uv add crewai
# or: pip install crewai
```

### 4. Run the Example

In a separate terminal, run the specific example:

```bash
# OpenAI example
python examples/usage_openai.py

# Anthropic example
python examples/usage_anthropic.py

# Overview
python examples/usage.py
```

### 5. Check the Traces

When you run the example, you should see trace data appear in the dev_server terminal output.

## Important Notes

- **Neatlogs tracks LLM API calls only**: Standard Python `logging` calls are NOT tracked. You must make actual LLM API calls (OpenAI, Anthropic, LangChain, etc.) to generate traces.

- **API URL must include the full path**: When setting `NEATLOGS_API_URL`, always include `/api/data/v2`:
  ```python
  os.environ["NEATLOGS_API_URL"] = "http://localhost:8000/api/data/v2"
  ```

- **Supported Providers**: OpenAI, Anthropic, LangChain, Google GenAI, CrewAI, Groq, LiteLLM

## Example Structure

Each provider-specific example follows the same pattern:

1. Set the `NEATLOGS_API_URL` environment variable
2. Initialize Neatlogs with `init()`
3. Make LLM API calls using the provider's SDK
4. Traces are automatically captured and sent to the dev server

## Troubleshooting

If traces aren't appearing:

1. **Verify the dev_server is running** on port 8000
   ```bash
   # You should see: INFO: Uvicorn running on http://127.0.0.1:8000
   ```

2. **Check that `NEATLOGS_API_URL` includes `/api/data/v2`**
   - Correct: `http://localhost:8000/api/data/v2`
   - Wrong: `http://localhost:8000`

3. **Ensure you're making actual LLM API calls** (not just logging)
   - The instrumentation only captures LLM provider API calls
   - Standard `logging.info()` calls will NOT generate traces

4. **Verify your API keys are set correctly**
   ```bash
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

5. **Enable debug mode** to see detailed logs:
   ```python
   init(api_key="test-key", debug=True)
   ```

6. **Check for error messages** in both the example output and dev_server output
