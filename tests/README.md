# Neatlogs Tests

This directory contains tests for the Neatlogs library.

## Running Tests

### Install test dependencies

```bash
uv sync --group dev
```

### Run all tests

```bash
uv run pytest
```

### Run specific test file

```bash
uv run pytest tests/test_openai_instrumentation.py
```

### Run with coverage

```bash
uv run pytest --cov=neatlogs --cov-report=html
```

### Run specific test

```bash
uv run pytest tests/test_openai_instrumentation.py::TestOpenAIInstrumentation::test_openai_chat_completion_creates_span
```

## Test Structure

- `conftest.py` - Shared pytest fixtures and configuration
- `test_openai_instrumentation.py` - Tests for OpenAI SDK instrumentation

## Writing Tests

Tests use mocking to avoid making real API calls:
- OpenAI client is mocked using `unittest.mock`
- Spans are captured using OpenTelemetry's `InMemorySpanExporter`
- No real API keys or network calls are required
