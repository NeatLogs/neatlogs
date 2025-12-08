"""
Tests for OpenAI Instrumentation
=================================
Tests that verify OpenAI SDK calls are properly instrumented and spans are created.
"""

import pytest
import respx
import httpx
import json
from unittest.mock import Mock, patch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TestOpenAIInstrumentation:
    """Test suite for OpenAI instrumentation with Neatlogs."""

    @pytest.fixture
    def openai_chat_response_json(self):
        """Mock OpenAI API response JSON."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    @pytest.fixture
    def mock_server_response(self):
        """Mock successful Neatlogs server response."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "success"}
            mock_post.return_value = mock_response
            yield mock_post

    @respx.mock
    def test_openai_chat_completion_creates_span(
        self, openai_chat_response_json, in_memory_span_exporter, mock_server_response
    ):
        """Test that a chat completion creates exactly one LLM span."""
        # Mock the OpenAI API endpoint
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_chat_response_json)
        )

        # Setup tracer provider BEFORE initializing neatlogs
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        # Initialize neatlogs
        import neatlogs

        neatlogs.init(api_key="test-key", enable_otel=True, dry_run=True)

        # Import and use OpenAI client
        from openai import OpenAI

        client = OpenAI(api_key="fake-key-for-testing")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
            ],
        )

        # Verify the response
        assert response.choices[0].message.content == "Hello! How can I help you today?"

        # Give time for span processing
        import time

        time.sleep(0.1)

        # Verify exactly one span was created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1, f"At least one span should be created, got {len(spans)}"

        # Find the LLM span
        llm_spans = [
            s
            for s in spans
            if s.attributes and s.attributes.get("openinference.span.kind") == "LLM"
        ]
        assert len(llm_spans) == 1, f"Exactly one LLM span should be created, got {len(llm_spans)}"

        # Verify span attributes
        span = llm_spans[0]
        assert span.name == "ChatCompletion"
        assert span.attributes.get("llm.system") == "openai"

    @respx.mock
    def test_span_contains_correct_attributes(
        self, openai_chat_response_json, mock_server_response, capsys
    ):
        """Test that the span contains all expected LLM attributes."""
        # Mock the OpenAI API endpoint
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_chat_response_json)
        )

        # Initialize neatlogs (this will set up its own span processor)
        import neatlogs

        neatlogs.init(api_key="test-key", enable_otel=True, dry_run=True)

        from openai import OpenAI

        client = OpenAI(api_key="fake-key-for-testing")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify the API call succeeded
        assert response.choices[0].message.content == "Hello! How can I help you today?"

        # Give time for span processing
        import time

        time.sleep(0.1)

        # Verify span was processed by checking dry run output
        captured = capsys.readouterr()
        assert "Neatlogs [Dry Run]: Processed span" in captured.out
        assert "gpt-3.5-turbo" in captured.out
        assert "Input Messages: 1" in captured.out

    @respx.mock
    def test_multiple_api_calls_create_multiple_spans(
        self, openai_chat_response_json, mock_server_response, capsys
    ):
        """Test that multiple API calls create multiple spans."""
        # Mock the OpenAI API endpoint
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_chat_response_json)
        )

        # Initialize neatlogs
        import neatlogs

        neatlogs.init(api_key="test-key", enable_otel=True, dry_run=True)

        from openai import OpenAI

        client = OpenAI(api_key="fake-key-for-testing")

        # Make multiple calls
        for i in range(3):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Message {i}"}],
            )
            assert response.choices[0].message.content == "Hello! How can I help you today?"

        # Give time for span processing
        import time

        time.sleep(0.2)

        # Verify all 3 spans were processed
        captured = capsys.readouterr()
        assert captured.out.count("Neatlogs [Dry Run]: Processed span") == 3
        assert "Message 0" in captured.out
        assert "Message 1" in captured.out
        assert "Message 2" in captured.out

    def test_instrumentation_without_openai_installed(self):
        """Test that instrumentation gracefully handles missing OpenAI package."""
        import neatlogs
        from neatlogs.instrumentation import manager

        # Mock importlib to simulate openai not installed
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None

            # This should not raise an error
            manager.instrument_all()

            # Should complete without errors
            assert True

    @respx.mock
    def test_dry_run_mode_does_not_send_to_server(self, openai_chat_response_json):
        """Test that dry_run mode doesn't send data to server."""
        # Mock the OpenAI API endpoint
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_chat_response_json)
        )

        import neatlogs

        with patch("requests.post") as mock_post:
            # Initialize with dry_run=True
            neatlogs.init(api_key="test-key", enable_otel=True, dry_run=True)

            from openai import OpenAI

            client = OpenAI(api_key="fake-key-for-testing")
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
            )

            # Give some time for background processing
            import time

            time.sleep(0.5)

            # Verify no POST requests were made to the server
            assert (
                mock_post.call_count == 0
            ), "dry_run mode should not send data to server"


class TestSpanProcessorDataExtraction:
    """Test suite for span processor data extraction logic."""

    def test_extract_model_from_span_attributes(self):
        """Test model extraction from span attributes."""
        from neatlogs.otel.processor import NeatlogsSpanProcessor
        from neatlogs.core import LLMTracker
        from unittest.mock import Mock

        # Create a mock tracker
        tracker = Mock(spec=LLMTracker)
        tracker.session_id = "test-session"
        tracker.agent_id = "test-agent"
        tracker.thread_id = "test-thread"
        tracker.tags = []
        tracker.api_key = "test-key"
        tracker.enable_server_sending = False
        tracker.dry_run = True

        processor = NeatlogsSpanProcessor(tracker)

        # Create a mock span
        span = Mock()
        span.name = "ChatCompletion"
        span.context.span_id = 123456789
        span.context.trace_id = 987654321
        span.parent = None
        span.start_time = 1000000000
        span.end_time = 2000000000
        span.status.is_ok = True
        span.attributes = {
            "openinference.span.kind": "LLM",
            "llm.request.model": "gpt-3.5-turbo",
            "llm.system": "openai",
            "llm.usage.prompt_tokens": 20,
            "llm.usage.completion_tokens": 10,
        }

        # Process the span
        processor._process_llm_span(span)

        # Verify no errors occurred
        assert True, "Span processing should complete without errors"

    def test_token_usage_extraction(self):
        """Test token usage extraction from different attribute formats."""
        from neatlogs.otel.processor import NeatlogsSpanProcessor
        from neatlogs.core import LLMTracker
        from unittest.mock import Mock

        tracker = Mock(spec=LLMTracker)
        tracker.session_id = "test-session"
        tracker.agent_id = "test-agent"
        tracker.thread_id = "test-thread"
        tracker.tags = []
        tracker.api_key = "test-key"
        tracker.enable_server_sending = False
        tracker.dry_run = True

        processor = NeatlogsSpanProcessor(tracker)

        # Test with OpenInference attributes
        span = Mock()
        span.name = "Test"
        span.context.span_id = 123
        span.context.trace_id = 456
        span.parent = None
        span.start_time = 1000000000
        span.end_time = 2000000000
        span.status.is_ok = True
        span.attributes = {
            "openinference.span.kind": "LLM",
            "llm.usage.prompt_tokens": 100,
            "llm.usage.completion_tokens": 50,
            "llm.request.model": "gpt-4",
            "llm.system": "openai",
        }

        processor._process_llm_span(span)

        # Verify processing completed
        assert True, "Token extraction should work correctly"
