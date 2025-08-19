"""
Core tracking functionality for NeatLogs Tracker
"""

import time
import json
from uuid import uuid4
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import logging
import requests
from dataclasses import asdict
import threading

# Thread-local context for agentic framework
_current_framework_ctx = threading.local()


def set_current_framework(framework: str):
    _current_framework_ctx.value = framework


def get_current_framework() -> str:
    return getattr(_current_framework_ctx, "value", None)


def clear_current_framework():
    if hasattr(_current_framework_ctx, "value"):
        del _current_framework_ctx.value


@dataclass
class LLMCallData:
    session_id: str
    agent_id: str
    thread_id: str
    span_id: str
    trace_id: str
    model: str
    provider: str
    framework: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    messages: List[Dict]
    completion: str
    timestamp: str
    start_time: float
    end_time: float
    duration: float
    tags: List[str]
    error_report: Optional[Dict] = None
    status: str = "SUCCESS"
    api_key: Optional[str] = None


class LLMSpan:
    def __init__(self, session_id, agent_id, thread_id, api_key, model=None, provider=None, framework=None, tags=None):
        self.span_id = str(uuid4())
        self.trace_id = thread_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.model = model
        self.provider = provider
        self.framework = framework
        self.tags = tags or []
        self.api_key = api_key
        self.messages = []
        self.completion = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0
        self.error_report = None
        self.status = "SUCCESS"
        self.start_time = None
        self.end_time = None

    def _get_parent_context(self, trace_obj):
        pass

    def start(self, parent_context=None):
        # Record start time for internal tracking
        self.start_time = time.time()

    def end(self, success=True, error=None):
        self.end_time = time.time()
        self.status = "SUCCESS" if success else "FAILURE"
        if error:
            self.error_report = {
                "error_type": type(error).__name__,
                "error_code": getattr(error, 'code', 'N/A'),
                "error_message": str(error),
                "stack_trace": traceback.format_exc()
            }

    def to_llm_call_data(self) -> LLMCallData:
        duration = (self.end_time - self.start_time) if self.end_time else 0
        return LLMCallData(
            session_id=self.session_id,
            agent_id=self.agent_id,
            thread_id=self.thread_id,
            span_id=self.span_id,
            trace_id=self.trace_id,
            model=self.model or "unknown",
            provider=self.provider or "unknown",
            framework=self.framework,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost=self.cost,
            messages=self.messages,
            completion=self.completion,
            timestamp=datetime.fromtimestamp(self.start_time).isoformat(),
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            tags=self.tags,
            error_report=self.error_report,
            status=self.status,
            api_key=self.api_key
        )


class LLMTracker:
    def __init__(self, api_key, session_id=None, agent_id=None, thread_id=None, tags=None, enable_server_sending=True):
        self.session_id = session_id or str(uuid4())
        self.agent_id = agent_id or "default-agent"
        self.thread_id = thread_id or str(uuid4())
        self.tags = tags or []
        self.api_key = api_key
        self.enable_server_sending = enable_server_sending
        self._threads = []

        # Initialize framework and provider as None (since framework_detector removed)
        self.framework = None
        self.provider = None

        self.setup_logging()
        self._lock = threading.Lock()
        self._active_spans = {}
        self._completed_calls = []
        self._patched_providers = set()

        logging.info(f"LLMTracker initialized - Session: {self.session_id}, "
                     f"Agent: {self.agent_id}, Thread: {self.thread_id}")
        logging.info(f"Framework: {self.framework}, Provider: {self.provider}")
        if self.tags:
            logging.info(f"Tags: {self.tags}")

        if self.api_key:
            logging.info(f"API Key: {self.api_key}")
        if self.enable_server_sending:
            logging.info(
                "Server sending enabled - trace data will be sent to NeatLogs server")
        logging.info("NeatLogs Tracker: Monitoring enabled.")

    def setup_logging(self):
        self.file_logger = logging.getLogger(f'llm_tracker_{self.session_id}')
        self.file_logger.setLevel(logging.INFO)
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)

        formatter = logging.Formatter('%(asctime)s - %(message)s')

    def start_llm_span(self, model=None, provider=None, framework=None) -> LLMSpan:
        # Defensive: always initialize framework to None
        _framework = framework
        logging.debug(
            f"LLMTracker: start_llm_span called with model={model}, provider={provider}, framework={_framework}")
        if _framework is None:
            _framework = get_current_framework()
            logging.debug(
                f"LLMTracker: got framework from thread-local context: {_framework}")
        # No fallback to framework_detector anymore

        span = LLMSpan(
            self.session_id,
            self.agent_id,
            self.thread_id,
            self.api_key,
            model,
            provider,
            _framework,
            self.tags
        )
        with self._lock:
            self._active_spans[span.span_id] = span
        logging.debug(
            f"LLMTracker: Starting span with framework={_framework}, model={model}, provider={provider}")
        span.start()
        return span

    def end_llm_span(self, span, success=True, error=None):
        span.end(success, error)
        with self._lock:
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]
            call_data = span.to_llm_call_data()
            self._completed_calls.append(call_data)
            self.log_llm_call(call_data)

    def log_llm_call(self, call_data: LLMCallData):
        log_entry = {
            "event_type": "LLM_CALL",
            "data": asdict(call_data)
        }
        self.file_logger.info(json.dumps(log_entry, indent=2))

        # Send data to server in background if enabled
        if self.enable_server_sending:
            self._send_data_to_server(call_data)

    def _send_data_to_server(self, call_data: LLMCallData):
        """Send trace data to NeatLogs server in background thread."""
        def send_in_background():
            try:
                url = "https://app.neatlogs.com/api/data/v2"
                headers = {"Content-Type": "application/json"}

                # Prepare trace data
                trace_data = {
                    "session_id": call_data.session_id,
                    "agent_id": call_data.agent_id,
                    "thread_id": call_data.thread_id,
                    "span_id": call_data.span_id,
                    "trace_id": call_data.trace_id,
                    "model": call_data.model,
                    "provider": call_data.provider,
                    "framework": call_data.framework,
                    "prompt_tokens": call_data.prompt_tokens,
                    "completion_tokens": call_data.completion_tokens,
                    "total_tokens": call_data.total_tokens,
                    "cost": call_data.cost,
                    "messages": call_data.messages,
                    "completion": call_data.completion,
                    "timestamp": call_data.timestamp,
                    "start_time": call_data.start_time,
                    "end_time": call_data.end_time,
                    "duration": call_data.duration,
                    "tags": call_data.tags,
                    "status": call_data.status,
                    "api_key": call_data.api_key
                }

                # Add error information if present
                if call_data.error_report:
                    trace_data["error"] = call_data.error_report

                # Prepare API payload
                api_data = {
                    "dataDump": json.dumps(trace_data),
                    "projectAPIKey": call_data.api_key or self.api_key,
                    "externalTraceId": call_data.trace_id,  # Use trace_id for grouping
                    "timestamp": datetime.now().timestamp()
                }

                # Send request
                response = requests.post(
                    # url, json=api_data, headers=headers, timeout=10)
                    url, json=api_data, headers=headers)
                if response.status_code in (200, 201):
                    logging.debug(
                        f"Successfully sent trace data to server: {call_data.trace_id}")
                else:
                    logging.warning(
                        f"Failed to send trace data to server. Status: {response.status_code}")

            except Exception as e:
                logging.error(f"Error sending trace data to server: {e}")

        # Start background thread
        thread = threading.Thread(target=send_in_background, daemon=False)
        thread.start()
        self._threads.append(thread)

    def get_session_stats(self) -> Dict:
        with self._lock:
            total_calls = len(self._completed_calls)
            successful_calls = sum(
                1 for call in self._completed_calls if call.status == "SUCCESS")
            failed_calls = total_calls - successful_calls
            total_cost = sum(call.cost for call in self._completed_calls)
            total_tokens = sum(
                call.total_tokens for call in self._completed_calls)
            return {
                "session_id": self.session_id,
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "active_spans": len(self._active_spans)
            }

    def add_tags(self, tags: List[str]):
        """Add tags to the tracker.

        Args:
            tags: List of tags to add
        """
        with self._lock:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
        logging.info(f"Added tags: {tags}")

    def shutdown(self):
        """Wait for all tracking threads to complete."""
        for thread in self._threads:
            thread.join()

    def send_existing_logs_to_server(self):
        """Send all existing completed calls to the server."""
        if not self.enable_server_sending:
            logging.warning(
                "Server sending is disabled. Enable it to send existing logs.")
            return

        with self._lock:
            for call_data in self._completed_calls:
                self._send_data_to_server(call_data)

        logging.info(
            f"Sent {len(self._completed_calls)} existing log entries to server")
