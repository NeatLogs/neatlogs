import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import httpx
import pytest
import requests
import urllib3
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.core.span_processor import NeatlogsSpanProcessor
from neatlogs.instrumentation.manager import InstrumentationManager


class _SnapshotExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(
            {"name": span.name, "attributes": dict(span.attributes or {})} for span in spans
        )
        return SpanExportResult.SUCCESS


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_http_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def _aiohttp_get(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            assert await response.text() == "ok"


def _request(library, url):
    if library == "requests":
        assert requests.get(url, timeout=2).text == "ok"
    elif library == "httpx":
        assert httpx.get(url, timeout=2).text == "ok"
    elif library == "urllib3":
        assert urllib3.PoolManager().request("GET", url, timeout=2).data == b"ok"
    else:
        asyncio.run(_aiohttp_get(url))


@pytest.mark.parametrize("library", ["requests", "httpx", "urllib3", "aiohttp"])
def test_explicit_http_instrumentation_emits_one_nested_client_span(library, local_http_url):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    manager = InstrumentationManager(provider, excluded_urls="dev-cloud.neatlogs.com")

    try:
        manager.instrument(libraries=[library])
        tracer = provider.get_tracer("neatlogs.http-test")
        with tracer.start_as_current_span("workflow") as root:
            _request(library, local_http_url)

        children = [
            span
            for span in exporter.get_finished_spans()
            if span.parent and span.parent.span_id == root.context.span_id
        ]
        assert len(children) == 1
        assert children[0].kind.name == "CLIENT"
    finally:
        manager.uninstrument_all()
        provider.shutdown()


@pytest.mark.parametrize("library", ["requests", "httpx", "urllib3", "aiohttp"])
def test_explicit_http_instrumentation_adds_safe_canonical_io_through_sdk_pipeline(
    library, local_http_url
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[library],
        tracer_provider=provider,
        workflow_name="http-io-regression",
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    secret_url = (
        local_http_url.replace("http://", "http://user:pass@")
        + "?api_key=secret&token=also-secret#fragment"
    )
    tracer = provider.get_tracer("app.http-io-regression")
    with tracer.start_as_current_span("workflow") as root:
        root.set_attribute("openinference.span.kind", "WORKFLOW")
        _request(library, secret_url)

    children = [
        span
        for span in exporter.get_finished_spans()
        if span.parent and span.parent.span_id == root.context.span_id
    ]

    assert len(children) == 1
    span = children[0]
    attrs = span.attributes
    assert attrs["neatlogs.span.kind"] == "http"
    assert json.loads(attrs["input.value"]) == {
        "method": "GET",
        "url": local_http_url,
    }
    assert json.loads(attrs["output.value"]) == {"status": 200}
    assert json.loads(attrs["neatlogs.http.input"]) == {
        "method": "GET",
        "url": local_http_url,
    }
    assert json.loads(attrs["neatlogs.http.output"]) == {"status": 200}

    serialized_attrs = json.dumps(dict(attrs), default=str)
    assert "api_key=secret" not in serialized_attrs
    assert "token=also-secret" not in serialized_attrs
    assert "user:pass" not in serialized_attrs
    assert "fragment" not in serialized_attrs
    assert "Cookie" not in serialized_attrs
    assert "Authorization" not in serialized_attrs


def test_sdk_pipeline_removes_stale_http_payloads_before_later_exporters():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        tracer_provider=provider,
        workflow_name="http-export-privacy",
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("app.http-export-privacy")
    with tracer.start_as_current_span("workflow"):
        with tracer.start_as_current_span(
            "GET",
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": "GET",
                "url.full": "https://user:pass@example.com/health?token=secret#fragment",
                "url.query": "token=secret",
                "http.request.header.authorization": "Bearer secret",
                "http.request.body": "secret-body",
                "input.value": "secret-input",
                "output.value": "secret-output",
            },
        ):
            pass

    attrs = next(span["attributes"] for span in exporter.spans if span["name"] == "GET")
    assert json.loads(attrs["input.value"]) == {
        "method": "GET",
        "url": "https://example.com/health",
    }
    assert "output.value" not in attrs
    serialized = json.dumps(attrs, default=str)
    for secret in (
        "user:pass",
        "token=secret",
        "Bearer secret",
        "secret-body",
        "secret-input",
        "secret-output",
        "fragment",
    ):
        assert secret not in serialized


def test_http_payloads_cannot_leak_through_workflow_root_backfill():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    processor = NeatlogsSpanProcessor(own_all_spans=True)
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("app.http-root-privacy")
    with tracer.start_as_current_span("workflow"):
        with tracer.start_as_current_span(
            "GET",
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": "GET",
                "url.full": "https://user:pass@example.com/health?token=secret#fragment",
                "input.value": "secret-input",
                "output.value": "secret-output",
            },
        ):
            pass

    root = next(span for span in exporter.spans if span["name"] == "workflow")
    assert json.loads(root["attributes"]["input.value"]) == {
        "method": "GET",
        "url": "https://example.com/health",
    }
    assert "output.value" not in root["attributes"]
    serialized = json.dumps(root, default=str)
    for secret in ("user:pass", "token=secret", "fragment", "secret-input", "secret-output"):
        assert secret not in serialized


def test_raw_http_debug_log_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEATLOGS_LOG_RAW_SPANS", "true")
    provider = TracerProvider()
    processor = NeatlogsSpanProcessor(own_all_spans=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("app.http-raw-log-privacy")
    with tracer.start_as_current_span(
        "GET",
        kind=SpanKind.CLIENT,
        attributes={
            "http.request.method": "GET",
            "url.full": "https://user:pass@example.com/health?token=secret#fragment",
            "http.request.header.authorization": "Bearer secret",
            "http.request.body": "secret-body",
            "input.value": "secret-input",
            "output.value": "secret-output",
        },
    ):
        pass

    raw_log = (tmp_path / "spans_raw_optimized.log").read_text(encoding="utf-8")
    assert "https://example.com/health" in raw_log
    for secret in (
        "user:pass",
        "token=secret",
        "fragment",
        "Bearer secret",
        "secret-body",
        "secret-input",
        "secret-output",
    ):
        assert secret not in raw_log

    provider.shutdown()


def test_empty_instrumentation_list_emits_no_client_span(local_http_url):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    manager = InstrumentationManager(provider)

    try:
        manager.instrument(libraries=[])
        with provider.get_tracer("neatlogs.http-test").start_as_current_span("workflow"):
            assert requests.get(local_http_url, timeout=2).text == "ok"

        assert [span.name for span in exporter.get_finished_spans()] == ["workflow"]
    finally:
        manager.uninstrument_all()
        provider.shutdown()
