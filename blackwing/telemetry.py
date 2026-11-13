"""Optional OTLP tracing with metadata-only spans."""
from functools import lru_cache, wraps
import os

from opentelemetry import trace


@lru_cache(maxsize=1)
def tracer():
    endpoint = os.getenv("BLACKWING_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "blackwing-review"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        return provider.get_tracer(__name__)
    return trace.get_tracer(__name__)


def traced(name):
    def decorate(function):
        @wraps(function)
        def instrumented(*args, **kwargs):
            with tracer().start_as_current_span(name, record_exception=False,
                                                set_status_on_exception=False) as span:
                try:
                    return function(*args, **kwargs)
                except Exception:
                    span.set_status(trace.StatusCode.ERROR)
                    raise
        return instrumented
    return decorate
