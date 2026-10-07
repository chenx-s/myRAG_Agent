"""Prometheus instrumentation for the FastAPI application."""

import re
from typing import Optional

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator, metrics


# RAG/LLM requests are often much slower than ordinary CRUD APIs.  These
# buckets preserve useful percentiles for both fast health checks and queries
# that can take several minutes.
REQUEST_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)


def setup_metrics(
    app: FastAPI,
    *,
    enabled: bool = True,
    endpoint: str = "/metrics",
) -> Optional[Instrumentator]:
    """Instrument *app* and expose Prometheus metrics on ``endpoint``.

    The endpoint itself is excluded so Prometheus scrapes do not inflate the
    application's QPS and latency figures.  Status codes are grouped (2xx,
    4xx, 5xx) to keep metric cardinality bounded.
    """
    if not enabled:
        return None

    if not endpoint.startswith("/"):
        raise ValueError("The Prometheus metrics endpoint must start with '/'.")

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[rf"^{re.escape(endpoint)}$"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )

    # Explicitly register only the core API metrics.  prometheus_client also
    # contributes the standard Python process/runtime metrics automatically.
    instrumentator.add(metrics.requests())
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=False,
            buckets=REQUEST_LATENCY_BUCKETS,
        )
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint=endpoint,
        include_in_schema=False,
        should_gzip=True,
    )
    return instrumentator
