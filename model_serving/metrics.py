"""Definitions for custom Prometheus metrics."""

from prometheus_client import Counter, Histogram

# --- Define Custom Prometheus Metrics ---
BATCH_PREDICTION_OUTCOMES = Counter(
    "batch_prediction_outcomes_total", # Metric name
    "Counts the outcomes (success/error) of individual predictions within a batch request.", # Description
    ["outcome"] # Label to distinguish success/error
)

FEATURE_SERVICE_LATENCY = Histogram(
    "feature_service_request_latency_seconds", # Metric name
    "Latency of requests to the Feature Service API.", # Description
    # Define buckets for latency histogram (e.g., 10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s)
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, float("inf")]
)

PREDICTED_YIELD_DISTRIBUTION = Histogram(
    "predicted_yield_distribution", # Metric name
    "Distribution of predicted yield values.", # Description
    # Define buckets appropriate for expected yield range
    # Example: 0-10, 10-20, ..., 90-100, >100
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float("inf")]
)
# ------------------------------------
