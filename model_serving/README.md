# Model Serving - AgriYield API

This directory contains the code for the FastAPI application that serves the trained AgriYield prediction model.

## Overview

The service exposes a REST API to get yield predictions based on county and year. It loads the appropriate model version from an MLflow Tracking Server and **fetches the required features by calling an external Feature Service API.**

It includes:
*   Single and batch prediction endpoints.
*   Asynchronous request handling.
*   Structured JSON logging (`structlog`).
*   Prometheus metrics endpoint (`/metrics`).
*   Rate limiting on prediction endpoints.
*   Retry logic for Feature Service calls.
*   Enhanced health checks.
*   Custom exception handling.
*   Detailed OpenAPI documentation (`/docs`).

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:** Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).
2.  **Install Dependencies:** Navigate to this directory (`model_serving`) in your terminal and run:
    ```bash
    poetry install
    ```
    This will install all required dependencies, including development dependencies like `pytest`.

## Configuration

The application requires the following environment variables to be set:

*   `MLFLOW_TRACKING_URI`: The URI of the MLflow Tracking Server.
*   `FEATURE_SERVICE_URL`: The base URL of the Feature Service API.
*   `MLFLOW_MODEL_NAME`: (Optional) Defaults to `AgriYieldPredictor`.
*   `MLFLOW_MODEL_STAGE`: (Optional) Defaults to `Production`.
*   `API_PREDICT_LIMIT`: (Optional) Rate limit for `/predict`. Defaults to `10/minute`.
*   `API_PREDICT_BATCH_LIMIT`: (Optional) Rate limit for `/predict_batch`. Defaults to `5/minute`.
*   `FEATURE_SERVICE_TIMEOUT_SECONDS`: (Optional) Timeout for feature service calls. Defaults to `10.0`.
*   `FEATURE_SERVICE_RETRY_ATTEMPTS`: (Optional) Number of retries for feature service. Defaults to `3`.
*   `FEATURE_SERVICE_RETRY_MIN_WAIT_SECONDS`: (Optional) Min wait between retries. Defaults to `1`.
*   `FEATURE_SERVICE_RETRY_MAX_WAIT_SECONDS`: (Optional) Max wait between retries. Defaults to `5`.
*   `UVICORN_WORKERS`: (Optional) Number of Uvicorn worker processes. Defaults to `2`. **Set based on available CPU cores (e.g., `2 * num_cores + 1`) for production.**

You can set these variables directly in your environment or use a `.env` file.

## Running Locally

To run the API server locally for development:

1.  Ensure dependencies are installed (`poetry install`).
2.  Set the required environment variables (e.g., `export MLFLOW_TRACKING_URI=...`, `export FEATURE_SERVICE_URL=...`).
3.  Run the Uvicorn server from the **project root directory**:
    ```bash
    poetry run uvicorn model_serving.app:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000` and OpenAPI docs at `http://localhost:8000/docs`.

## Running with Docker

A multi-stage Dockerfile is provided to containerize the application efficiently and securely (non-root user).

1.  **Generate Lock File:** Make sure you have generated the `poetry.lock` file:
    ```bash
    cd model_serving
    poetry lock
    cd ..
    ```
2.  **Build the Image:** From the project root directory:
    ```bash
    docker build -t agriyield-api -f model_serving/Dockerfile .
    ```
3.  **Run the Container:**
    ```bash
    docker run -p 8000:8000 \
      -e MLFLOW_TRACKING_URI=your_mlflow_uri \
      -e FEATURE_SERVICE_URL=your_feature_service_url \
      agriyield-api
      # Optionally set other config variables, e.g.:
      # -e UVICORN_WORKERS=4 \
    ```
    Replace `<...>` with your actual URIs. **The container now runs Uvicorn with multiple worker processes (default 2, configurable via `UVICORN_WORKERS`) for better concurrency.**

## API Endpoints

Access the interactive OpenAPI documentation at the `/docs` endpoint when the service is running.

*   **`POST /predict`**: Get a single yield prediction. (Rate limited)
    *   Request Body: `PredictionRequest` (county, year)
    *   Success Response: JSON with county, year, predicted_yield.
*   **`POST /predict_batch`**: Get multiple yield predictions. (Rate limited)
    *   Request Body: `BatchPredictionRequest` (list of `PredictionRequest`)
    *   Success Response: `BatchPredictionResponse` (list of results with optional prediction/error per item).
*   **`GET /health`**: Health check endpoint. Checks model load status and Feature Service connectivity.
    *   Success Response: `{"status": "ok"}`
*   **`GET /model_info`**: Get information about the loaded model.
    *   Success Response: JSON with model name, stage, MLflow URI.
*   **`GET /metrics`**: Exposes Prometheus metrics for monitoring request counts, latency, errors, etc.

## Testing

Tests are located in the `model_serving/tests` directory and use `pytest`.

*   `test_predict.py`: Unit tests for prediction logic (mocking API calls).
*   `test_app.py`: Integration tests for API endpoints using FastAPI's `TestClient` and `pytest-httpx` for mocking the Feature Service API.

1.  Ensure development dependencies are installed (`poetry install`).
2.  Run tests from the **project root directory**:
    ```bash
    poetry run python -m pytest model_serving/tests/
    ```

## Monitoring & Resilience

*   **Structured Logging:** Logs are output in JSON format using `structlog` for easier parsing and analysis.
*   **Prometheus Metrics:** Key metrics are exposed at `/metrics`.
*   **Rate Limiting:** Prediction endpoints (`/predict`, `/predict_batch`) have rate limits configured using `slowapi`.
*   **Retry Logic:** Calls to the Feature Service API automatically retry on transient errors (connection issues, 5xx responses) using `tenacity`.
*   **Custom Exceptions:** Specific exceptions (`exceptions.py`) are used for clearer error identification.
*   **Drift:** Placeholders for data/concept drift detection exist in `drift.py`.
