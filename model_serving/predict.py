import pandas as pd
import structlog  # Import structlog
import httpx
import asyncio
import time  # Import time for latency measurement
from typing import List, Dict, Any, Tuple
from model_serving.mlflow_loader import settings
# Import custom exceptions
from model_serving.exceptions import (
    FeatureServiceError, FeatureNotFoundError, FeatureRequestError,
    FeatureResponseError, InvalidModelInputError, InvalidModelOutputError,
    ConfigurationError, ModelInferenceError, ModelServingBaseError
)
# Import tenacity components
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import the custom metric from metrics module
from model_serving.metrics import FEATURE_SERVICE_LATENCY

# Get structlog logger instance
logger = structlog.get_logger(__name__)

# Create a reusable async client
async_client = httpx.AsyncClient()

# --- Define retry conditions ---
should_retry_feature_fetch = retry_if_exception_type((FeatureRequestError, FeatureResponseError))

# --- Apply retry decorator using Settings ---
# Define defaults in case settings are not loaded
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1
DEFAULT_RETRY_MAX_WAIT = 5

# Determine retry parameters from settings or use defaults
retry_attempts = settings.feature_service.retry_attempts if settings else DEFAULT_RETRY_ATTEMPTS
min_wait = settings.feature_service.retry_min_wait_seconds if settings else DEFAULT_RETRY_MIN_WAIT
max_wait = settings.feature_service.retry_max_wait_seconds if settings else DEFAULT_RETRY_MAX_WAIT

@retry(
    stop=stop_after_attempt(retry_attempts),
    wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
    retry=should_retry_feature_fetch,
    reraise=True
)
async def get_features_from_api(county: str, year: int) -> dict | None:
    """Fetches features for a given county and year from the feature service API, with retry logic."""
    log = logger.bind(county=county, year=year)

    if not settings or not settings.feature_service.url:
        log.error("Feature service URL is not configured.")
        raise ConfigurationError("Feature service URL is not configured.")

    feature_url = f"{settings.feature_service.url}/features"
    params = {"county": county, "year": year}
    # --- Use timeout from settings --- 
    timeout = settings.feature_service.timeout_seconds if settings else 10.0

    start_time = time.time()
    response = None
    try:
        log.debug("Fetching features from API", url=feature_url, params=params, timeout=timeout)
        # --- Pass timeout to httpx call --- 
        response = await async_client.get(feature_url, params=params, timeout=timeout)

        # Observe latency
        duration = time.time() - start_time
        FEATURE_SERVICE_LATENCY.observe(duration)

        # Check for server errors (5xx) before raising for status
        if response.status_code >= 500:
            log.warning("Feature service returned server error", status_code=response.status_code)
            response.raise_for_status()

        # Raise for 4xx errors *after* the 5xx check, so we don't retry 4xx
        response.raise_for_status()

        features = response.json()
        log.debug("Received features successfully", feature_keys=list(features.keys()))
        return features
    except httpx.HTTPStatusError as e:
        # Latency was already observed if response was received
        if e.response.status_code == 404:
            log.warning("Feature service returned 404 (will not retry)")
            raise FeatureNotFoundError(f"Features not found via API for county '{county}' and year {year}.") from e
        elif e.response.status_code >= 500:
            log.error("HTTP server error fetching features (will retry)", status_code=e.response.status_code, error=str(e))
            raise FeatureResponseError(f"Feature service returned status {e.response.status_code}") from e
        else:  # Other 4xx errors
            log.error("HTTP client error fetching features (will not retry)", status_code=e.response.status_code, error=str(e))
            # Raise generic FeatureServiceError for non-404 client errors that we don't retry
            raise FeatureServiceError(f"Feature service returned client error status {e.response.status_code}") from e
    except httpx.RequestError as e:
        # --- Observe latency even on request errors (e.g., timeout) ---
        duration = time.time() - start_time
        FEATURE_SERVICE_LATENCY.observe(duration)
        # -------------------------------------------------------------
        log.error("Request error fetching features (will retry)", error=str(e))
        raise FeatureRequestError(f"Failed to connect to feature service: {e}") from e
    except Exception as e:
        # --- Observe latency for other unexpected errors during/after call ---
        # Check if response exists, otherwise latency might not be meaningful if error was before call
        if response is not None:
            duration = time.time() - start_time
            FEATURE_SERVICE_LATENCY.observe(duration)
        # -------------------------------------------------------------------
        log.exception("Unexpected error fetching or parsing features (will not retry)")
        raise FeatureServiceError("An unexpected error occurred while fetching features.") from e


async def predict_yield(model, county: str, year: int) -> float:
    """Predicts yield by fetching features from API and running the model."""
    log = logger.bind(county=county, year=year)

    try:
        log.debug("Fetching features for single prediction")
        # Exceptions from get_features_from_api (FeatureNotFoundError, FeatureRequestError, etc.) will propagate
        features = await get_features_from_api(county, year)

        # --- DataFrame Creation ---
        try:
            if isinstance(features, dict):
                input_df = pd.DataFrame([features])
            else:
                # Attempt to load from JSON string if applicable (example)
                input_df = pd.read_json(features, orient='split')
            log.debug("Input DataFrame created", shape=input_df.shape)
        except Exception as e:
            log.error("Failed to create DataFrame from features", feature_type=type(features), error=str(e))
            # Raise specific InvalidModelInputError if DataFrame creation fails
            raise InvalidModelInputError(f"Could not process features into DataFrame: {e}") from e

        # --- Model Prediction ---
        try:
            result = model.predict(input_df)
            log.debug("Model prediction executed")
        except Exception as e:
            log.exception("An unexpected error occurred during model.predict()")
            # Raise generic ModelInferenceError for errors during predict() call
            raise ModelInferenceError("Model inference failed") from e

        # --- Result Processing ---
        try:
            if hasattr(result, '__len__') and len(result) > 0:
                prediction = float(result[0])  # Ensure conversion to float
                log.debug("Prediction result processed", prediction=prediction)
                return prediction
            else:
                log.error("Unexpected model output format", model_output=result)
                # Raise specific InvalidModelOutputError
                raise InvalidModelOutputError("Model prediction did not return expected format.")
        except (ValueError, TypeError) as e:
            log.error("Could not convert model output to float", model_output=result, error=str(e))
            # Raise specific InvalidModelOutputError if conversion fails
            raise InvalidModelOutputError(f"Could not process model output: {e}") from e

    # --- Catch and Re-raise Specific Errors ---
    # Catch custom errors first if specific handling is needed here (currently not)
    except (FeatureNotFoundError, FeatureRequestError, FeatureResponseError) as e:
        log.warning("Prediction failed due to feature service issue", error=str(e))
        raise  # Re-raise to be handled by API layer
    except (InvalidModelInputError, InvalidModelOutputError, ModelInferenceError) as e:
        log.warning("Prediction failed due to model inference issue", error=str(e))
        raise  # Re-raise to be handled by API layer
    except ConfigurationError as e:  # Catch config errors propagated from get_features
        log.error("Prediction failed due to configuration error", error=str(e))
        raise  # Re-raise
    # Catch broader exceptions last
    except Exception as e:
        log.exception("An unexpected error occurred in predict_yield")
        # Wrap unexpected errors in a generic base error if desired, or re-raise
        raise ModelServingBaseError("An internal error occurred during the prediction process.") from e


async def predict_yield_batch(model, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes a batch of prediction requests."""
    log = logger.bind(batch_size=len(requests))
    log.debug("Starting batch prediction process")

    results = [{} for _ in requests]
    feature_tasks = []
    original_indices = []

    # --- Feature Fetching ---
    for i, req in enumerate(requests):
        county = req.get("county")
        year = req.get("year")
        if county and year:
            feature_tasks.append(asyncio.create_task(get_features_from_api(county, year), name=f"fetch_{i}"))
            original_indices.append(i)
        else:
            log.warning("Invalid item in batch request", item_index=i, request_item=req)
            results[i] = {"county": county, "year": year, "error": "Invalid request data (missing county or year)"}

    log.debug("Gathering feature fetch tasks", task_count=len(feature_tasks))
    feature_responses = await asyncio.gather(*feature_tasks, return_exceptions=True)
    log.debug("Feature fetch tasks completed")

    valid_features = []
    valid_indices = []

    # Process feature responses, mapping exceptions to error messages
    for i, response in enumerate(feature_responses):
        original_index = original_indices[i]
        request_info = {"county": requests[original_index]["county"], "year": requests[original_index]["year"]}
        item_log = log.bind(item_index=original_index, county=request_info["county"], year=request_info["year"])

        if isinstance(response, Exception):
            # Map custom exceptions (and others) to user-friendly messages
            if isinstance(response, FeatureNotFoundError):
                error_message = f"Features not found: {response}"
            elif isinstance(response, FeatureServiceError):  # Catches RequestError, ResponseError, base FeatureServiceError
                error_message = f"Feature service error: {response}"
            elif isinstance(response, ConfigurationError):
                error_message = f"Configuration error preventing feature fetch: {response}"
            else:  # Catch other unexpected exceptions during feature fetch
                error_message = f"Unexpected error fetching features: {response}"
                item_log.error("Unexpected exception during batch feature fetch", error=str(response), exc_info=response)  # Log full exception

            item_log.warning("Error processing batch item during feature fetch", error=error_message)
            results[original_index] = {**request_info, "error": error_message}
        elif isinstance(response, dict):
            item_log.debug("Feature fetch successful for batch item")
            valid_features.append(response)
            valid_indices.append(original_index)
        else:
            error_message = f"Unexpected feature format received from API: {type(response)}"
            item_log.error("Error processing batch item due to unexpected feature format", error=error_message)
            results[original_index] = {**request_info, "error": error_message}

    if not valid_features:
        log.warning("No valid features fetched for batch prediction.")
        return results

    # --- DataFrame Creation ---
    try:
        input_df = pd.DataFrame(valid_features)
        log.debug("Batch input DataFrame created", shape=input_df.shape, valid_item_count=len(valid_features))
    except Exception as e:
        log.error("Failed to create DataFrame from batch features", error=str(e), exc_info=True)
        error_message = "Internal error processing batch features."
        # Use InvalidModelInputError conceptually, map to message for batch response
        for idx in valid_indices:
            request_info = {"county": requests[idx]["county"], "year": requests[idx]["year"]}
            results[idx] = {**request_info, "error": error_message}
        return results

    # --- Model Prediction ---
    try:
        batch_predictions = model.predict(input_df)
        log.info("Batch model prediction executed", prediction_count=len(batch_predictions))

        if len(batch_predictions) != len(valid_features):
            log.error("Mismatch between prediction count and input count", prediction_count=len(batch_predictions), input_count=len(valid_features))
            # Raise InvalidModelOutputError conceptually
            raise InvalidModelOutputError(f"Model returned {len(batch_predictions)} predictions for {len(valid_features)} inputs.")

        # Map predictions back
        for i, prediction in enumerate(batch_predictions):
            original_index = valid_indices[i]
            request_info = {"county": requests[original_index]["county"], "year": requests[original_index]["year"]}
            item_log = log.bind(item_index=original_index, county=request_info["county"], year=request_info["year"])

            try:
                results[original_index] = {**request_info, "predicted_yield": float(prediction)}
                item_log.debug("Mapped batch prediction successfully", prediction=results[original_index]["predicted_yield"])
            except (ValueError, TypeError) as e:
                item_log.error("Error processing prediction for batch item", error=str(e))
                results[original_index] = {**request_info, "error": "Invalid prediction format from model."}
    except (InvalidModelOutputError, ModelInferenceError) as e:  # Catch specific model errors
        log.error("Error during batch model prediction or output processing", error=str(e), exc_info=True)
        error_message = "Internal error during batch prediction."
        for idx in valid_indices:
            request_info = {"county": requests[idx]["county"], "year": requests[idx]["year"]}
            results[idx] = {**request_info, "error": error_message}
    except Exception as e:  # Catch other unexpected errors during prediction
        log.exception("An unexpected error occurred during batch model prediction")
        error_message = "Internal error during batch prediction."
        for idx in valid_indices:
            request_info = {"county": requests[idx]["county"], "year": requests[idx]["year"]}
            results[idx] = {**request_info, "error": error_message}

    log.debug("Finished batch prediction process")
    return results