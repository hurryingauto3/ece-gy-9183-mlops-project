import pandas as pd
import structlog  # Import structlog
import httpx
import asyncio
import time  # Import time for latency measurement
from typing import List, Dict, Any, Tuple
from .mlflow_loader import (
    settings,
    get_model_and_mapping,
)  # Import the dependency function here too

# Import custom exceptions
from model_serving.exceptions import (
    FeatureServiceError,
    FeatureNotFoundError,
    FeatureRequestError,
    FeatureResponseError,
    InvalidModelInputError,
    InvalidModelOutputError,
    ConfigurationError,
    ModelInferenceError,
    ModelServingBaseError,
)

# Import tenacity components
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Import the custom metric from metrics module
from model_serving.metrics import FEATURE_SERVICE_LATENCY

# Get structlog logger instance
logger = structlog.get_logger(__name__)

# Create a reusable async client
async_client = httpx.AsyncClient()

# --- Define retry conditions ---
should_retry_feature_fetch = retry_if_exception_type(
    (FeatureRequestError, FeatureResponseError)
)

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1
DEFAULT_RETRY_MAX_WAIT = 5

retry_attempts = (
    settings.feature_service.retry_attempts if settings else DEFAULT_RETRY_ATTEMPTS
)
min_wait = (
    settings.feature_service.retry_min_wait_seconds
    if settings
    else DEFAULT_RETRY_MIN_WAIT
)
max_wait = (
    settings.feature_service.retry_max_wait_seconds
    if settings
    else DEFAULT_RETRY_MAX_WAIT
)


@retry(
    stop=stop_after_attempt(retry_attempts),
    wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
    retry=should_retry_feature_fetch,
    reraise=True,
)
async def get_features_from_api(
    county: str, year: int
) -> Dict[str, Any] | None:  # Changed return type hint
    """Fetches features for a given county and year from the feature service API, with retry logic."""
    log = logger.bind(county=county, year=year)

    if not settings or not settings.feature_service.url:
        log.error("Feature service URL is not configured.")
        raise ConfigurationError("Feature service URL is not configured.")

    # --- Use the new Feature Service endpoint ---
    feature_url = f"{settings.feature_service.url}/features"
    # Use query parameters, not body, for GET request
    params = {"county": county, "year": year}
    # ------------------------------------------

    timeout = settings.feature_service.timeout_seconds if settings else 10.0

    start_time = time.time()
    response = None
    try:
        log.debug(
            "Fetching features from Feature Service API",
            url=feature_url,
            params=params,
            timeout=timeout,
        )
        response = await async_client.get(
            feature_url, params=params, timeout=timeout
        )  # Use get for query params

        duration = time.time() - start_time
        FEATURE_SERVICE_LATENCY.observe(duration)

        if response.status_code == 404:
            log.warning(
                "Feature Service returned 404", status_code=response.status_code
            )
            # Raise FeatureNotFoundError specifically for 404
            raise FeatureNotFoundError(
                f"Features not found via Feature Service for county '{county}' and year {year}."
            )

        # Raise for other 4xx/5xx errors
        response.raise_for_status()

        # --- Parse the expected FeatureService response structure ---
        feature_response_data = response.json()
        # Expecting a dictionary with 'fips_code', 'year', 'weather_data'
        if (
            not isinstance(feature_response_data, dict)
            or "weather_data" not in feature_response_data
        ):
            log.error(
                "Unexpected response format from Feature Service",
                response_body=response.text,
            )
            raise FeatureResponseError(
                "Unexpected response format from Feature Service."
            )

        log.debug(
            "Received features successfully from Feature Service",
            data_points=len(feature_response_data.get("weather_data", [])),
        )
        # Return the full response dictionary, predict_yield will extract what it needs
        return feature_response_data

    except httpx.HTTPStatusError as e:
        # Latency already observed
        # 404 is handled above, here we catch other HTTP errors
        if e.response.status_code >= 500:
            log.error(
                "Feature Service returned server error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise FeatureResponseError(
                f"Feature Service returned status {e.response.status_code}"
            ) from e
        else:  # Other 4xx errors (e.g., 400, 422)
            log.error(
                "Feature Service returned client error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise FeatureServiceError(
                f"Feature Service returned client error status {e.response.status_code}"
            ) from e
    except httpx.RequestError as e:
        duration = time.time() - start_time
        FEATURE_SERVICE_LATENCY.observe(duration)
        log.error(
            "Request error fetching features from Feature Service (will retry)",
            error=str(e),
        )
        raise FeatureRequestError(f"Failed to connect to Feature Service: {e}") from e
    except Exception as e:
        if response is not None:
            duration = time.time() - start_time
            FEATURE_SERVICE_LATENCY.observe(duration)
        log.exception(
            "Unexpected error fetching or parsing features from Feature Service"
        )
        raise FeatureServiceError(
            "An unexpected error occurred while fetching features from Feature Service."
        ) from e


async def predict_yield(
    model, fips_mapping: Dict[str, int], county: str, year: int
) -> float:
    """Predicts yield by fetching features from Feature Service and running the model."""
    log = logger.bind(county=county, year=year)

    try:
        log.debug("Fetching features for single prediction from Feature Service")
        # get_features_from_api now returns the dictionary response, not just weather data
        feature_response = await get_features_from_api(
            county, year
        )  # Exceptions propagate

        # --- Extract data from Feature Service Response and Prepare Model Input ---
        weather_data_list = feature_response.get("weather_data")
        if not weather_data_list:
            # This case should ideally be handled by get_features_from_api raising 404 if no data
            # But defensive check here.
            log.warning("Received empty weather_data list from Feature Service")
            raise FeatureNotFoundError(
                f"No valid weather data found in Features Service response for {county}, {year}."
            )

        # Convert list of dicts to pandas DataFrame, then to tensor
        try:
            weather_df = pd.DataFrame(weather_data_list)
            # Ensure columns match training order and drop non-feature columns if any slipped through
            # This is a critical step for consistency. You might need to store the
            # list of expected weather feature column names from your training process.
            # For now, assume the FS provides exactly the weather columns needed in the correct order.
            weather_tensor = torch.tensor(weather_df.values, dtype=torch.float32)
            # Add batch dimension (batch size = 1 for single prediction)
            weather_tensor = weather_tensor.unsqueeze(0)  # Shape (1, seq_len, features)

            # Get FIPS ID from the mapping
            if county not in fips_mapping:
                log.warning(
                    "County FIPS not found in loaded FIPS mapping", county=county
                )
                raise InvalidModelInputError(
                    f"County FIPS '{county}' is not in the model's trained mapping."
                )
            fips_id = fips_mapping[county]
            fips_id_tensor = torch.tensor([fips_id], dtype=torch.long)  # Shape (1,)

            # Move tensors to device (assuming device is determined in app.py or passed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weather_tensor = weather_tensor.to(device)
            fips_id_tensor = fips_id_tensor.to(device)

            log.debug(
                "Model input tensors prepared",
                weather_shape=weather_tensor.shape,
                fips_shape=fips_id_tensor.shape,
            )

        except Exception as e:
            log.error(
                "Failed to prepare model input tensors from Feature Service data",
                error=str(e),
                exc_info=True,
            )
            raise InvalidModelInputError(
                f"Failed to process Feature Service data into model input: {e}"
            ) from e

        # --- Model Prediction ---
        try:
            # Ensure model is in evaluation mode for inference (dropout off)
            model.eval()
            with torch.no_grad():
                prediction = model(weather_tensor, fips_id_tensor)

            log.debug("Model prediction executed")
        except Exception as e:
            log.exception("An unexpected error occurred during model inference")
            raise ModelInferenceError("Model inference failed.") from e

        # --- Result Processing ---
        try:
            # prediction is a tensor, convert to scalar float
            prediction_scalar = prediction.item()  # Assuming batch size 1 output

            log.debug("Prediction result processed", prediction=prediction_scalar)
            return prediction_scalar
        except Exception as e:
            log.error(
                "Could not convert model output tensor to scalar",
                output_type=type(prediction),
                output_value=prediction,
                error=str(e),
            )
            raise InvalidModelOutputError(f"Could not process model output: {e}") from e

    # --- Catch and Re-raise Specific Errors ---
    # Catch custom errors propagated from get_features_from_api or created here
    except (
        FeatureNotFoundError,
        FeatureRequestError,
        FeatureResponseError,
        FeatureServiceError,
    ) as e:
        log.warning("Prediction failed due to feature service issue", error=str(e))
        raise  # Re-raise to be handled by API layer
    except (InvalidModelInputError, InvalidModelOutputError, ModelInferenceError) as e:
        log.warning("Prediction failed due to model inference issue", error=str(e))
        raise  # Re-raise to be handled by API layer
    except ConfigurationError as e:
        log.error("Prediction failed due to configuration error", error=str(e))
        raise  # Re-raise
    except Exception as e:
        log.exception("An unexpected error occurred in predict_yield")
        raise ModelServingBaseError(
            "An internal error occurred during the prediction process."
        ) from e


async def predict_yield_batch(
    model, fips_mapping: Dict[str, int], requests: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Processes a batch of prediction requests."""
    log = logger.bind(batch_size=len(requests))
    log.debug("Starting batch prediction process")

    results = [{} for _ in requests]
    feature_tasks = []
    original_indices = []

    # --- Feature Fetching (Batch) ---
    # Call get_features_from_api for each request concurrently
    for i, req in enumerate(requests):
        county = req.get("county")
        year = req.get("year")
        if county and year:
            # Store original request info with the task index
            feature_tasks.append(
                asyncio.create_task(
                    get_features_from_api(county, year), name=f"fetch_batch_{i}"
                )
            )
            original_indices.append(i)
        else:
            log.warning("Invalid item in batch request", item_index=i, request_item=req)
            results[i] = {
                "county": req.get("county"),
                "year": req.get("year"),
                "error": "Invalid request data (missing county or year)",
            }

    log.debug("Gathering batch feature fetch tasks", task_count=len(feature_tasks))
    # Use asyncio.gather to run tasks concurrently
    feature_responses = await asyncio.gather(*feature_tasks, return_exceptions=True)
    log.debug("Batch Feature Service tasks completed")

    # --- Process Batch Feature Responses and Prepare for Model ---
    # Create lists to collect weather tensors and FIPS ID tensors for padding/batching
    weather_tensors_list = []  # List of tensors (variable length)
    fips_id_tensors_list = []  # List of FIPS ID tensors (scalar)
    valid_original_indices = (
        []
    )  # Indices in the *original* requests list for valid items

    for i, response in enumerate(feature_responses):
        original_index = original_indices[i]
        request_info = {
            "county": requests[original_index]["county"],
            "year": requests[original_index]["year"],
        }
        item_log = log.bind(
            item_index=original_index,
            county=request_info["county"],
            year=request_info["year"],
        )

        if isinstance(response, Exception):
            # Handle exceptions from get_features_from_api
            if isinstance(response, FeatureNotFoundError):
                error_message = f"Features not found: {response}"
            elif isinstance(response, FeatureServiceError):
                error_message = f"Feature service error: {response}"
            elif isinstance(response, ConfigurationError):
                error_message = f"Configuration error: {response}"
            else:
                error_message = f"Unexpected error fetching features: {response}"
                item_log.error(
                    "Unexpected exception during batch feature fetch",
                    error=str(response),
                    exc_info=response,
                )

            item_log.warning(
                "Error processing batch item during feature fetch", error=error_message
            )
            results[original_index] = {**request_info, "error": error_message}

        elif isinstance(response, dict) and "weather_data" in response:
            # Process valid response from Feature Service
            weather_data_list = response.get("weather_data")
            if not weather_data_list:
                item_log.warning("Received empty weather_data list in response")
                results[original_index] = {
                    **request_info,
                    "error": f"No valid weather data found in season for {request_info['county']}, {request_info['year']}.",
                }
                continue  # Skip to the next response

            try:
                # Convert list of dicts to DataFrame, then to tensor
                weather_df = pd.DataFrame(weather_data_list)
                weather_tensor = torch.tensor(
                    weather_df.values, dtype=torch.float32
                )  # Shape (seq_len, features)
                weather_tensors_list.append(weather_tensor)

                # Get FIPS ID tensor
                county = request_info["county"]  # Use county from original request
                if county not in fips_mapping:
                    item_log.warning(
                        "County FIPS not found in loaded FIPS mapping", county=county
                    )
                    results[original_index] = {
                        **request_info,
                        "error": f"County FIPS '{county}' is not in the model's trained mapping.",
                    }
                    continue  # Skip this item
                fips_id = fips_mapping[county]
                fips_id_tensor = torch.tensor(
                    fips_id, dtype=torch.long
                )  # Shape () - scalar
                fips_id_tensors_list.append(fips_id_tensor)

                valid_original_indices.append(original_index)
                item_log.debug("Feature data processed successfully for batch item")

            except Exception as e:
                item_log.error(
                    "Failed to process Feature Service data into tensors for batch item",
                    error=str(e),
                    exc_info=True,
                )
                results[original_index] = {
                    **request_info,
                    "error": f"Failed to process feature data: {e}",
                }

        else:
            # Unexpected response type
            error_message = (
                f"Unexpected response type from Feature Service: {type(response)}"
            )
            item_log.error(
                "Unexpected response type for batch item", response_type=type(response)
            )
            results[original_index] = {**request_info, "error": error_message}

    if not weather_tensors_list:
        log.warning("No valid items to process after fetching features for the batch.")
        # Ensure any items that didn't get processed have an error set
        for i, result in enumerate(results):
            if (
                not result
            ):  # If still empty, it means it wasn't processed/assigned an error
                results[i] = {
                    "county": requests[i].get("county"),
                    "year": requests[i].get("year"),
                    "error": "Item skipped due to upstream error.",
                }

        return results  # Return batch results with errors

    # --- Prepare Batch Tensors using collate_fn ---
    try:
        # collate_fn expects a list of tuples: [(weather_tensor, dummy_y, fips_id), ...]
        # We only have weather_tensor and fips_id_tensor here.
        # We can adapt or manually replicate collate_fn's padding logic.
        # Let's manually replicate padding and stacking for batch prediction.

        # Padding weather sequences
        # Use the collate_fn logic directly but with our lists
        # Need to add a dummy element for the second item in the tuple since collate_fn expects 3
        batch_for_collate = [
            (w_t, torch.tensor(0.0), f_id_t)
            for w_t, f_id_t in zip(weather_tensors_list, fips_id_tensors_list)
        ]

        weather_batch_padded, _, fips_batch_stacked = torch.nn.utils.rnn.collate_fn(
            batch_for_collate
        )

        # Move batch tensors to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weather_batch_padded = weather_batch_padded.to(device)
        fips_batch_stacked = fips_batch_stacked.to(device)

        log.debug(
            "Batch tensors prepared for model",
            weather_batch_shape=weather_batch_padded.shape,
            fips_batch_shape=fips_batch_stacked.shape,
        )

    except Exception as e:
        log.error("Failed to pad and stack batch tensors", error=str(e), exc_info=True)
        error_message = "Internal error preparing data for model."
        # Mark all currently valid items as errored
        for idx in valid_original_indices:
            request_info = {
                "county": requests[idx]["county"],
                "year": requests[idx]["year"],
            }
            results[idx] = {**request_info, "error": error_message}
        return results  # Return batch results with errors

    # --- Model Prediction (Batch) ---
    try:
        # Ensure model is in evaluation mode
        model.eval()
        with torch.no_grad():
            batch_predictions_tensor = model(weather_batch_padded, fips_batch_stacked)

        log.info(
            "Batch model prediction executed",
            prediction_count=batch_predictions_tensor.size(0),
        )

        # --- Process Batch Results ---
        batch_predictions_list = (
            batch_predictions_tensor.cpu().numpy().tolist()
        )  # Convert to Python list

        if len(batch_predictions_list) != len(valid_original_indices):
            log.error(
                "Mismatch between prediction count and valid input count",
                prediction_count=len(batch_predictions_list),
                valid_input_count=len(valid_original_indices),
            )
            raise InvalidModelOutputError(
                f"Model returned {len(batch_predictions_list)} predictions for {len(valid_original_indices)} inputs."
            )

        # Map predictions back to the original results list using valid_original_indices
        for i, prediction_scalar in enumerate(batch_predictions_list):
            original_index = valid_original_indices[i]
            request_info = {
                "county": requests[original_index]["county"],
                "year": requests[original_index]["year"],
            }
            # Place the successful prediction in the result list at the original index
            results[original_index] = {
                **request_info,
                "predicted_yield": prediction_scalar,
            }
            log.debug(
                "Mapped batch prediction successfully",
                item_index=original_index,
                prediction=prediction_scalar,
            )

    except (InvalidModelOutputError, ModelInferenceError) as e:
        log.error(
            "Error during batch model prediction or output processing",
            error=str(e),
            exc_info=True,
        )
        error_message = "Internal error during batch prediction."
        # Mark all currently valid items as errored
        for idx in valid_original_indices:
            request_info = {
                "county": requests[idx]["county"],
                "year": requests[idx]["year"],
            }
            results[idx] = {**request_info, "error": error_message}
    except Exception as e:
        log.exception("An unexpected error occurred during batch model prediction")
        error_message = "Internal error during batch prediction."
        for idx in valid_original_indices:
            request_info = {
                "county": requests[idx]["county"],
                "year": requests[idx]["year"],
            }
            results[idx] = {**request_info, "error": error_message}

    log.debug("Finished batch prediction process")
    return results
