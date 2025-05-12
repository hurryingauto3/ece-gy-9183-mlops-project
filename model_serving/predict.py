import pandas as pd
import structlog  # Import structlog
import httpx
import time  # Import time for latency measurement
from typing import List, Dict, Any, Tuple, Optional
from .mlflow_loader import (
    settings,
    get_app_settings,
)  # Import the dependency function here too
from .schemas import PredictionRequest, BatchPredictionResponseItem # Added Pydantic models

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
from model_serving.metrics import FEATURE_SERVICE_LATENCY, BATCH_PREDICTION_OUTCOMES # Added BATCH_PREDICTION_OUTCOMES
# Import drift monitoring function
from model_serving.drift import log_prediction_for_monitoring # Added log_prediction_for_monitoring

# Import torch
import torch
from datetime import date

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
    county: str, year: int, cut_off_date: Optional[str] = None, crop_name: Optional[str] = None
) -> Dict[str, Any]:  # Return type is the full feature service response dict
    """Fetches features for a given county, year, cut-off date, and crop from the feature service API."""
    log = logger.bind(county=county, year=year, cut_off_date=cut_off_date, crop_name=crop_name)
    app_settings = await get_app_settings() # Get app settings

    if not app_settings.feature_service.url:
        log.error("Feature service URL is not configured.")
        raise ConfigurationError("Feature service URL is not configured.")

    # MODIFIED: Robust URL joining
    base_url = str(app_settings.feature_service.url).rstrip('/') 
    endpoint_path = "features"
    feature_url = f"{base_url}/{endpoint_path}"

    params = {"county": county, "year": year}
    if cut_off_date:
        params["cut_off_date"] = cut_off_date # Feature service expects YYYY-MM-DD string
    if crop_name:
        params["crop"] = crop_name
    
    timeout = app_settings.feature_service.timeout_seconds

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
    model: Any, 
    fips_mapping: Dict[str, int], 
    crop_mapping: Dict[str, int], 
    county: str, 
    year: int, 
    cut_off_date: date, # datetime.date object from Pydantic
    crop_name: str, 
    histogram_bins: List[float]
) -> Dict[str, List[float]]: # Returns histogram dict
    """Predicts yield histogram by fetching features and running the model."""
    log = logger.bind(county=county, year=year, crop_name=crop_name, cut_off_date=str(cut_off_date))
    app_settings = await get_app_settings() # Get app settings

    try:
        log.debug("Fetching features for single histogram prediction")
        # Convert date to ISO string for the API call
        cut_off_date_str = cut_off_date.isoformat() if cut_off_date else None
        feature_response = await get_features_from_api(
            county, year, cut_off_date_str, crop_name
        )

        weather_data_list = feature_response.get("weather_data")
        if not weather_data_list:
            log.warning("Received empty weather_data list from Feature Service")
            raise FeatureNotFoundError(
                f"No valid weather data found via Feature Service for {county}, {year}, {crop_name}, up to {cut_off_date_str}."
            )

        try:
            weather_df = pd.DataFrame(weather_data_list)
            # Placeholder: Ensure weather_df columns match training order.
            # This might involve fetching expected column order from model metadata or config.
            weather_tensor = torch.tensor(weather_df.values, dtype=torch.float32)
            weather_tensor = weather_tensor.unsqueeze(0)  # Shape (1, seq_len, features)

            if county not in fips_mapping:
                log.warning("County FIPS not found in loaded FIPS mapping", fips_code=county)
                raise InvalidModelInputError(f"County FIPS '{county}' is not in the model's FIPS mapping.")
            fips_id = fips_mapping[county]
            fips_id_tensor = torch.tensor([fips_id], dtype=torch.long)

            # Handle crop ID
            if crop_name.lower() not in crop_mapping:
                log.warning("Crop name not found in loaded crop mapping", crop_name=crop_name)
                # Option: Fallback to a default crop ID if defined, or raise error
                # For now, raising error. Consider adding a default_crop_id to crop_mapping if needed.
                default_crop_key = next((k for k in crop_mapping if "default" in k.lower()), None)
                if default_crop_key:
                    crop_id = crop_mapping[default_crop_key]
                    log.info(f"Using default crop ID for '{crop_name}' as it was not found in mapping.")
                else:
                    raise InvalidModelInputError(f"Crop name '{crop_name}' is not in the model's crop mapping and no default found.")
            else:
                crop_id = crop_mapping[crop_name.lower()]
            crop_id_tensor = torch.tensor([crop_id], dtype=torch.long)
            
            # Device handling (assuming model is already on the correct device from mlflow_loader)
            # If not, move tensors to model.device
            # device = next(model.parameters()).device
            # weather_tensor = weather_tensor.to(device)
            # fips_id_tensor = fips_id_tensor.to(device)
            # crop_id_tensor = crop_id_tensor.to(device)

            log.debug(
                "Model input tensors prepared",
                weather_shape=str(weather_tensor.shape),
                fips_shape=str(fips_id_tensor.shape),
                crop_shape=str(crop_id_tensor.shape),
            )
        except Exception as e:
            log.error("Failed to prepare model input tensors", error=str(e), exc_info=True)
            raise InvalidModelInputError(f"Failed to process data into model input: {e}") from e

        try:
            model.eval()
            with torch.no_grad():
                # Model now expects weather, fips_ids, crop_ids
                logits = model(weather_tensor, fips_id_tensor, crop_id_tensor) # (B, num_bins)
            
            # Ensure output is as expected (e.g., (1, num_bins))
            if logits.ndim != 2 or logits.shape[0] != 1:
                log.error("Unexpected model output shape", shape=str(logits.shape))
                raise ModelInferenceError("Model output has unexpected shape.")

            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist() # Get list of probabilities

            # Validate number of probabilities against number of bins implied by edges
            if len(probabilities) != (len(histogram_bins) - 1):
                log.error(
                    "Mismatch between model output size and histogram_bins",
                    num_probabilities=len(probabilities),
                    num_expected_bins=len(histogram_bins) - 1
                )
                raise InvalidModelOutputError(
                    f"Model produced {len(probabilities)} probabilities, but histogram_bins imply {len(histogram_bins)-1} bins."
                )

            # Prepare histogram dictionary
            histogram_output = {
                "bin_edges": histogram_bins,
                "probabilities": probabilities
            }
            log.debug("Histogram prediction successful", histogram=histogram_output)
            return histogram_output

        except Exception as e:
            log.error("Model inference or output processing failed", error=str(e), exc_info=True)
            raise ModelInferenceError(f"Error during model inference: {e}") from e

    # FeatureNotFoundError, ConfigurationError, etc. from get_features_from_api will propagate
    # Other exceptions like InvalidModelInputError are raised above
    except ModelServingBaseError: # Re-raise known errors
        raise
    except Exception as e: # Catch any other unexpected error during the process
        log.exception("Unexpected error in predict_yield pipeline")
        raise ModelServingBaseError("An unexpected error occurred during yield prediction.") from e


# async def predict_yield_batch(
#     model: Any, 
#     fips_mapping: Dict[str, int], 
#     crop_mapping: Dict[str, int], 
#     requests: List[PredictionRequest] # Changed to List[PredictionRequest]
# ) -> List[BatchPredictionResponseItem]:
#     """Processes a batch of prediction requests concurrently."""
#     log = logger.bind(batch_size=len(requests))
#     log.info("Starting batch histogram prediction")

#     # Create a list of coroutines for each prediction
#     # Each item in `requests` is a Pydantic `PredictionRequest` model
#     tasks = [
#         predict_yield(
#             model=model,
#             fips_mapping=fips_mapping,
#             crop_mapping=crop_mapping,
#             county=req.county,
#             year=req.year,
#             cut_off_date=req.cut_off_date,
#             crop_name=req.crop,
#             histogram_bins=req.histogram_bins
#         )
#         for req in requests
#     ]

#     # Run all prediction tasks concurrently and gather results
#     # `return_exceptions=True` allows us to get exceptions instead of failing the whole batch
#     results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

#     # Process results to form BatchPredictionResponseItem list
#     response_items: List[BatchPredictionResponseItem] = []
#     for i, res_or_exc in enumerate(results_or_exceptions):
#         req_item = requests[i] # Original request for context
#         item_log = logger.bind(county=req_item.county, year=req_item.year, crop=req_item.crop, cut_off_date=str(req_item.cut_off_date))

#         if isinstance(res_or_exc, Exception):
#             error_message = f"Error processing item: {type(res_or_exc).__name__} - {str(res_or_exc)}"
#             item_log.warning("Item in batch failed", error=error_message, exc_info=False) # exc_info=False to avoid large logs for common errors
#             response_items.append(
#                 BatchPredictionResponseItem(
#                     county=req_item.county,
#                     year=req_item.year,
#                     cut_off_date=req_item.cut_off_date,
#                     crop=req_item.crop,
#                     predicted_histogram=None,
#                     error=error_message
#                 )
#             )
#             BATCH_PREDICTION_OUTCOMES.labels(outcome="error").inc()
#         else:
#             # res_or_exc is the histogram dictionary
#             item_log.info("Item in batch succeeded")
#             response_items.append(
#                 BatchPredictionResponseItem(
#                     county=req_item.county,
#                     year=req_item.year,
#                     cut_off_date=req_item.cut_off_date,
#                     crop=req_item.crop,
#                     predicted_histogram=res_or_exc, # This is the histogram dict
#                     error=None
#                 )
#             )
#             BATCH_PREDICTION_OUTCOMES.labels(outcome="success").inc()
#             # TODO: Add Prometheus histogram metric observation if applicable from res_or_exc
#             # For example, if you want to observe the mean of the predicted distribution:
#             # mean_pred = calculate_mean_from_histogram(res_or_exc["bin_edges"], res_or_exc["probabilities"])
#             # PREDICTED_YIELD_DISTRIBUTION.observe(mean_pred)

#         # Log for monitoring (simplified for batch)
#         log_prediction_for_monitoring(
#             request_data=req_item.model_dump(), # Log the full request
#             prediction=res_or_exc if not isinstance(res_or_exc, Exception) else None,
#             error=str(res_or_exc) if isinstance(res_or_exc, Exception) else None
#         )

#     log.info("Batch histogram prediction processing complete")
#     return response_items
