import logging
import time
import os
import asyncio
import sys
import structlog
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
    Body,
    Depends,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, HttpUrl, Field
from typing import List, Dict, Any, Optional

# Import modules needed by the app
# Note: Ensure 'torch' is implicitly handled by predict.py or dependencies
from model_serving.predict import predict_yield, predict_yield_batch, async_client
# Consolidate imports from mlflow_loader
from model_serving.mlflow_loader import (
    get_model_and_mapping, # This is the async dependency function
    settings, # Global settings instance
    ModelServingBaseError, # Custom exception
    ConfigurationError,  # Custom exception
    # No need to import get_model_and_mapping or settings from .mlflow_loader again
)
from model_serving.drift import check_data_drift, log_prediction_for_monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import httpx # Import httpx for exceptions (already there)
from model_serving.metrics import (
    BATCH_PREDICTION_OUTCOMES,
    PREDICTED_YIELD_DISTRIBUTION,
    FEATURE_SERVICE_LATENCY, # Make sure this metric is imported
)
# Import custom exceptions
from model_serving.exceptions import (
    FeatureNotFoundError,
    FeatureServiceError,
    InvalidModelInputError,
    InvalidModelOutputError,
    ModelInferenceError,
)

# --- Configure Structured Logging ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
root_logger = logging.getLogger()

# Remove existing handlers to avoid duplicate logs if any were configured before (good practice in modular apps)
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(handler)
# Use settings for log level if available, otherwise default
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper() # Read LOG_LEVEL from env, default to INFO
log_level = getattr(logging, log_level_str, logging.INFO)
root_logger.setLevel(log_level)


logger = structlog.get_logger(__name__)
# ------------------------------------

# --- Configure Rate Limiting using Settings ---
# Use a default if settings failed to load, or handle error appropriately
default_limit = "10/minute" # Single prediction default
default_batch_limit = "5/minute" # Batch prediction default
limiter = Limiter(
    key_func=get_remote_address,
    # We don't set a default_limits list here, but apply limits per endpoint using @limiter.limit
    # Using enabled=settings is not None will prevent errors if settings load fails
    enabled=settings is not None
)
# ---------------------------------------------

# --- API Metadata ---
tags_metadata = [
    {
        "name": "Predictions",
        "description": "Endpoints for generating yield predictions.",
    },
    {
        "name": "Service Info",
        "description": "Endpoints for service health and model information.",
    },
]

# --- FastAPI App Instance with Metadata ---
app = FastAPI(
    title="AgriYield Model API",
    description="API for predicting agricultural yield based on county and year.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# --- Add Rate Limiting State and Middleware/Handler ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Apply SlowAPIMiddleware
if settings is not None: # Only add middleware if settings loaded successfully
    app.add_middleware(SlowAPIMiddleware)
else:
    logger.warning("Settings failed to load, Rate Limiting middleware will be skipped.")
# ----------------------------------------------------

# --- Add Prometheus Metrics ---
# Instrumentator should be added after other middleware like SlowAPI
Instrumentator().instrument(app).expose(app)
# ----------------------------


# Middleware for logging requests (Update to use structlog)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # Bind request details to context for all logs within the request scope
    structlog.contextvars.bind_contextvars(
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown",
        request_id=request.headers.get(
            "X-Request-ID", "N/A"
        ),
    )
    logger.info("Received request")

    response = await call_next(request)

    process_time = time.time() - start_time
    # Log end of request with status and duration
    logger.info(
        "Request finished",
        status_code=response.status_code,
        process_time_seconds=round(process_time, 4),
    )
    structlog.contextvars.clear_contextvars()
    return response


# --- Pydantic Models with Examples ---
class PredictionRequest(BaseModel):
    county: str = Field(..., example="TestCounty", description="Name of the county.")
    year: int = Field(
        ..., example=2023, description="Year for the prediction (e.g., 1980-2050)."
    )

    @field_validator("year")
    def year_must_be_reasonable(cls, v):
        if not (1980 <= v <= 2050):  # Example range, adjust as needed
            raise ValueError("Year must be between 1980 and 2050")
        return v

    @field_validator("county")
    def county_must_not_be_empty(cls, v):
        if not v or v.isspace():
            raise ValueError("County cannot be empty")
        # Add more specific county validation if a list of valid counties exists
        return v.strip()


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(
        ..., min_length=1, description="A list of prediction requests."
    )


class BatchPredictionResponseItem(BaseModel):
    county: str = Field(
        ..., example="TestCounty", description="Name of the county from the request."
    )
    year: int = Field(..., example=2023, description="Year from the request.")
    predicted_yield: Optional[float] = Field(
        None, example=45.67, description="Predicted yield value, if successful."
    )
    error: Optional[str] = Field(
        None,
        example="Features not found",
        description="Error message, if prediction failed for this item.",
    )


class BatchPredictionResponse(BaseModel):
    responses: List[BatchPredictionResponseItem] = Field(
        ..., description="List of results corresponding to the batch requests."
    )


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="AgriYieldPredictor")
    model_stage: str = Field(..., example="Production")
    mlflow_uri: str = Field(..., example="http://mlflow.example.com")
    # Add model_version if tracked in mlflow_loader
    # model_version: str = Field(..., example="1")


# --- API Endpoints with Tags and Descriptions ---


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Get a single yield prediction",
    description="Fetch features and predict yield for a single county and year.",
    response_description="The prediction result or error information.",
)
@limiter.limit(
    lambda: settings.api.predict_limit if settings else default_limit
)
async def get_prediction(
    request: Request, # Passed by SlowAPI middleware
    req: PredictionRequest = Body(
        ..., example={"county": "SampleCounty", "year": 2024}
    ),
    # CORRECT: Pass the callable function name
    model_tuple=Depends(get_model_and_mapping),
):
    # Unpack the model and mapping from the dependency result
    model, fips_mapping = model_tuple

    structlog.contextvars.bind_contextvars(county=req.county, year=req.year)
    prediction_result = None
    error_message = None
    try:
        # check_data_drift(req.model_dump()) # Data drift check needs to be robust (handle potentially empty features)
        logger.info("Starting prediction")
        # Pass the fips_mapping to predict_yield
        prediction_result = await predict_yield(
            model, fips_mapping, req.county, req.year
        )
        logger.info("Prediction successful", predicted_yield=prediction_result)

        if prediction_result is not None:
            PREDICTED_YIELD_DISTRIBUTION.observe(prediction_result)

        # log_prediction_for_monitoring will be called in finally block

        return {
            "county": req.county,
            "year": req.year,
            "predicted_yield": prediction_result,
        }
    except FeatureNotFoundError as e:
        error_message = str(e)
        logger.warning("Data not found for prediction", error=error_message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_message)
    except (
        FeatureServiceError,
        ConfigurationError,
        ModelServingBaseError,
    ) as e: # Catch specific service/config errors
        error_message = f"Service dependency failed: {e}"
        logger.error(
            "Service dependency or configuration error during prediction",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message
        )
    except (
        InvalidModelInputError,
        InvalidModelOutputError,
        ModelInferenceError,
    ) as e: # Catch specific model errors
        error_message = f"Invalid data or model output: {e}"
        logger.warning("Invalid data or model output during prediction", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=error_message
        )
    except Exception as e: # Catch any other unexpected errors
        error_message = "An internal server error occurred during prediction."
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
        )
    finally:
        # Ensure log_prediction_for_monitoring is called even if an exception occurs before return
        log_prediction_for_monitoring( # This function handles error logging internally
            request_data=req.model_dump(),
            prediction=prediction_result if error_message is None else None,
            error=error_message,
        )
        structlog.contextvars.unbind_contextvars("county", "year")


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Get multiple yield predictions in batch",
    description="Fetch features concurrently and predict yield for multiple county/year pairs.",
    response_description="A list of prediction results or errors for each item in the batch request.",
)
@limiter.limit(
    lambda: settings.api.predict_batch_limit if settings else default_batch_limit
)
async def get_batch_predictions(
    request: Request, # Passed by SlowAPI middleware
    batch_req: BatchPredictionRequest = Body(
        ...,
        example={
            "requests": [
                {"county": "CountyA", "year": 2022},
                {"county": "CountyB", "year": 2023},
            ]
        },
    ),
    # CORRECT: Pass the callable function name
    model_tuple=Depends(get_model_and_mapping),
):
    # Unpack the model and mapping
    model, fips_mapping = model_tuple

    request_count = len(batch_req.requests)
    structlog.contextvars.bind_contextvars(batch_size=request_count)
    logger.info("Received batch prediction request")
    # Initialize results list with placeholders for each request item
    results = [{ "county": r.county, "year": r.year, "predicted_yield": None, "error": None } for r in batch_req.requests]


    try:
        # check_data_drift(...) # Adapt drift check for batch input

        request_dicts = [req.model_dump() for req in batch_req.requests]

        # Pass the model, mapping, and request_dicts to predict_yield_batch
        processed_results = await predict_yield_batch(model, fips_mapping, request_dicts)
        # predict_yield_batch is expected to return a list of dictionaries mirroring BatchPredictionResponseItem structure

        # Update the original results list with processed_results based on county/year or index
        # Assuming predict_yield_batch returns results in the same order as requests
        for i, item_result in enumerate(processed_results):
             results[i] = item_result # Overwrite the placeholder with the result

        # Log metrics and monitoring for each item in the *final* results list
        success_count = 0
        failure_count = 0
        for item_result in results:
            predicted_yield = item_result.get("predicted_yield")
            error_message = item_result.get("error")

            log_prediction_for_monitoring(
                request_data={"county": item_result.get("county"), "year": item_result.get("year")},
                prediction=predicted_yield,
                error=error_message,
            )

            if error_message is None and predicted_yield is not None:
                BATCH_PREDICTION_OUTCOMES.labels(outcome="success").inc()
                PREDICTED_YIELD_DISTRIBUTION.observe(predicted_yield)
                success_count += 1
            else:
                BATCH_PREDICTION_OUTCOMES.labels(outcome="error").inc()
                failure_count += 1


        logger.info(
            "Batch results summary",
            success_count=success_count,
            failure_count=failure_count,
        )
        return BatchPredictionResponse(responses=results)

    except (
        ConfigurationError,
        ModelServingBaseError,
    ) as e: # Catch fatal errors during batch processing setup/teardown
        logger.error(
            "Fatal Model serving error during batch prediction setup",
            error=str(e),
            exc_info=True,
        )
        # Log errors for each item and return errors for all items
        for i, req in enumerate(batch_req.requests):
            error_message = f"Fatal service error during batch setup: {e}"
            results[i] = {
                 "county": req.county,
                 "year": req.year,
                 "error": error_message,
            }
            BATCH_PREDICTION_OUTCOMES.labels(outcome="error").inc()
            log_prediction_for_monitoring(
                request_data={"county": req.county, "year": req.year},
                prediction=None,
                error=error_message,
            )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Fatal service error during batch setup: {e}",
        )

    except Exception as e: # Catch any other unexpected errors during batch processing
        logger.exception(
            "Unexpected error during batch prediction setup or finalization"
        )
        # Ensure all items have an error if a global error occurred
        for i, req in enumerate(batch_req.requests):
             if "error" not in results[i]: # Only assign if not already assigned by predict_yield_batch
                 error_message = "An internal error occurred during batch processing."
                 results[i]["error"] = error_message
                 BATCH_PREDICTION_OUTCOMES.labels(outcome="error").inc()
                 log_prediction_for_monitoring(
                     request_data={"county": req.county, "year": req.year},
                     prediction=None,
                     error=error_message,
                 )


        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during batch processing.",
        )

    finally:
        structlog.contextvars.unbind_contextvars("batch_size")


# --- Model Info Endpoint ---
@app.get(
    "/model_info",
    response_model=ModelInfoResponse,
    tags=["Service Info"],
    summary="Get loaded model information",
    description="Returns details about the ML model currently loaded by the service.",
    response_description="Information about the loaded model.",
)
def model_info():
    # No dependency needed here, just report configuration/globals
    if loaded_model_instance is None or loaded_fips_mapping is None:
         logger.warning("Model info requested but model not loaded.")
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="Model or FIPS mapping is not available due to loading error.",
         )

    # Retrieve loaded model info from settings (loaded at startup)
    model_name = settings.mlflow.model_name
    model_stage = settings.mlflow.model_stage
    mlflow_uri = settings.mlflow.tracking_uri

    # Optionally, add details about the specific model version loaded
    # This requires storing the version info during load_ml_model in mlflow_loader.py
    # and making it accessible here (e.g., as a global variable or part of the loaded_model_instance tuple)
    # version = loaded_model_version if 'loaded_model_version' in globals() else "unknown"

    logger.info(
        "Returning model info",
        model_name=model_name,
        model_stage=model_stage,
        mlflow_uri=mlflow_uri,
        # model_version=version # Add if available
    )
    return {
        "model_name": model_name,
        "model_stage": model_stage,
        "mlflow_uri": mlflow_uri,
        # "model_version": version # Add this if you track it
    }


# --- Health Check Endpoint (Consolidated and Fixed) ---
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service Info"],
    summary="Perform a health check",
    description="Checks if the service is running and essential dependencies (model, feature service) are available.",
    response_description="Service status.",
)
# CORRECT: Dependency ensures model is loaded or throws ModelServingBaseError/ConfigurationError
async def health_check(
    model_tuple=Depends(get_model_and_mapping), # <-- CORRECTED LINE - removes ()
):
    # The dependency 'get_model_and_mapping' ensures the model is loaded.
    # The fact that this function is reached without the dependency raising an exception
    # means the model loading part of the health check passed.

    # Check feature service connectivity only if URL is configured and settings loaded
    if settings and settings.feature_service and settings.feature_service.url:
        feature_service_ok = False
        # Use the client initialized globally in predict.py
        # Use the feature service URL directly, stripping potential trailing slashes
        feature_service_base_url = str(settings.feature_service.url).rstrip("/")

        try:
            # Option 1: Ping the base URL (like the second definition did)
            # start_time = time.time()
            # response = await async_client.head(feature_service_base_url, timeout=settings.feature_service.timeout_seconds)
            # elapsed_time = time.time() - start_time
            # FEATURE_SERVICE_LATENCY.observe(elapsed_time)
            # if 200 <= response.status_code < 400: # Check for success codes
            #     feature_service_ok = True
            #     logger.debug("Health check: Feature service base URL reachable", url=feature_service_base_url, status_code=response.status_code)
            # else:
            #      logger.warning("Health check: Feature service base URL returned unexpected status", url=feature_service_base_url, status_code=response.status_code)

            # Option 2: Ping a specific /health endpoint on FS (assuming it exists and is standard)
            feature_service_health_url = f"{feature_service_base_url}/health"
            start_time = time.time()
            # Use GET for a health endpoint
            response = await async_client.get(feature_service_health_url, timeout=settings.feature_service.timeout_seconds)
            elapsed_time = time.time() - start_time
            FEATURE_SERVICE_LATENCY.observe(elapsed_time) # Log latency for this request

            if response.status_code == 200: # Check specifically for 200 OK for a health endpoint
                feature_service_ok = True
                logger.debug(
                    "Health check: Feature service reachable and healthy",
                    url=feature_service_health_url,
                    status_code=response.status_code,
                )
            else:
                # Treat any non-200 from a health endpoint as unhealthy
                logger.warning(
                    "Health check: Feature service reachable but unhealthy or error status",
                    url=feature_service_health_url,
                    status_code=response.status_code,
                    response_body=response.text, # Log body for non-200 responses
                )


        except httpx.RequestError as e:
            # Note: Latency for failed requests might not be captured by the Histogram unless added specifically in except block
            logger.error(
                "Health check failed: Could not connect to Feature Service health endpoint",
                error=str(e),
                url=f"{feature_service_base_url}/health", # Log the specific health check URL
            )
        except Exception as e:
            logger.exception(
                "Health check failed: Unexpected error during Feature Service health check"
            )

        if not feature_service_ok:
            # Raise 503 if feature service check failed
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Dependency failed: Cannot reach Feature Service or it's unhealthy",
            )
    else:
        logger.warning(
            "Health check: Feature Service URL is None or not configured, skipping connectivity check."
        )


    # If we reached here, the model loaded successfully (due to Depends) AND the Feature Service is reachable/healthy (if configured)
    logger.debug("Health check successful")
    return {"status": "ok"}


@app.exception_handler(ModelServingBaseError)
async def model_serving_exception_handler(request: Request, exc: ModelServingBaseError):
    logger.error(f"ModelServingBaseError caught by handler: {exc}", exc_info=True)
    # Return 503 for service dependency errors
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Service dependency failed: {exc}"},
    )

# Add specific handlers for other custom exceptions if needed, or let them fall through

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing httpx client...")
    # Check if async_client was successfully initialized
    if async_client:
        await async_client.aclose()
        logger.info("Httpx client closed.")
    else:
        logger.warning("Httpx client was not initialized, skipping close.")