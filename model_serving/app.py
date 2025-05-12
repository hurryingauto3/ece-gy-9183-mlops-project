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
from typing import List, Dict, Any, Optional
from datetime import date # Import date

# Import modules needed by the app
# Note: Ensure 'torch' is implicitly handled by predict.py or dependencies
from model_serving.predict import predict_yield, predict_yield_batch, async_client
# Consolidate imports from mlflow_loader
from model_serving.mlflow_loader import (
    get_model_and_mapping, # This is the async dependency function
    get_app_settings, # ADDED: For startup and /model_info
    # settings, # settings is loaded via get_app_settings now
    ModelServingBaseError, # Custom exception
    ConfigurationError,  # Custom exception
    AppSettings, # <--- ADD THIS IMPORT
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

# Import schemas
from .schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    BatchPredictionResponseItem,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
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

def get_dynamic_predict_limit(request: Request) -> str:
    print(f"!!! DEBUG: get_dynamic_predict_limit CALLED. Request type: {type(request)}")
    # Directly return default_limit to avoid any other potential errors within this function
    return default_limit

# def get_dynamic_predict_batch_limit(request: Request) -> str:
#     print(f"!!! DEBUG: get_dynamic_predict_batch_limit CALLED. Request type: {type(request)}")
#     # Directly return default_batch_limit to avoid any other potential errors within this function
#     return default_batch_limit

limiter = Limiter(
    key_func=get_remote_address,
    # We don't set a default_limits list here, but apply limits per endpoint using @limiter.limit
    # Using enabled=settings is not None will prevent errors if settings load fails
    # This should be enabled=True and rely on get_app_settings to fail first if config is bad.
    # However, if settings object itself is None at this point of definition, it may error.
    # Safest is to initialize Limiter and then check settings inside the lambda for limits.
    enabled=True # Rate limiting should be enabled by default
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
# We will ensure settings are loaded by get_app_settings at startup or first request.
app.add_middleware(SlowAPIMiddleware)
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


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Get a single yield histogram prediction",
    description="Fetch features up to a cut-off date for a specific crop and predict yield histogram for a county and year.",
    response_description="The histogram prediction result or error information.",
)
# @limiter.limit(
#     get_dynamic_predict_limit  # Use the new async helper function
# )
async def get_prediction(
    request: Request, # Passed by SlowAPI middleware
    req: PredictionRequest = Body(
        ..., 
        example=PredictionRequest(
            county="19153", 
            year=2023, 
            cut_off_date="2023-08-01", 
            crop="corn", 
            histogram_bins=[0, 50, 100, 150, 200, 250]
        ).model_dump()
    ),
    model_data=Depends(get_model_and_mapping), # Renamed for clarity: model, fips_map, crop_map
):
    model, fips_mapping, crop_mapping = model_data # Unpack all three

    structlog.contextvars.bind_contextvars(county=req.county, year=req.year, crop=req.crop, cut_off_date=req.cut_off_date)
    prediction_output = None
    error_message = None
    status_code = status.HTTP_200_OK

    try:
        logger.info("Starting histogram prediction")
        # predict_yield will now return a dict for the histogram
        histogram_data = await predict_yield(
            model=model,
            fips_mapping=fips_mapping,
            crop_mapping=crop_mapping, # Pass crop_mapping
            county=req.county,
            year=req.year,
            cut_off_date=req.cut_off_date,
            crop_name=req.crop, # Pass crop name
            histogram_bins=req.histogram_bins # Pass histogram_bins
        )
        prediction_output = BatchPredictionResponseItem(
            county=req.county,
            year=req.year,
            cut_off_date=req.cut_off_date,
            crop=req.crop,
            predicted_histogram=histogram_data,
            error=None
        )
        # TODO: Log prediction_output for monitoring (consider what part of histogram to log)
        # log_prediction_for_monitoring(req.county, req.year, req.crop, histogram_data) 

    except ModelServingBaseError as e:
        logger.warning("Prediction failed due to known error", error_type=type(e).__name__, detail=str(e))
        error_message = str(e)
        status_code = e.status_code if hasattr(e, 'status_code') else status.HTTP_400_BAD_REQUEST
        # prediction_output remains None, will be structured in the final response part
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        error_message = "An unexpected error occurred."
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        # prediction_output remains None

    if error_message:
        # For single prediction, if error, return the error directly with appropriate status code
        # The BatchPredictionResponseItem structure is more for the batch endpoint's itemization
        # However, to keep it somewhat consistent for now, or if we decide to wrap single responses similarly:
        final_response_item = BatchPredictionResponseItem(
            county=req.county, year=req.year, cut_off_date=req.cut_off_date, crop=req.crop, predicted_histogram=None, error=error_message
        )
        # MODIFIED: Added mode='json' for proper date serialization
        return JSONResponse(status_code=status_code, content=final_response_item.model_dump(mode='json', exclude_none=True))

    return prediction_output # This is already a BatchPredictionResponseItem


# @app.post(
#     "/predict_batch",
#     response_model=BatchPredictionResponse,
#     tags=["Predictions"],
#     summary="Get multiple yield histogram predictions in batch",
#     description="Fetch features concurrently and predict yield histograms for multiple county/year/crop/date combinations.",
#     response_description="A list of histogram prediction results or errors for each item in the batch request.",
# )
# # @limiter.limit(
# #     get_dynamic_predict_batch_limit  # Use the new async helper function
# # )
# async def get_batch_predictions(
#     request: Request, # Passed by SlowAPI middleware
#     batch_req: BatchPredictionRequest = Body(
#         ...,
#         example=BatchPredictionRequest(
#             requests=[
#                 PredictionRequest(
#                     county="19153", 
#                     year=2023, 
#                     cut_off_date="2023-08-01", 
#                     crop="corn", 
#                     histogram_bins=[0, 50, 100, 150, 200, 250]
#                 ),
#                 PredictionRequest(
#                     county="17031", 
#                     year=2022, 
#                     cut_off_date="2022-07-15", 
#                     crop="soybeans", 
#                     histogram_bins=[0, 20, 40, 60, 80]
#                 )
#             ]
#         ).model_dump()
#     ),
#     model_data=Depends(get_model_and_mapping), # model, fips_map, crop_map
# ):
#     model, fips_mapping, crop_mapping = model_data # Unpack

#     # Call the batch prediction function (which needs to be updated)
#     results = await predict_yield_batch(
#         model=model,
#         fips_mapping=fips_mapping,
#         crop_mapping=crop_mapping, # Pass crop_mapping
#         requests=batch_req.requests # Pass the list of PredictionRequest objects
#     )
#     return BatchPredictionResponse(responses=results)


# --- Model Info Endpoint ---
@app.get(
    "/model_info",
    response_model=ModelInfoResponse,
    tags=["Service Info"],
    summary="Get loaded model information",
    description="Returns details about the ML model currently loaded by the service.",
    response_description="Information about the loaded model.",
)
async def model_info():
    app_settings = await get_app_settings() # Ensures settings are loaded

    # No need to check if app_settings is None, get_app_settings will raise if load fails

    logger.info(
        "Returning model info based on AppSettings",
        model_name=app_settings.mlflow.model_name,
        model_stage=app_settings.mlflow.model_stage,
        mlflow_uri=str(app_settings.mlflow.tracking_uri),
    )
    return {
        "model_name": app_settings.mlflow.model_name,
        "model_stage": app_settings.mlflow.model_stage,
        "mlflow_uri": str(app_settings.mlflow.tracking_uri), # Ensure string conversion for HttpUrl
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
async def health_check(
    app_settings: AppSettings = Depends(get_app_settings), # Now AppSettings should be defined
    model_data = Depends(get_model_and_mapping), 
):
    # The get_app_settings dependency ensures settings are loaded.
    # The get_model_and_mapping dependency ensures an attempt to load the model was made.
    # Model loading success/failure is implicitly checked by get_model_and_mapping not raising an unhandled error here.
    # Actual check for feature service:
    if app_settings.feature_service and app_settings.feature_service.url:
        feature_service_ok = False
        feature_service_base_url = str(app_settings.feature_service.url).rstrip("/")
        try:
            feature_service_health_url = f"{feature_service_base_url}/health"
            start_time = time.time()
            response = await async_client.get(feature_service_health_url, timeout=app_settings.feature_service.timeout_seconds)
            elapsed_time = time.time() - start_time
            FEATURE_SERVICE_LATENCY.observe(elapsed_time)

            if response.status_code == 200:
                feature_service_ok = True
                logger.debug(
                    "Health check: Feature service reachable and healthy",
                    url=feature_service_health_url,
                    status_code=response.status_code,
                )
            else:
                logger.warning(
                    "Health check: Feature service reachable but unhealthy or error status",
                    url=feature_service_health_url,
                    status_code=response.status_code,
                    response_body=response.text,
                )
        except httpx.RequestError as e:
            logger.error(
                "Health check failed: Could not connect to Feature Service health endpoint",
                error=str(e),
                url=f"{feature_service_base_url}/health",
            )
        except Exception as e:
            logger.exception(
                "Health check failed: Unexpected error during Feature Service health check"
            )

        if not feature_service_ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Dependency failed: Cannot reach Feature Service or it's unhealthy",
            )
    else:
        logger.warning(
            "Health check: Feature Service URL is None or not configured, skipping connectivity check."
        )

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

# --- Startup event to pre-load model and settings ---
@app.on_event("startup")
async def startup_event_tasks():
    logger.info("Application startup: Initializing settings and attempting to pre-load model...")
    try:
        app_settings = await get_app_settings()
        app.state.app_settings = app_settings # STORE SETTINGS IN APP.STATE
        logger.info("Application settings initialized and stored in app.state.")
        
        # Use rate limit values from settings for the Limiter instance
        # Accessing limiter from app.state.limiter
        # This is a bit tricky as limiter defaults are usually static. 
        # The lambda in @limiter.limit directly uses settings, which is better.
        # For default_limits on the Limiter object itself, it would need re-init or modification here.
        # For now, we rely on per-endpoint lambdas correctly using get_app_settings.

        # Attempt to pre-load model and mappings
        await get_model_and_mapping() 
        logger.info("Model and mappings pre-loaded successfully (or attempt was made).")
    except ConfigurationError as e:
        logger.error(f"Startup initialization or model pre-loading failed: {e}. Service will run but some functionalities may be impaired.")
    except Exception as e:
        logger.error(f"Unexpected error during startup tasks: {e}", exc_info=True)