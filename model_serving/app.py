import logging  # Keep standard logging import for configuration
import time
import asyncio
import sys  # Import sys for stdout configuration
import structlog  # Import structlog
from fastapi import FastAPI, HTTPException, Request, status, Body, Depends  # Import Body for examples and Depends for dependency injection
from fastapi.responses import JSONResponse  # Import JSONResponse for exception handler
from pydantic import BaseModel, field_validator, HttpUrl, Field
from typing import List, Dict, Any, Optional
from model_serving.predict import predict_yield, predict_yield_batch, async_client  # Import the httpx client instance
from model_serving.mlflow_loader import get_model, settings, ModelServingBaseError, ConfigurationError  # Import settings and exceptions
from model_serving.drift import check_data_drift, log_prediction_for_monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import httpx  # Import httpx for exceptions
from model_serving.metrics import (
    BATCH_PREDICTION_OUTCOMES,
    PREDICTED_YIELD_DISTRIBUTION
)
from model_serving.exceptions import (
    FeatureNotFoundError, FeatureServiceError,
    InvalidModelInputError, InvalidModelOutputError, ModelInferenceError,
)

# --- Configure Structured Logging ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,  # Merge context variables
        structlog.stdlib.add_logger_name,  # Add logger name (e.g., 'model_serving.app')
        structlog.stdlib.add_log_level,  # Add log level (e.g., 'info', 'warning')
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Prepare for standard logging formatter
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Define the formatter for structlog messages, rendering them as JSON
formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),  # Use JSONRenderer
)

# Configure standard logging handler to use the structlog formatter
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
root_logger = logging.getLogger()

# Remove existing handlers to avoid duplicate logs if any were configured before
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)  # Set root logger level

# Get structlog logger instance for this module
logger = structlog.get_logger(__name__)
# ------------------------------------

# --- Configure Rate Limiting using Settings ---
# Use a default if settings failed to load, or handle error appropriately
default_limit = "10/minute"
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[default_limit],
    enabled=settings is not None  # Disable limiter if settings failed
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
    version="1.0.0",  # Add API version
    openapi_tags=tags_metadata
)

# --- Add Rate Limiting State and Middleware/Handler ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
# ----------------------------------------------------

# --- Add Prometheus Metrics ---
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
        request_id=request.headers.get("X-Request-ID", "N/A")  # Example: Use request ID header if present
    )
    logger.info("Received request")  # Log start of request

    response = await call_next(request)

    process_time = time.time() - start_time
    # Log end of request with status and duration
    logger.info(
        "Request finished",
        status_code=response.status_code,
        process_time_seconds=round(process_time, 4)
    )
    structlog.contextvars.clear_contextvars()  # Clear context for next request
    return response

# --- Pydantic Models with Examples ---
class PredictionRequest(BaseModel):
    county: str = Field(..., example="TestCounty", description="Name of the county.")
    year: int = Field(..., example=2023, description="Year for the prediction (e.g., 1980-2050).")

    @field_validator('year')
    def year_must_be_reasonable(cls, v):
        if not (1980 <= v <= 2050):  # Example range, adjust as needed
            raise ValueError('Year must be between 1980 and 2050')
        return v

    @field_validator('county')
    def county_must_not_be_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('County cannot be empty')
        # Add more specific county validation if a list of valid counties exists
        return v.strip()

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., min_length=1, description="A list of prediction requests.")

class BatchPredictionResponseItem(BaseModel):
    county: str = Field(..., example="TestCounty", description="Name of the county from the request.")
    year: int = Field(..., example=2023, description="Year from the request.")
    predicted_yield: Optional[float] = Field(None, example=45.67, description="Predicted yield value, if successful.")
    error: Optional[str] = Field(None, example="Features not found", description="Error message, if prediction failed for this item.")

class BatchPredictionResponse(BaseModel):
    responses: List[BatchPredictionResponseItem] = Field(..., description="List of results corresponding to the batch requests.")

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="AgriYieldPredictor")
    model_stage: str = Field(..., example="Production")
    mlflow_uri: str = Field(..., example="http://mlflow.example.com")

# --- API Endpoints with Tags and Descriptions ---

@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Get a single yield prediction",
    description="Fetch features and predict yield for a single county and year.",
    response_description="The prediction result or error information."
)
@limiter.limit(lambda: settings.api.predict_limit if settings else default_limit)
async def get_prediction(
    request: Request,
    req: PredictionRequest = Body(..., example={"county": "SampleCounty", "year": 2024}),
    model = Depends(get_model)  # Inject model dependency
):
    structlog.contextvars.bind_contextvars(county=req.county, year=req.year)
    prediction_result = None
    error_message = None
    try:
        check_data_drift(req.model_dump())
        logger.info("Starting prediction")
        prediction_result = await predict_yield(model, req.county, req.year)
        logger.info("Prediction successful", predicted_yield=prediction_result)

        if prediction_result is not None:
            PREDICTED_YIELD_DISTRIBUTION.observe(prediction_result)

        log_prediction_for_monitoring(
            request_data=req.model_dump(),
            prediction=prediction_result,
            error=None
        )
        return {"county": req.county, "year": req.year, "predicted_yield": prediction_result}
    except ModelServingBaseError as e:
        error_message = f"Model error: {e}"
        logger.error("Model serving error during prediction", error=str(e), exc_info=True)
        raise e
    except FeatureNotFoundError as e:
        error_message = str(e)
        logger.warning("Data not found for prediction", error=error_message)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_message)
    except (FeatureServiceError, ConfigurationError) as e:
        error_message = f"Service dependency failed: {e}"
        logger.error("Service dependency or configuration error during prediction", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except (InvalidModelInputError, InvalidModelOutputError) as e:
        error_message = f"Invalid data or model output: {e}"
        logger.warning("Invalid data or model output during prediction", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except Exception as e:
        error_message = "An internal server error occurred during prediction."
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message)
    finally:
        if error_message is not None:
            log_prediction_for_monitoring(
                request_data=req.model_dump(),
                prediction=None,
                error=error_message
            )
        structlog.contextvars.unbind_contextvars("county", "year")


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,  # Explicit response model
    tags=["Predictions"],
    summary="Get multiple yield predictions in batch",
    description="Fetch features concurrently and predict yield for multiple county/year pairs.",
    response_description="A list of prediction results or errors for each item in the batch request."
)
@limiter.limit(lambda: settings.api.predict_batch_limit if settings else default_limit)
async def get_batch_predictions(
    request: Request,
    batch_req: BatchPredictionRequest = Body(..., example={
        "requests": [
            {"county": "CountyA", "year": 2022},
            {"county": "CountyB", "year": 2023}
        ]
    }),
    model = Depends(get_model)  # Inject model dependency
):
    request_count = len(batch_req.requests)
    structlog.contextvars.bind_contextvars(batch_size=request_count)
    logger.info("Received batch prediction request")
    try:
        request_dicts = [req.model_dump() for req in batch_req.requests]
        if request_dicts:
            check_data_drift(request_dicts[0])
        results = await predict_yield_batch(model, request_dicts)

        success_count = 0
        failure_count = 0
        for item_result in results:
            predicted_yield = item_result.get("predicted_yield")
            if item_result.get("error") is None and predicted_yield is not None:
                BATCH_PREDICTION_OUTCOMES.labels(outcome='success').inc()
                PREDICTED_YIELD_DISTRIBUTION.observe(predicted_yield)
                success_count += 1
            else:
                BATCH_PREDICTION_OUTCOMES.labels(outcome='error').inc()
                failure_count += 1

        logger.info("Batch results summary", success_count=success_count, failure_count=failure_count)
        return BatchPredictionResponse(responses=results)
    except ModelServingBaseError as e:
        logger.error("Model serving error during batch prediction", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.exception("Unexpected error during batch prediction setup or finalization")
        error_response = [{"county": r.county, "year": r.year, "error": "An internal error occurred during batch processing."} for r in batch_req.requests]
        return BatchPredictionResponse(responses=error_response)
    finally:
        structlog.contextvars.unbind_contextvars("batch_size")


@app.get(
    "/model_info",
    response_model=ModelInfoResponse,  # Explicit response model
    tags=["Service Info"],
    summary="Get loaded model information",
    description="Returns details about the ML model currently loaded by the service.",
    response_description="Information about the loaded model."
)
def model_info():
    # --- Return dummy data as MLflow connection is bypassed ---
    logger.info("Returning dummy model info as MLflow connection is bypassed.")
    return {
        "model_name": "N/A (MLflow bypassed)",
        "model_stage": "N/A (MLflow bypassed)",
        "mlflow_uri": settings.mlflow.tracking_uri if settings and settings.mlflow else "N/A (MLflow bypassed)"
    }
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Original logic:
    # if not settings:
    #      raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Configuration not loaded.")
    # return {
    #     "model_name": settings.mlflow.model_name,
    #     "model_stage": settings.mlflow.model_stage,
    #     "mlflow_uri": settings.mlflow.tracking_uri
    # }


@app.get(
    "/health",
    response_model=HealthResponse,  # Explicit response model
    tags=["Service Info"],
    summary="Perform a health check",
    description="Checks if the service is running and essential dependencies (model, feature service) are available.",
    response_description="Service status."
)
async def health_check(model = Depends(get_model)):  # Inject model dependency
    if settings and settings.feature_service.url:
        feature_service_ok = False
        base_url = "invalid_url_initially"
        try:
            base_url = str(settings.feature_service.url).rstrip('/') + "/"
            start_time = time.time()
            response = await async_client.head(base_url, timeout=5.0)
            elapsed_time = time.time() - start_time
            if response.status_code < 500:
                feature_service_ok = True
                if response.status_code >= 400:
                    logger.warning("Health check: Feature service reachable but returned client error status", status_code=response.status_code, url=base_url)
            else:
                logger.error("Health check failed: Feature service returned server error status", status_code=response.status_code, url=base_url)
        except httpx.RequestError as e:
            logger.error("Health check failed: Could not connect to Feature Service", error=str(e), url=base_url)
        except Exception as e:
            logger.exception("Health check failed: Unexpected error during Feature Service check")
        if not feature_service_ok:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Dependency failed: Cannot reach Feature Service")
    else:
        logger.warning("Health check: Feature Service URL not configured, skipping connectivity check.")

    logger.debug("Health check successful")
    return {"status": "ok"}

@app.exception_handler(ModelServingBaseError)
async def model_serving_exception_handler(request: Request, exc: ModelServingBaseError):
    logger.error(f"ModelServingBaseError caught by handler: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Service dependency failed: {exc}"},
    )

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing httpx client...")
    await async_client.aclose()
    logger.info("Httpx client closed.")