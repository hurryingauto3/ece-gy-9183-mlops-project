import logging
import sys
import structlog
import asyncio # Import asyncio for startup/shutdown
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
# Import run_in_threadpool to handle synchronous I/O in an async endpoint
from starlette.concurrency import run_in_threadpool

from .config import settings
# Import the data loading function and connection management functions
from .data_loader import (
    load_and_process_weather_features,
    initialize_swift_connection,
    close_swift_connection
)
from .models import FeaturesResponse
# Import OpenStack SDK exceptions if needed for specific handling
from openstack.exceptions import ResourceNotFound, SDKException # Import SDKException


# --- Configure Structured Logging (Keep as is) ---
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
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
logger = structlog.get_logger(__name__)
# ------------------------------------


app = FastAPI(
    title="AgriYield Feature Service",
    description="Provides weather features for yield prediction.",
    version="1.0.0"
)

# --- Startup and Shutdown Events for Swift Connection ---
@app.on_event("startup")
async def startup_event():
    """Initializes the OpenStack Swift connection on application startup."""
    logger.info("Application startup: Initializing Swift connection...")
    try:
        initialize_swift_connection()
        logger.info("Swift connection initialized successfully.")
    except Exception as e:
        logger.critical("Failed to initialize Swift connection on startup.", error=str(e), exc_info=True)
        # Depending on desired behavior, you might want to exit or mark health check as failed
        # For critical dependency failure, exiting is often appropriate in production
        # import os
        # os._exit(1) # Force exit if connection is essential
        # For now, we'll rely on the health check and endpoints to handle the failure.


@app.on_event("shutdown")
async def shutdown_event():
    """Closes the OpenStack Swift connection on application shutdown."""
    logger.info("Application shutdown: Closing Swift connection...")
    close_swift_connection()
    logger.info("Swift connection closing process finished.")
# --------------------------------------------------------


# --- Exception Handlers (Update to include OpenStack exceptions) ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for FastAPI's HTTPException."""
    if exc.status_code != 404:
         logger.error("HTTP Exception caught by handler", status_code=exc.status_code, detail=exc.detail, path=request.url.path)
    else:
         logger.warning("HTTP Exception caught by handler (404 Not Found)", status_code=exc.status_code, detail=exc.detail, path=request.url.path)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Custom handler for Pydantic validation errors."""
    logger.warning("Request validation failed", detail=exc.errors(), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

# Add handlers for specific OpenStack exceptions if you want different status codes
@app.exception_handler(ResourceNotFound) # This might be redundant if handled in load_and_process
async def openstack_resource_not_found_handler(request, exc):
     logger.warning("OpenStack Resource Not Found during feature fetch", error=str(exc), path=request.url.path)
     return JSONResponse(
         status_code=status.HTTP_404_NOT_FOUND,
         content={"detail": f"Data resource not found in cloud storage: {exc}"},
     )

@app.exception_handler(SDKException) # Generic SDK errors (authentication, network, etc.)
async def openstack_sdk_exception_handler(request, exc):
     logger.error("OpenStack SDK Exception during feature fetch", error=str(exc), path=request.url.path, exc_info=True)
     return JSONResponse(
         status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Service Unavailable for dependency errors
         content={"detail": f"Cloud storage service error: {exc}"},
     )

@app.exception_handler(ConnectionError) # Handle custom ConnectionError from data_loader
async def swift_connection_error_handler(request, exc):
    logger.error("Swift Connection Error during feature fetch", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Swift connection error: {exc}"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch unhandled exceptions and log them."""
    logger.exception("Unhandled exception occurred during request processing", path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )
# --------------------------


# --- API Endpoint (Update to use run_in_threadpool) ---
@app.get(
    "/features",
    response_model=FeaturesResponse,
    summary="Get weather features for a county and year",
    description="Retrieves daily weather data for the April-October season for the specified county (FIPS) and year from OpenStack Swift."
)
async def get_features(county: str, year: int):
    """
    Fetches weather data from OpenStack Swift for the April-October season.
    """
    structlog.contextvars.bind_contextvars(fips_code=county, year=year)
    logger.info("Received request for features")

    try:
        # --- Use run_in_threadpool for the blocking I/O call ---
        # load_and_process_weather_features is synchronous, so run it in a thread
        weather_data = await run_in_threadpool(load_and_process_weather_features, county, year)
        # -----------------------------------------------------

        if weather_data is None:
            # load_and_process_weather_features returns None if the object wasn't found
            logger.warning("Features data object not found in Swift")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Features not found for FIPS {county}, year {year}")

        # If weather_data is an empty list (file found, but no valid data)
        if not weather_data:
            logger.warning("Features object found but contains no valid season data after processing")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No valid weather data found in season for FIPS {county}, year {year}")

        logger.info("Features successfully loaded and processed from Swift", data_points_count=len(weather_data))
        return FeaturesResponse(fips_code=county, year=year, weather_data=weather_data)

    # Note: Specific exceptions like ResourceNotFound, SDKException, ConnectionError,
    # and generic Exception raised by load_and_process_weather_features will be
    # caught by the exception handlers defined above.
    except HTTPException:
         # Re-raise HTTPExceptions that we explicitly raise (e.g., 404)
         raise
    except Exception as e:
        # Catch any unexpected exceptions *after* the threadpool call finishes
        # The specific handlers above catch most expected errors.
        # This catch-all is a fallback before the global handler.
        logger.error("Unexpected error during feature request processing", error=str(e), exc_info=True)
        raise # Let the global handler process it

    finally:
        structlog.contextvars.clear_contextvars()


# --- Health Check Endpoint ---
@app.get("/health", summary="Feature service health check")
async def health_check():
    """Checks service health, including Swift connection status."""
    logger.debug("Health check received")

    # Check if the Swift connection object exists
    from .data_loader import swift_conn # Import the global variable

    if swift_conn is None:
        logger.error("Health check failed: Swift connection is not initialized or is broken.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Swift connection is not available.")

    # Optional: Add a quick test using the connection, e.g., list containers (might be slow)
    # A simpler approach is to trust that if `swift_conn` is not None, it was initialized
    # successfully and will likely work for simple downloads, relying on retry logic
    # and endpoint error handling for transient issues.

    # You could try a lightweight operation like listing containers or getting container metadata
    # try:
    #      swift_conn.object_store.get_container(settings.openstack.swift_container_name)
    #      logger.debug("Health check: Swift container access confirmed.")
    # except Exception as e:
    #      logger.error("Health check failed: Could not access Swift container.", error=str(e))
    #      raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not access Swift container: {e}")


    logger.debug("Health check successful")
    return {"status": "ok"}
# -----------------------------