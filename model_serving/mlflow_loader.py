import mlflow.pyfunc
import structlog # Import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl, Field # Import HttpUrl and Field for validation
from typing import List # Import List
from model_serving.exceptions import ConfigurationError, ModelServingBaseError # Import exceptions

# Get structlog logger instance
logger = structlog.get_logger(__name__)

class MlflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='MLFLOW_') # Reads MLFLOW_TRACKING_URI etc.
    tracking_uri: str
    model_name: str = Field("AgriYieldPredictor", description="Name of the model in MLflow.") # Default value with description
    model_stage: str = Field("Production", description="Stage of the model to load.")       # Default value with description

class FeatureServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='FEATURE_SERVICE_')
    url: HttpUrl # Use HttpUrl for validation, reads FEATURE_SERVICE_URL
    timeout_seconds: float = Field(10.0, description="Timeout for requests to the feature service.") # Add timeout configuration
    retry_attempts: int = Field(3, description="Number of retry attempts for feature service calls.") # Add retry configuration
    retry_min_wait_seconds: int = Field(1, description="Minimum wait time (seconds) between retries.") # Add retry configuration
    retry_max_wait_seconds: int = Field(5, description="Maximum wait time (seconds) between retries.") # Add retry configuration

class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='API_')
    predict_limit: str = Field("10/minute", description="Rate limit for the /predict endpoint.") # Add rate limit configuration
    predict_batch_limit: str = Field("5/minute", description="Rate limit for the /predict_batch endpoint.") # Add rate limit configuration

class Settings(BaseSettings):
    mlflow: MlflowSettings = MlflowSettings()
    feature_service: FeatureServiceSettings = FeatureServiceSettings()
    api: ApiSettings = ApiSettings() # Add API settings
    # Add other settings sections if needed, e.g., database, external APIs

# Load settings instance - reads from environment variables
try:
    settings = Settings()
    # --- Comment out MLflow URI setting ---
    # mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    # --- Log that it's skipped ---
    logger.info("Skipping MLflow tracking URI setup for testing.")
    # ------------------------------------
    logger.info(
        "Settings loaded",
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        feature_service_url=str(settings.feature_service.url), # Convert HttpUrl for logging
        feature_service_timeout=settings.feature_service.timeout_seconds,
        feature_service_retries=settings.feature_service.retry_attempts,
        api_predict_limit=settings.api.predict_limit,
        api_batch_limit=settings.api.predict_batch_limit,
    )
except Exception as e:
    # Use structlog, log exception info
    logger.error("Failed to load settings or set MLflow tracking URI", error=str(e), exc_info=True)
    settings = None # Indicate settings failed to load

# --- Model Loading Function ---
def load_ml_model():
    """
    Loads the ML model from MLflow based on configuration settings.
    Raises ConfigurationError if settings are missing, or ModelServingBaseError if loading fails.
    """
    if not settings:
        logger.error("Cannot load model: Settings are not available.")
        raise ConfigurationError("MLflow settings are not configured.")

    try:
        uri = f"models:/{settings.mlflow.model_name}/{settings.mlflow.model_stage}"
        logger.info("Attempting to load model from MLflow (call commented out)", model_uri=uri)
        # --- Comment out actual MLflow call ---
        # model = mlflow.pyfunc.load_model(uri)
        # --- Return None instead ---
        
        # ---------------------------
        # if model is None: # This check is now redundant
        #      raise ModelServingBaseError("mlflow.pyfunc.load_model returned None")

        model = None
        logger.warning("MLflow model loading skipped for testing, returning None.")
        return model
    except Exception as e:
        # This block might still catch errors if uri creation fails, etc.
        logger.error("Error during model loading preparation", error=str(e), exc_info=True)
        raise ModelServingBaseError(f"Failed during model loading preparation: {e}") from e

# --- Load model instance at startup ---
# We still load it once, but store it for the dependency function
# Handle potential errors during initial load
try:
    loaded_model_instance = load_ml_model()
    # logger.info("Initial model load successful.") # Logged inside load_ml_model now if successful
except (ConfigurationError, ModelServingBaseError) as e:
    logger.critical("CRITICAL: Initial model load failed during startup.", error=str(e), exc_info=True)
    # Depending on desired behavior, could exit here or let endpoints fail
    loaded_model_instance = None
    # Consider exiting: import sys; sys.exit(1)

# --- Dependency Function ---
async def get_model():
    """FastAPI dependency function to provide the loaded model instance."""
    if loaded_model_instance is None:
        # This case should ideally not happen if startup check is robust,
        # but handles it defensively.
        logger.error("Model requested but not available (initial load failed).")
        raise ModelServingBaseError("Model is not available due to loading error.")
    return loaded_model_instance