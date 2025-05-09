import mlflow.pytorch  # Use mlflow.pytorch instead of pyfunc
import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl, Field
from typing import List
from .exceptions import ConfigurationError, ModelServingBaseError
import json  # Import json for reading the mapping artifact
import os  # Import os for handling artifact path

# Get structlog logger instance
logger = structlog.get_logger(__name__)


# (Keep Settings classes as they are)
class MlflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")
    tracking_uri: str
    model_name: str = Field(
        "AgriYieldPredictor", description="Name of the model in MLflow."
    )
    model_stage: str = Field("Production", description="Stage of the model to load.")


class FeatureServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEATURE_SERVICE_")
    url: HttpUrl
    timeout_seconds: float = Field(
        10.0, description="Timeout for requests to the feature service."
    )
    retry_attempts: int = Field(
        3, description="Number of retry attempts for feature service calls."
    )
    retry_min_wait_seconds: int = Field(
        1, description="Minimum wait time (seconds) between retries."
    )
    retry_max_wait_seconds: int = Field(
        5, description="Maximum wait time (seconds) between retries."
    )


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")
    predict_limit: str = Field(
        "10/minute", description="Rate limit for the /predict endpoint."
    )
    predict_batch_limit: str = Field(
        "5/minute", description="Rate limit for the /predict_batch endpoint."
    )


class Settings(BaseSettings):
    mlflow: MlflowSettings = MlflowSettings()
    feature_service: FeatureServiceSettings = FeatureServiceSettings()
    api: ApiSettings = ApiSettings()


# Load settings instance
try:
    settings = Settings()
    # Set MLflow tracking URI - needed for loading the model from the registry
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    logger.info(
        "Settings loaded and MLflow tracking URI set",
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        feature_service_url=str(settings.feature_service.url),
        feature_service_timeout=settings.feature_service.timeout_seconds,
        feature_service_retries=settings.feature_service.retry_attempts,
        api_predict_limit=settings.api.predict_limit,
        api_batch_limit=settings.api.predict_batch_limit,
    )
except Exception as e:
    logger.error(
        "Failed to load settings or set MLflow tracking URI",
        error=str(e),
        exc_info=True,
    )
    settings = None


# --- Model Loading Function ---
# This function will now return the PyTorch model and the fips_mapping
def load_ml_model():
    """
    Loads the PyTorch ML model and associated artifacts from MLflow based on configuration settings.
    Returns a tuple: (pytorch_model, fips_mapping_dict).
    Raises ConfigurationError if settings are missing, or ModelServingBaseError if loading fails.
    """
    if not settings:
        logger.error("Cannot load model: Settings are not available.")
        raise ConfigurationError("MLflow settings are not configured.")

    model_uri = f"models:/{settings.mlflow.model_name}/{settings.mlflow.model_stage}"
    fips_mapping_artifact_path = (
        "fips_mapping/fips_mapping.json"  # Path *within* the MLflow run artifact
    )

    logger.info(
        "Attempting to load model and artifacts from MLflow", model_uri=model_uri
    )

    try:
        # Load the PyTorch model
        pytorch_model = mlflow.pytorch.load_model(model_uri)
        logger.info("PyTorch model loaded successfully.")

        # Load the FIPS mapping artifact
        # MLflow provides a way to download artifacts associated with the model version
        # Find the model version associated with the stage
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(
            settings.mlflow.model_name, stages=[settings.mlflow.model_stage]
        )[0]
        run_id = latest_version.run_id

        # Download the artifact
        local_path = client.download_artifacts(run_id, fips_mapping_artifact_path)
        logger.info("FIPS mapping artifact downloaded", local_path=local_path)

        # Read the JSON mapping
        with open(local_path, "r") as f:
            fips_mapping = json.load(f)
        logger.info("FIPS mapping loaded successfully.", num_entries=len(fips_mapping))

        # Clean up the downloaded artifact file (optional, but good practice)
        # os.remove(local_path) # Careful with this if local_path is a directory! artifact_path="fips_mapping" logs a directory.
        # A safer way is to download to a temp dir and clean up the dir.
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir_path = client.download_artifacts(
                run_id, "fips_mapping", dst_path=tmpdir
            )
            mapping_file_path = os.path.join(local_dir_path, "fips_mapping.json")
            with open(mapping_file_path, "r") as f:
                fips_mapping = json.load(f)
            logger.info("FIPS mapping loaded from temporary directory.")

        return pytorch_model, fips_mapping

    except Exception as e:
        logger.error(
            "Error loading model or artifacts from MLflow", error=str(e), exc_info=True
        )
        # Wrap in custom exception for consistent handling
        raise ModelServingBaseError(
            f"Failed to load model or artifacts from MLflow: {e}"
        ) from e


# --- Load model instance and mapping at startup ---
try:
    loaded_model_instance, loaded_fips_mapping = load_ml_model()
    logger.info("Initial model and FIPS mapping load successful.")
except (ConfigurationError, ModelServingBaseError) as e:
    logger.critical(
        "CRITICAL: Initial model load failed during startup.",
        error=str(e),
        exc_info=True,
    )
    loaded_model_instance = None
    loaded_fips_mapping = None


# --- Dependency Function ---
async def get_model_and_mapping():
    """FastAPI dependency function to provide the loaded model instance and FIPS mapping."""
    if loaded_model_instance is None or loaded_fips_mapping is None:
        logger.error(
            "Model or FIPS mapping requested but not available (initial load failed)."
        )
        raise ModelServingBaseError(
            "Model or FIPS mapping is not available due to loading error."
        )
    return loaded_model_instance, loaded_fips_mapping  # Return both


# Update the `get_model` dependency in app.py and predict.py to `get_model_and_mapping`
# and adjust where it's used to unpack the tuple.z
