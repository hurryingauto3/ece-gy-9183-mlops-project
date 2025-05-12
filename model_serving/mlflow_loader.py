import mlflow.pytorch  # Use mlflow.pytorch instead of pyfunc
import structlog
import torch # Import torch for type hinting if model is PyTorch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl, Field
from typing import List, Optional, Tuple, Dict, Any
from .exceptions import ConfigurationError, ModelServingBaseError
import json  # Import json for reading the mapping artifact
import os  # Import os for handling artifact path
import asyncio
from mlflow.exceptions import MlflowException

# Get structlog logger instance
logger = structlog.get_logger(__name__)


# --- Pydantic Settings Models (assuming these are defined as before) ---
class MLflowSettings(BaseSettings):
    tracking_uri: HttpUrl = Field(..., description="MLflow tracking server URI.")
    model_name: str = Field(..., description="Name of the model in MLflow registry.")
    model_stage: str = Field("Production", description="Stage of the model to load.")
    fips_mapping_artifact_name: str = Field("fips_to_id_mapping.json", description="Name of FIPS mapping artifact.")
    crop_mapping_artifact_name: str = Field("crop_to_id_mapping.json", description="Name of Crop mapping artifact.")
    # Dummy mode settings
    dummy_mode: bool = Field(False, description="Enable dummy model mode. Overrides MLflow if paths are set.")
    dummy_model_path: Optional[str] = Field(None, description="LOCAL path to dummy PyTorch model state_dict (.pth).")
    dummy_model_num_bins: int = Field(5, description="Number of bins for the dummy model if instantiated directly.") # Used if dummy_model_path is not found
    dummy_fips_mapping_path: Optional[str] = Field(None, description="LOCAL path to dummy FIPS mapping JSON.")
    dummy_crop_mapping_path: Optional[str] = Field(None, description="LOCAL path to dummy Crop mapping JSON.")

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

class FeatureServiceSettings(BaseSettings):
    url: HttpUrl = Field(..., description="URL of the feature service API.")
    timeout_seconds: float = Field(10.0, description="Timeout for feature service requests.")
    retry_attempts: int = Field(3, description="Number of retry attempts for feature service.")
    retry_min_wait_seconds: int = Field(1, description="Minimum wait seconds between retries.")
    retry_max_wait_seconds: int = Field(5, description="Maximum wait seconds between retries.")
    model_config = SettingsConfigDict(env_prefix="FEATURE_SERVICE_", extra="ignore")

class APISettings(BaseSettings):
    predict_limit: str = Field("10/minute", description="Rate limit for single predictions.")
    predict_batch_limit: str = Field("5/minute", description="Rate limit for batch predictions.")
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

class AppSettings(BaseSettings):
    mlflow: MLflowSettings = MLflowSettings()
    feature_service: FeatureServiceSettings = FeatureServiceSettings()
    api: APISettings = APISettings()

# Global variable for settings, initialized asynchronously
settings: Optional[AppSettings] = None
_settings_lock = asyncio.Lock()

async def get_app_settings() -> AppSettings:
    """Initializes and returns the application settings. Ensures it happens once."""
    global settings
    if settings is None:
        async with _settings_lock:
            if settings is None: # Double check after lock
                try:
                    current_settings = AppSettings()
                    # Validate essential MLflow settings
                    if not current_settings.mlflow.tracking_uri or not current_settings.mlflow.model_name:
                        logger.critical("MLflow tracking URI or model name is not configured in settings.")
                        raise ConfigurationError("MLflow tracking URI or model name missing in AppSettings.")
                    settings = current_settings # Assign to global only if successful
                    logger.info("Application settings initialized successfully.")
                except Exception as e: # Catches Pydantic validation errors or other init issues
                    logger.critical("Failed to initialize AppSettings.", error=str(e), exc_info=True)
                    # settings remains None
                    raise ConfigurationError(f"Failed to initialize AppSettings: {e}") from e
    if settings is None: # Should not happen if logic above is correct and no exception swallowed
        raise ConfigurationError("Application settings could not be loaded.")
    return settings

# Global cache for model and mappings
# Stores: (model, fips_county_mapping, unique_fips_ids, crop_name_to_id_mapping, unique_crop_ids)
# The unique_fips_ids and unique_crop_ids are used by the model during init
_cached_model_data: Optional[Tuple[Any, Dict[str, int], List[int], Dict[str, int], List[int]]] = None
_model_load_lock = asyncio.Lock()

# Helper function for dummy model loading (synchronous part)
def _try_load_dummy_model_and_mappings(mlflow_settings: MLflowSettings) -> Optional[Tuple[Any, Dict[str, int], List[int], Dict[str, int], List[int]]]:
    logger.warning("Attempting to load LOCAL DUMMY model and mappings.")
    try:
        fips_map = {"00000": 0, "DUMMY": 0} # Default dummy FIPS map
        if mlflow_settings.dummy_fips_mapping_path and os.path.exists(mlflow_settings.dummy_fips_mapping_path):
            with open(mlflow_settings.dummy_fips_mapping_path, "r") as f:
                fips_map = json.load(f)
            logger.info(f"Loaded dummy FIPS mapping from {mlflow_settings.dummy_fips_mapping_path}")
        else:
            logger.info(f"Using default hardcoded dummy FIPS mapping for DUMMY mode. Path not found or not set: {mlflow_settings.dummy_fips_mapping_path}")

        crop_map = {"corn": 0, "soybeans": 1, "dummy_crop": 2} # Default dummy crop map
        if mlflow_settings.dummy_crop_mapping_path and os.path.exists(mlflow_settings.dummy_crop_mapping_path):
            with open(mlflow_settings.dummy_crop_mapping_path, "r") as f:
                crop_map = json.load(f)
            logger.info(f"Loaded dummy CROP mapping from {mlflow_settings.dummy_crop_mapping_path}")
        else:
            logger.info(f"Using default hardcoded dummy CROP mapping for DUMMY mode. Path not found or not set: {mlflow_settings.dummy_crop_mapping_path}")
        
        dummy_model_instance = None
        from model_training.model import DummyLSTMTCNHistogramPredictor # Import here

        if mlflow_settings.dummy_model_path and os.path.exists(mlflow_settings.dummy_model_path):
            logger.info(f"Attempting to load dummy model state_dict from: {mlflow_settings.dummy_model_path}")
            dummy_model_instance = DummyLSTMTCNHistogramPredictor(
                input_dim=1, num_fips=len(fips_map), num_crops=len(crop_map), 
                num_bins=mlflow_settings.dummy_model_num_bins
            )
            dummy_model_instance.load_state_dict(torch.load(mlflow_settings.dummy_model_path, map_location=torch.device('cpu')))
            dummy_model_instance.eval()
            logger.info(f"Loaded dummy model from state_dict: {mlflow_settings.dummy_model_path}")
        else:
            logger.warning(f"Dummy model path not found or not specified ({mlflow_settings.dummy_model_path}). Instantiating a new default DummyLSTMTCNHistogramPredictor.")
            dummy_model_instance = DummyLSTMTCNHistogramPredictor(
                input_dim=1, num_fips=len(fips_map), 
                num_crops=len(crop_map), 
                num_bins=mlflow_settings.dummy_model_num_bins
            )
            dummy_model_instance.eval()

        unique_fips_ids = sorted(list(set(fips_map.values())))
        unique_crop_ids = sorted(list(set(crop_map.values())))
        logger.info("DUMMY MODE: Successfully loaded/instantiated dummy model and mappings.")
        return (dummy_model_instance, fips_map, unique_fips_ids, crop_map, unique_crop_ids)
    except Exception as e:
        logger.error(f"DUMMY MODE FAILED: Could not load local dummy model/mappings: {e}", exc_info=True)
        return None

async def _load_model_and_mappings_from_mlflow(app_settings: AppSettings):
    global _cached_model_data

    # Attempt DUMMY MODE first if explicitly enabled
    if app_settings.mlflow.dummy_mode:
        dummy_data = _try_load_dummy_model_and_mappings(app_settings.mlflow)
        if dummy_data:
            _cached_model_data = dummy_data
            return # Successfully loaded dummy data
        else:
            logger.error("Explicit dummy mode was enabled but failed. Will NOT attempt MLflow.")
            # If dummy_mode is True, we assume it's the only mode desired.
            # Raise config error so service indicates it couldn't load what was explicitly requested.
            raise ConfigurationError("Dummy mode enabled but failed to load dummy model/mappings.")

    # --- MLFLOW MODE (Original logic if not in explicit dummy_mode or if dummy_mode was false) ---
    logger.info(
        "Attempting to load model and mappings from MLflow...",
        model_name=app_settings.mlflow.model_name,
        model_stage=app_settings.mlflow.model_stage,
        mlflow_tracking_uri=str(app_settings.mlflow.tracking_uri), # Ensure str for set_tracking_uri
    )
    try:
        mlflow.set_tracking_uri(str(app_settings.mlflow.tracking_uri))
        model_uri = f"models:/{app_settings.mlflow.model_name}/{app_settings.mlflow.model_stage}"
        
        loaded_model = mlflow.pytorch.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow.")

        fips_mapping_artifact_uri = f"{model_uri.rstrip('/')}/{app_settings.mlflow.fips_mapping_artifact_name}"
        logger.debug(f"Attempting to download FIPS mapping from: {fips_mapping_artifact_uri}")
        fips_mapping_path = mlflow.artifacts.download_artifacts(
            artifact_uri=fips_mapping_artifact_uri
        )
        with open(fips_mapping_path, "r") as f:
            fips_county_to_id_map = json.load(f)
        logger.info("FIPS mapping loaded successfully.")
        unique_fips_ids = sorted(list(set(fips_county_to_id_map.values())))

        try:
            crop_mapping_artifact_uri = f"{model_uri.rstrip('/')}/{app_settings.mlflow.crop_mapping_artifact_name}"
            logger.debug(f"Attempting to download Crop mapping from: {crop_mapping_artifact_uri}")
            crop_mapping_path = mlflow.artifacts.download_artifacts(
                artifact_uri=crop_mapping_artifact_uri
            )
            with open(crop_mapping_path, "r") as f:
                crop_name_to_id_map = json.load(f)
            logger.info("Crop mapping loaded successfully from MLflow artifact.")
        except MlflowException as e:
            logger.warning(
                f"Failed to download crop mapping artifact '{app_settings.mlflow.crop_mapping_artifact_name}': {e}. Using placeholder.",
                exc_info=False # Keep log concise for this common case
            )
            crop_name_to_id_map = {"corn": 0, "soybeans": 1, "wheat": 2, "default_crop": 3} 
            logger.info("Using placeholder crop mapping.", mapping=crop_name_to_id_map)
        
        unique_crop_ids = sorted(list(set(crop_name_to_id_map.values())))

        if not isinstance(fips_county_to_id_map, dict):
            raise ConfigurationError("FIPS mapping is not a dictionary.")
        if not isinstance(crop_name_to_id_map, dict):
            raise ConfigurationError("Crop mapping is not a dictionary.")

        # The model itself is LSTMTCNHistogramPredictor which needs num_fips, num_crops, num_bins at __init__.
        # These should have been set correctly during training and saving the model.
        # We are just loading the saved model here.

        _cached_model_data = (loaded_model, fips_county_to_id_map, unique_fips_ids, crop_name_to_id_map, unique_crop_ids)
        logger.info("Model and all mappings are now cached.")

    except MlflowException as e:
        logger.critical("MLflow Core exception during model or artifact loading.", error=str(e), exc_info=True)
        logger.warning("MLflow loading failed. Attempting to load DUMMY model as FALLBACK.")
        dummy_data = _try_load_dummy_model_and_mappings(app_settings.mlflow) # Try dummy as fallback
        if dummy_data:
            _cached_model_data = dummy_data
            logger.info("Successfully loaded DUMMY model and mappings as a fallback to MLflow failure.")
            return
        else:
            logger.error("Fallback to DUMMY model also failed after MLflow exception.")
            _cached_model_data = None 
            raise ConfigurationError(f"MLflow core error: {e}. Fallback dummy model also failed or not configured.") from e
    except Exception as e:
        logger.critical("Unexpected error during model or mapping loading from MLflow.", error=str(e), exc_info=True)
        logger.warning("Unexpected MLflow error. Attempting to load DUMMY model as FALLBACK.")
        dummy_data = _try_load_dummy_model_and_mappings(app_settings.mlflow) # Try dummy as fallback
        if dummy_data:
            _cached_model_data = dummy_data
            logger.info("Successfully loaded DUMMY model and mappings as a fallback to unexpected MLflow error.")
            return
        else:
            logger.error("Fallback to DUMMY model also failed after unexpected MLflow error.")
            _cached_model_data = None
            raise ConfigurationError(f"Failed to load model/mappings via MLflow: {e}. Fallback dummy also failed.") from e


async def get_model_and_mapping() -> Tuple[Any, Dict[str, int], Dict[str, int]]:
    """
    Provides the cached ML model, FIPS mapping, and Crop mapping.
    Loads them from MLflow if not already cached. This function is used as a FastAPI dependency.
    Ensures AppSettings are loaded first.
    """
    app_settings = await get_app_settings() # Ensure settings are loaded

    if _cached_model_data is None:
        async with _model_load_lock:
            if _cached_model_data is None: # Double-check after acquiring the lock
                await _load_model_and_mappings_from_mlflow(app_settings)
            
    if _cached_model_data is None: # Should be caught by exceptions in _load_model_and_mappings_from_mlflow
        logger.critical("Model and mappings are still None after loading attempt.")
        raise ConfigurationError("Model and mappings could not be loaded and are unavailable.")

    model, fips_map, _, crop_map, _ = _cached_model_data
    return model, fips_map, crop_map

