# model_training/train_job.py
import os
import sys
import logging
import argparse
import json
import tempfile # For saving temporary files like fips_mapping
import structlog # Use structlog for structured logging
import datetime # For timestamping runs if needed

import torch
import openstack # For initializing Swift connection
from openstack.exceptions import SDKException, ConnectionException # Import specific exceptions

# Import configuration parsing
from pydantic import BaseModel, Field, HttpUrl # Use Pydantic for config validation
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Union # More flexible type hinting

# Import MLflow
import mlflow
import mlflow.pytorch
import mlflow.tracking # To get client for artifact download if needed

# Import components from your local package
# Ensure correct relative imports
from .swift_data_loader import SwiftCropYieldDataset, WEATHER_FEATURE_COLUMNS # Import dataset and expected columns
from ..model import LSTMTCNRegressor # Import model definition
from ..utils import get_dataloaders_by_year, train_model, evaluate_model, collate_fn # Import helper functions


# --- Configuration with Pydantic Settings ---
class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='MLFLOW_')
    tracking_uri: str
    model_name: str = Field("AgriYieldPredictor", description="Name of the model in MLflow.")
    model_stage: str = Field("Production", description="Stage for registering the model.") # Can change to 'Staging'
    experiment_name: str = Field("Crop Yield Training", description="Name of the MLflow experiment.")


class OpenStackSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OS_') # Reads OS_AUTH_URL etc.
    # Note: We don't need all OS_* vars defined here, openstack.connect() finds them
    # But define the ones we explicitly use or might override
    auth_url: HttpUrl # Pydantic validates URL format
    project_name: str
    project_domain_name: str = "Default"
    username: str
    user_domain_name: str = "Default"
    password: str
    region_name: Optional[str] = None # Optional
    swift_container_name: str = Field(..., description="Name of the OpenStack Swift container storing the data lake.")


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='TRAIN_') # Reads TRAIN_EPOCHS etc.
    num_epochs: int = Field(50, description="Number of training epochs.")
    learning_rate: float = Field(1e-3, description="Learning rate for the optimizer.")
    batch_size: int = Field(32, description="Batch size for dataloaders.")
    fips_embedding_dim: int = Field(16, description="Dimension of the FIPS embedding vector.")
    hidden_dim: int = Field(64, description="Hidden dimension of the LSTM.")
    lstm_layers: int = Field(1, description="Number of LSTM layers.")
    # Use a list of integers for tcn_channels
    tcn_channels: List[int] = Field([64, 32], description="List of output channels for TCN layers.")
    dropout_rate: float = Field(0.1, description="Dropout rate.")

    # Data Split Configuration
    # Allow None, single str year, or list of str years
    holdout_year: Optional[Union[str, List[str]]] = Field(None, description="Year(s) to hold out for testing (e.g., '2022' or '2022,2023').")
    val_year_ratio: float = Field(0.2, description="Proportion (0-1) or number (>1) of remaining years for validation.")

    # Other Training Parameters
    crop_name: str = Field("corn", description="Name of the crop to train for.")

    # Add a root directory setting if data needs to be cached locally before dataset init
    # local_data_cache_dir: Path = Field("/app/data_cache", description="Temporary local directory for data caching (inside container).")


class JobSettings(BaseSettings):
    """Overall settings for the training job."""
    mlflow: MLflowSettings = MLflowSettings()
    openstack: OpenStackSettings = OpenStackSettings()
    training: TrainingSettings = TrainingSettings()


# --- Logging Setup ---
# Configure standard logging to stdout, optionally with structlog for JSON output
def setup_logging():
    """Sets up structured logging to stdout."""
    # Configure structlog processors
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"), # Add timestamp
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter, # Prepare for standard logger
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Define the formatter (JSON output)
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(), # Output as JSON
    )

    # Set up the root logger to use the formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Get the root logger and remove any default handlers
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Set global logging level (can be INFO, DEBUG, etc.)
    # Could read this from an environment variable if desired
    root_logger.setLevel(logging.INFO)

    # Optionally silence chatty loggers
    logging.getLogger('openstack').setLevel(logging.WARNING)
    logging.getLogger('keystoneauth1').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    # Get a structlog logger for this module
    return structlog.get_logger(__name__)


# --- Main Execution Function ---
def main():
    logger = setup_logging() # Set up logging first

    logger.info("--- Model Training Job Start ---")
    start_time = datetime.datetime.now()
    job_status = "FAILED" # Default status

    settings = None
    swift_conn = None

    try:
        # 1. Load Configuration
        # Pydantic Settings automatically reads from environment variables
        settings = JobSettings()
        logger.info("Configuration loaded.")
        # Log config values (be careful with secrets!)
        logger.info("Job Configuration:",
                     mlflow_tracking_uri=settings.mlflow.tracking_uri,
                     mlflow_model_name=settings.mlflow.model_name,
                     mlflow_model_stage=settings.mlflow.model_stage,
                     mlflow_experiment_name=settings.mlflow.experiment_name,
                     swift_container_name=settings.openstack.swift_container_name,
                     train_epochs=settings.training.num_epochs,
                     train_lr=settings.training.learning_rate,
                     train_batch_size=settings.training.batch_size,
                     train_fips_embedding_dim=settings.training.fips_embedding_dim,
                     train_hidden_dim=settings.training.hidden_dim,
                     train_lstm_layers=settings.training.lstm_layers,
                     train_tcn_channels=settings.training.tcn_channels,
                     train_dropout_rate=settings.training.dropout_rate,
                     train_holdout_year=settings.training.holdout_year,
                     train_val_year_ratio=settings.training.val_year_ratio,
                     train_crop_name=settings.training.crop_name,
                     # Add other settings if needed
                     openstack_auth_url=settings.openstack.auth_url, # Example non-secret OS var
                     openstack_project_name=settings.openstack.project_name,
                     openstack_region_name=settings.openstack.region_name, # Might be None
                    )


        # 2. Initialize MLflow
        logger.info(f"MLflow tracking URI set to: {settings.mlflow.tracking_uri}")
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name) # Set experiment name

        # Start MLflow run
        # Pass config as parameters
        with mlflow.start_run() as active_run:
            logger.info(f"MLflow Run started: {active_run.info.run_id}")
            # Log parameters again within the run (redundant with start_run params, but explicit)
            # Or use mlflow.log_params(settings.training.model_dump()) etc.
            mlflow.log_param("crop_name", settings.training.crop_name)
            mlflow.log_param("num_epochs", settings.training.num_epochs)
            mlflow.log_param("learning_rate", settings.training.learning_rate)
            mlflow.log_param("batch_size", settings.training.batch_size)
            mlflow.log_param("fips_embedding_dim", settings.training.fips_embedding_dim)
            mlflow.log_param("hidden_dim", settings.training.hidden_dim)
            mlflow.log_param("lstm_layers", settings.training.lstm_layers)
            mlflow.log_param("tcn_channels", str(settings.training.tcn_channels)) # Log list as string
            mlflow.log_param("dropout_rate", settings.training.dropout_rate)
            mlflow.log_param("holdout_year_config", str(settings.training.holdout_year)) # Log original config
            mlflow.log_param("val_year_ratio_config", settings.training.val_year_ratio)
            mlflow.log_param("swift_container_name", settings.openstack.swift_container_name)
            mlflow.log_param("weather_feature_columns_count", len(WEATHER_FEATURE_COLUMNS))


            # 3. Initialize OpenStack Swift Connection
            try:
                # openstack.connect() reads OS_* env vars automatically from process environment
                logger.info("Connecting to OpenStack Swift using environment variables...")
                swift_conn = openstack.connect()
                swift_conn.authorize() # Ensure connection is authorized
                logger.info("OpenStack connection successful.")

                # Optional: Verify container exists
                try:
                     swift_conn.object_store.get_container(settings.openstack.swift_container_name)
                     logger.info(f"Swift container '{settings.openstack.swift_container_name}' found.")
                except openstack.exceptions.NotFoundException:
                     logger.critical(f"Swift container '{settings.openstack.swift_container_name}' not found. ETL job may not have run or used a different name.")
                     # End the MLflow run as it failed before data loading
                     mlflow.end_run("FAILED")
                     sys.exit(1)
                except Exception as e:
                     logger.critical(f"Error verifying Swift container '{settings.openstack.swift_container_name}': {e}", exc_info=True)
                     mlflow.end_run("FAILED")
                     sys.exit(1)


            except Exception as e: # Catch SDKException, ConnectionError etc.
                logger.critical(f"Failed to initialize OpenStack Swift connection: {e}", exc_info=True)
                # Note: If start_run failed, this won't run end_run.
                # The outer try/finally handles end_run status.
                raise # Re-raise to trigger the outer exception handling


            # 4. Load Dataset from Swift
            dataset = None
            try:
                logger.info(f"Loading dataset from Swift container '{settings.openstack.swift_container_name}'...")
                dataset = SwiftCropYieldDataset(
                    swift_container_name=settings.openstack.swift_container_name,
                    crop_name=settings.training.crop_name,
                    swift_conn=swift_conn # Pass the active connection
                )
                logger.info("Dataset loaded from Swift.")

                if len(dataset) == 0:
                    logger.critical("Dataset loaded but contains no samples. Check ETL or data paths.")
                    raise ValueError("Dataset is empty.")

                # Get data dimensions and FIPS mapping from the fully loaded dataset
                # These are needed for model initialization and logging/saving
                # Accessing dataset[0] is safe if len(dataset) > 0
                sample_weather, _, _ = dataset[0]
                input_dim = sample_weather.shape[-1] # Number of weather features
                fips_to_id_mapping, id_to_fips_mapping = dataset.get_fips_mapping()
                num_fips = dataset.get_num_fips() # Number of FIPS with actual samples

                logger.info(f"Detected input weather dimension from loaded data: {input_dim}")
                logger.info(f"Number of unique FIPS codes with samples loaded: {num_fips}")
                logger.info(f"Years with samples loaded: {dataset.get_years()}")

                # Log dataset details to MLflow
                mlflow.log_param("dataset_size", len(dataset))
                mlflow.log_param("input_weather_dim_actual", input_dim)
                mlflow.log_param("num_unique_fips_in_dataset", num_fips)
                mlflow.log_param("dataset_years_loaded", str(dataset.get_years()))

                # Log the FIPS mapping as an artifact
                # MLflow automatically handles saving to the configured artifact location (Swift)
                with tempfile.TemporaryDirectory() as tmpdir:
                     mapping_path = os.path.join(tmpdir, "fips_to_id_mapping.json")
                     with open(mapping_path, "w") as f:
                         json.dump(fips_to_id_mapping, f)
                     # Log the artifact relative to the run's artifact root
                     mlflow.log_artifact(mapping_path, artifact_path="fips_mapping")
                     logger.info(f"FIPS mapping logged as artifact 'fips_mapping/fips_to_id_mapping.json'.")


            except Exception as e: # Catch errors from SwiftCropYieldDataset init or data access
                logger.critical(f"Failed to load dataset from Swift: {e}", exc_info=True)
                raise # Re-raise to trigger outer exception handling


            # 5. Prepare DataLoaders
            train_loader, val_loader, holdout_dataset = None, None, None
            try:
                logger.info("Preparing dataloaders...")
                # Use the get_dataloaders_by_year utility function
                train_loader, val_loader, holdout_dataset = get_dataloaders_by_year(
                    dataset,
                    holdout_year=settings.training.holdout_year,
                    val_year_ratio=settings.training.val_year_ratio,
                    batch_size=settings.training.batch_size
                )

                # Log data split details to MLflow
                mlflow.log_param("train_samples", len(train_loader.dataset))
                mlflow.log_param("val_samples", len(val_loader.dataset))
                mlflow.log_param("holdout_samples", len(holdout_dataset) if holdout_dataset else 0)
                mlflow.log_param("split_holdout_year_used", str(settings.training.holdout_year)) # Log used config
                mlflow.log_param("split_val_year_ratio_used", settings.training.val_year_ratio)
                mlflow.log_param("train_years_used", str(sorted(dataset.get_years()))) # Log actual years used for train/val after split
                mlflow.log_param("val_years_used", str(sorted(dataset.get_years()))) # This logic needs refinement in get_dataloaders_by_year to return years used for each split


            except ValueError as e: # Catch errors from get_dataloaders_by_year (e.g., insufficient data)
                logger.critical(f"Failed to create dataloaders based on years: {e}", exc_info=True)
                raise # Re-raise to trigger outer exception handling


            # 6. Instantiate Model
            model = None
            try:
                logger.info("Instantiating model...")
                # Ensure num_fips is > 0 before creating embedding layer
                if num_fips <= 0:
                    logger.critical("Number of FIPS codes with samples is 0. Cannot instantiate model.")
                    raise ValueError("Cannot instantiate model with 0 FIPS codes.")

                model = LSTMTCNRegressor(
                    input_dim=input_dim, # Use detected input dim
                    num_fips=num_fips,   # Use number of FIPS with samples
                    fips_embedding_dim=settings.training.fips_embedding_dim,
                    hidden_dim=settings.training.hidden_dim,
                    lstm_layers=settings.training.lstm_layers,
                    tcn_channels=settings.training.tcn_channels,
                    dropout_rate=settings.training.dropout_rate
                )
                # Log model architecture details implicitly via logged parameters
                logger.info("Model instantiated.")

            except Exception as e:
                logger.critical(f"Failed to instantiate model: {e}", exc_info=True)
                raise # Re-raise to trigger outer exception handling


            # 7. Train Model
            trained_model = None
            try:
                logger.info("Starting training process...")
                trained_model = train_model(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=settings.training.num_epochs,
                    lr=settings.training.learning_rate
                )
                logger.info("Training process finished.")

            except Exception as e:
                logger.critical(f"Model training failed: {e}", exc_info=True)
                raise # Re-raise to trigger outer exception handling


            # 8. Evaluate Model (on Validation set)
            # Validation metrics are logged per epoch in train_model
            # We run evaluate_model here to get final validation metrics for logging once
            val_rmse, val_mae = None, None
            try:
                logger.info("Evaluating model on validation set...")
                val_rmse, val_mae = evaluate_model(trained_model, val_loader)
                # Log final validation metrics
                mlflow.log_metric("final_val_rmse", val_rmse)
                mlflow.log_metric("final_val_mae", val_mae)
                logger.info("Validation evaluation complete.")

            except Exception as e:
                logger.error(f"Failed to evaluate model on validation set: {e}", exc_info=True)
                # Evaluation failure might be a warning, depending on policy
                # Decide if this should fail the job or just log the error.
                # For now, let's log and continue to model logging.

            # Optional: Evaluate on the holdout set as a final, non-training-influenced check
            if holdout_dataset and len(holdout_dataset) > 0:
                 holdout_loader = DataLoader(holdout_dataset, batch_size=settings.training.batch_size, shuffle=False, collate_fn=collate_fn)
                 try:
                      logger.info("Evaluating model on holdout set...")
                      holdout_rmse, holdout_mae = evaluate_model(trained_model, holdout_loader)
                      mlflow.log_metric("holdout_rmse", holdout_rmse)
                      mlflow.log_metric("holdout_mae", holdout_mae)
                      logger.info("Holdout evaluation complete.")
                 except Exception as e:
                      logger.error(f"Failed to evaluate model on holdout set: {e}", exc_info=True)
                      # This evaluation failure might be less critical than training/val


            # 9. Log and Register Model
            try:
                logger.info(f"Logging model '{settings.mlflow.model_name}'...")
                # Log the trained PyTorch model
                # Use mlflow.pytorch.log_model which handles PyTorch specific logging
                # It will upload to the configured MLflow artifact location (Swift)
                mlflow.pytorch.log_model(
                    trained_model,
                    artifact_path="pytorch_model", # Artifact path within the run
                    registered_model_name=settings.mlflow.model_name, # Name in MLflow Model Registry
                    # Add pip requirements for reproducibility
                    # Use poetry export --without-hashes -f requirements.txt > requirements.txt locally
                    # Then include it in the Dockerfile and reference it here
                    # pip_requirements=["torch", "pandas", "numpy", "openstacksdk", "tqdm", "scikit-learn"], # List core deps
                    # Using a requirements.txt file is better practice
                    # Example: requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt") # If requirements.txt is in the same dir
                    # pip_requirements=requirements_path # Pass the path to the file

                    # Add a signature for better model understanding (requires sample input)
                    # Can create a dummy input tensor here matching the expected shape (batch=1)
                    # sample_input = (torch.randn(1, 180, input_dim), torch.tensor([0], dtype=torch.long)) # Example shape (batch=1, max_seq_len, features), dummy fips_id
                    # try:
                    #     signature = mlflow.models.infer_signature(sample_input, trained_model(*sample_input))
                    #     mlflow.pytorch.log_model(..., signature=signature) # Add signature
                    #     logger.info("Model signature inferred and logged.")
                    # except Exception as sig_e:
                    #     logger.warning(f"Failed to infer/log model signature: {sig_e}")
                    #     # Log model without signature if inference fails
                    #     mlflow.pytorch.log_model(
                    #         trained_model,
                    #         artifact_path="pytorch_model",
                    #         registered_model_name=settings.mlflow.model_name,
                    #         pip_requirements=["torch", "pandas", "numpy", "openstacksdk", "tqdm", "scikit-learn"]
                    #     )


                )
                logger.info(f"Model logged as artifact 'pytorch_model' and registered as '{settings.mlflow.model_name}'.")

                # Job succeeded if we reached here
                job_status = "FINISHED"

                # Optional: Transition the newly logged version to a stage (e.g., 'Staging')
                # This can be automated here, or handled by a separate job/pipeline step
                # after validation. Transitioning to 'Production' automatically might be risky.
                # Example of getting the latest version and transitioning:
                # try:
                #      client = mlflow.tracking.MlflowClient()
                #      # Find the model version created by *this* run
                #      latest_version = None
                #      for mv in client.search_model_versions(f"name='{settings.mlflow.model_name}'"):
                #           if mv.run_id == active_run.info.run_id:
                #                latest_version = mv
                #                break
                #
                #      if latest_version:
                #           client.transition_model_version_stage(
                #               name=settings.mlflow.model_name,
                #               version=latest_version.version,
                #               stage="Staging", # Transition to staging for review
                #               archive_existing_versions=False # Don't archive existing production/staging versions
                #           )
                #           logger.info(f"Transitioned model version {latest_version.version} to 'Staging'.")
                #      else:
                #           logger.warning("Could not find the model version created by this run to transition its stage.")
                # except Exception as transition_e:
                #      logger.error(f"Failed to transition model version stage: {transition_e}", exc_info=True)
                #      # This failure doesn't necessarily mean the *job* failed, but the desired deployment step failed.
                #      # Mark the run with a warning or specific tag instead of failing the whole job? Depends on policy.


            except Exception as e:
                logger.critical(f"Failed to log or register model: {e}", exc_info=True)
                # If model logging fails, the job is considered failed as the primary artifact wasn't saved.
                job_status = "FAILED" # Ensure status is failed
                raise # Re-raise to trigger outer exception handling


    except Exception as e:
        # This catches exceptions re-raised from inside the blocks (config, swift, data, model, train, log)
        # The specific error is already logged inside the block where it occurred.
        # Just log a final message and ensure the MLflow run status is set.
        logger.critical(f"Training job encountered a critical error. Final status: {job_status}", exc_info=True)
        # If the exception happened *before* start_run, active_run won't exist.
        # If it happened *inside* start_run, the 'with' block handles end_run("FAILED").
        # If it happened *after* start_run but outside the 'with' block (less likely), manually end.
        # The current structure with the 'with mlflow.start_run()' block handles this correctly.


    finally:
        # 10. Cleanup (Close Swift Connection)
        if swift_conn:
            try:
                swift_conn.close()
                logger.info("Swift connection closed.")
            except Exception as e:
                 logger.warning(f"Error closing OpenStack Swift connection: {e}")

        # End the MLflow run explicitly if needed (the 'with' block does this)
        # If an exception occurs, the 'with' block automatically ends the run with status 'FAILED'
        # Otherwise, it ends with 'FINISHED'.
        # If you weren't using the 'with' block, you'd need explicit try/except/finally with mlflow.end_run() calls.
        # mlflow.end_run(status=job_status) # This is handled by the 'with' block

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logger.info(f"--- Model Training Job Finished ({job_status}) ---")
        logger.info(f"Total job duration: {duration}")

        # Exit with appropriate status code
        if job_status == "FINISHED":
             sys.exit(0) # Success
        else:
             sys.exit(1) # Failure


if __name__ == "__main__":
    main()