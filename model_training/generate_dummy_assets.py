import torch
import json
import os
from pathlib import Path

# If model.py is in the same directory or PYTHONPATH when this script runs:
from model import DummyLSTMTCNHistogramPredictor, logger

# Original import commented out:
# from model_training.model import DummyLSTMTCNHistogramPredictor, logger # Assuming logger is also in model.py


def generate_assets(output_dir_str: str):
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating dummy assets in directory: {output_dir}")

    # --- Model Parameters ---
    # These should align with what mlflClow_loader.py might expect for a dummy model
    # or what your tests require.
    dummy_input_dim = 10  # Example: number of weather features after processing
    dummy_num_fips = 3    # For dummy FIPS IDs 0, 1, 2
    dummy_num_crops = 3   # For dummy Crop IDs 0, 1, 2
    dummy_num_bins = int(os.environ.get("MLFLOW_DUMMY_MODEL_NUM_BINS", "5")) # Get from env or default

    # --- 1. Generate and Save Dummy Model State ---
    model_path = output_dir / "dummy_model_state.pth"
    try:
        logger.info(f"Initializing DummyLSTMTCNHistogramPredictor with num_bins: {dummy_num_bins}")
        dummy_model = DummyLSTMTCNHistogramPredictor(
            input_dim=dummy_input_dim,
            num_fips=dummy_num_fips,
            num_crops=dummy_num_crops,
            num_bins=dummy_num_bins
        )
        torch.save(dummy_model.state_dict(), model_path)
        logger.info(f"Dummy model state_dict saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save dummy model: {e}", exc_info=True)
        raise

    # --- 2. Generate and Save Dummy FIPS Mapping ---
    fips_map_path = output_dir / "fips_to_id_mapping.json"
    # Basic FIPS map, ensure "DUMMY" and "00000" are present for feature_serving dummy mode
    dummy_fips_map = {
        "DUMMY": 0,
        "00000": 0,  # Often used as a dummy/default FIPS
        "19153": 1,  # Example real FIPS
        "17031": 2   # Another example
    }
    # Ensure num_fips in model init is >= max_id_in_fips_map + 1
    # For simplicity, the model was initialized with dummy_num_fips which should be enough.

    try:
        with open(fips_map_path, "w") as f:
            json.dump(dummy_fips_map, f, indent=4)
        logger.info(f"Dummy FIPS mapping saved to: {fips_map_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save dummy FIPS mapping: {e}", exc_info=True)
        raise

    # --- 3. Generate and Save Dummy Crop Mapping ---
    crop_map_path = output_dir / "crop_to_id_mapping.json"
    dummy_crop_map = {
        "corn": 0,
        "soybeans": 1,
        "dummy_crop": 2 # A generic dummy crop
    }
    # Ensure num_crops in model init is >= max_id_in_crop_map + 1

    try:
        with open(crop_map_path, "w") as f:
            json.dump(dummy_crop_map, f, indent=4)
        logger.info(f"Dummy CROP mapping saved to: {crop_map_path}")
    except Exception as e:
        logger.error(f"Failed to generate or save dummy CROP mapping: {e}", exc_info=True)
        raise

    logger.info("All dummy assets generated successfully.")

if __name__ == "__main__":
    # This script will be run inside the model-serving container.
    # The output directory should match where mlflow_loader.py expects to find them.
    # The Docker entrypoint script will ensure this path exists.
    ASSETS_OUTPUT_DIR = os.environ.get("DUMMY_ASSETS_DIR", "/app/dummy_assets")
    
    # Setup basic logging for the script itself if run directly
    # Note: structlog from the parent model_training.model might not be fully configured
    # if this script is run in a very minimal environment before the main app starts.
    # For simplicity, using print or basic logging here.
    # If using logger from model.py, ensure it's configured.
    
    print(f"--- Starting dummy asset generation in {ASSETS_OUTPUT_DIR} ---")
    try:
        generate_assets(ASSETS_OUTPUT_DIR)
        print(f"--- Dummy asset generation finished successfully. ---")
    except Exception as e:
        print(f"[ERROR] Failed to generate dummy assets: {e}")
        # Exit with an error code if generation fails, so the entrypoint script can react if needed
        import sys
        sys.exit(1) 