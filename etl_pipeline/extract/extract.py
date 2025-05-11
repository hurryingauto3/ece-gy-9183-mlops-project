#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import shutil
from datetime import timedelta
import gdown

output_dir = "/mnt/swift_store/raw_data"

# Ensure dependencies are available - this check is more for local dev/testing
try:
    import gdown
except ImportError:
    print("Error: Required Python packages are not installed.")
    print("Ensure Poetry is installed, dependencies are added to pyproject.toml, and `poetry install` has been run.")
    sys.exit(1)

# --- Setup Functions ---

def setup_logger():
    """Sets up and returns a logger."""
    logger = logging.getLogger("drive_to_local_etl")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set level to INFO by default, DEBUG might be useful for detailed file-by-file output
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG) # Uncomment for detailed debugging
    return logger

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="ETL pipeline to download Google Drive files locally.")
    # Accept multiple folder IDs (space-separated)
    p.add_argument("folder_ids", nargs='+', help="Google Drive root folder IDs (from the URL)")
    # Default output dir inside the VM. Files will be downloaded here temporarily.
    p.add_argument("-o", "--output", default="/mnt/swift_store/raw_data", help="Temporary local directory for downloads (inside VM)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress gdown progress bars")
    p.add_argument("--no-cookies", action="store_false", dest="use_cookies",
                   help="Disable cookies for gdown (for public folders)")
    p.add_argument("-r", "--retry", type=int, default=3, help="Max retry attempts per file for download")

    p.add_argument("--clean", action="store_true", help="Clean up the root local download directory after successful download")

    return p.parse_args()

# --- Recursive Function to Process Folder Contents File-by-File ---
def process_folder_contents(gdrive_folder_id, local_base_output_dir, current_local_path_rel, logger, quiet, use_cookies, retries):
    """
    Downloads contents of a Google Drive folder using gdown.download_folder.
    Applies folder-specific cleanup after download.
    """
    # Download to the base output dir; gdown will create its own subfolder
    download_parent_path = os.path.join(local_base_output_dir, current_local_path_rel)
    os.makedirs(download_parent_path, exist_ok=True)

    logger.info(f"Processing Google Drive folder ID: {gdrive_folder_id} (Download to: '{download_parent_path}')")

    try:
        # Download folder
        success = gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{gdrive_folder_id}",
            output=download_parent_path,
            quiet=quiet,
            use_cookies=use_cookies
        )

        if not success:
            logger.warning(f"Download failed or folder appears empty: {gdrive_folder_id}")
            return False

        logger.info(f"Successfully downloaded folder ID: {gdrive_folder_id}")

        # --- Locate the actual downloaded folder ---
        downloaded_folders = [
            os.path.join(download_parent_path, name)
            for name in os.listdir(download_parent_path)
            if os.path.isdir(os.path.join(download_parent_path, name))
        ]

        if not downloaded_folders:
            logger.warning("No folders found after download.")
            return False

        # Assume first new folder is the downloaded one (simplest case)
        downloaded_path = downloaded_folders[0]

        # --- Post-download cleanup ---
        year_whitelist = {"2017", "2018", "2019", "2020", "2021"}

        if "WRF-HRRR" in downloaded_path:
            hrrr_data_path = os.path.join(downloaded_path, "data")
            if os.path.isdir(hrrr_data_path):
                for folder in os.listdir(hrrr_data_path):
                    folder_path = os.path.join(hrrr_data_path, folder)
                    if os.path.isdir(folder_path) and folder not in year_whitelist:
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted HRRR/data folder not in allowed list: {folder_path}")

        elif "USDA Crop" in downloaded_path:
            usda_crop_path = os.path.join(downloaded_path, "data", "crop")
            if os.path.isdir(usda_crop_path):
                for folder in os.listdir(usda_crop_path):
                    folder_path = os.path.join(usda_crop_path, folder)
                    if os.path.isdir(folder_path) and folder not in year_whitelist:
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted USDA/data/crop folder not in allowed list: {folder_path}")

        return True

    except Exception as e:
        logger.critical(f"Critical error while downloading Google Drive folder ID {gdrive_folder_id}: {e}")
        return False

# --- Main ETL Pipeline Function (Runs inside Docker) ---
def main():
    """Orchestrates the file-by-file download steps inside the container."""
    logger = setup_logger()
    args = parse_args()

    # Define the base local directory where files will be temporarily downloaded
    base_download_dir = args.output
    print("DEBUG ARGS:", args.folder_ids)

    logger.info("--- ETL Pipeline Start (File-by-File) ---")
    logger.info(f"Google Drive folder IDs: {args.folder_ids}")
    logger.info(f"Base temporary download path (inside VM): {base_download_dir}")
    logger.info(f"Quiet mode: {args.quiet}")
    logger.info(f"Use cookies: {args.use_cookies}")
    logger.info(f"Max retries per file (download): {args.retry}")
    logger.info(f"Clean up root download dir at end: {args.clean}")

    # Ensure the base local directory exists
    os.makedirs(base_download_dir, exist_ok=True)

    pipeline_success = False

    # Iterate through all the provided Google Drive folder IDs and process them
    for folder_id in args.folder_ids:
        pipeline_success = process_folder_contents(
            folder_id,
            base_download_dir,
            '',  # Initial relative local path (relative to base_download_dir)
            logger, args.quiet, args.use_cookies, args.retry
        )

        if not pipeline_success:
            logger.critical(f"ETL pipeline failed during file/folder processing for folder ID: {folder_id}")
            sys.exit(1)  # Exit with non-zero code if any folder processing fails

    logger.info("File-by-File ETL Pipeline Finished.")
    sys.exit(0)  # Exit successfully if all folders processed without critical errors

if __name__ == "__main__":
    main()