#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import shutil
from datetime import timedelta
import gdown

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
    p.add_argument("-o", "--output", default="/app/downloaded_temp", help="Temporary local directory for downloads (inside VM)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress gdown progress bars")
    p.add_argument("--no-cookies", action="store_false", dest="use_cookies",
                   help="Disable cookies for gdown (for public folders)")
    p.add_argument("-r", "--retry", type=int, default=3, help="Max retry attempts per file for download")

    p.add_argument("--clean", action="store_true", help="Clean up the root local download directory after successful download")

    return p.parse_args()

# --- Recursive Function to Process Folder Contents File-by-File ---
def process_folder_contents(gdrive_folder_id, local_base_output_dir, current_local_path_rel, logger, quiet, use_cookies, retries):
    """
    Recursively processes Google Drive folder contents, downloading files one by one.
    """
    # Build the full local path for the current level inside the base output directory
    full_current_local_path = os.path.join(local_base_output_dir, current_local_path_rel)
    os.makedirs(full_current_local_path, exist_ok=True)  # Ensure directory exists locally

    logger.info(f"Processing Google Drive folder ID: {gdrive_folder_id} (Local path: '{full_current_local_path}')")

    try:
        # This should list files in the Google Drive folder
        items = gdown.get_urls_from_gdrive_folder(
            gdrive_folder_id,
            quiet=quiet,
            use_cookies=use_cookies
        )

        if not items:
            logger.warning(f"Google Drive folder ID {gdrive_folder_id} appears empty or could not be listed.")
            return True  # Consider an empty folder processed successfully

        for item in items:
            item_name = item.get('name')
            item_id = item.get('id')
            is_folder = item.get('isFolder', False)  # Default to False if key is missing
            # Basic check for essential info
            if not item_name or not item_id:
                logger.warning(f"Skipping item with missing name or ID in folder {gdrive_folder_id}: {item}")
                continue

            # Construct the full local path for this item (file or subdir)
            item_local_path = os.path.join(full_current_local_path, item_name)

            if is_folder:
                # Recursively process sub-folder
                logger.info(f"Found folder: '{item_name}' (ID: {item_id}). Recursing...")
                success = process_folder_contents(
                    item_id,
                    local_base_output_dir,      # Pass the *base* output dir down
                    os.path.join(current_local_path_rel, item_name),  # Build the relative local path
                    logger, quiet, use_cookies, retries
                )
                if not success:
                    logger.error(f"Recursive processing failed for folder '{item_name}' (ID: {item_id}). Propagating failure.")
                    return False  # Propagate failure up

            else:  # It's a file
                logger.info(f"Found file: '{item_name}' (ID: {item_id}). Processing...")

                # Ensure the local directory for this file exists before downloading
                item_local_dir = os.path.dirname(item_local_path)
                os.makedirs(item_local_dir, exist_ok=True)

                # --- Download the file ---
                download_successful = False
                last_download_exception = None
                for attempt in range(1, retries + 1):
                    try:
                        logger.info(f"Download attempt {attempt}/{retries} for file '{item_name}' (ID: {item_id}) to '{item_local_path}'")
                        gdown.download(
                            id=item_id,
                            output=item_local_path,
                            quiet=quiet,
                            use_cookies=use_cookies,
                        )
                        logger.info(f"Download successful for '{item_name}'.")
                        download_successful = True
                        break  # Exit retry loop on success
                    except Exception as e:
                        last_download_exception = e
                        logger.error(f"Download attempt {attempt} failed for '{item_name}': {e}")
                        if attempt < retries:
                            wait_time = 15 * attempt
                            logger.info(f"Retrying download in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.critical(f"Max download retries ({retries}) reached for file '{item_name}'.")

                if not download_successful:
                    logger.critical(f"Failed to download file '{item_name}' (ID: {item_id}) after {retries} attempts.")
                    continue  # Move to the next item in the Google Drive folder listing

        # If loop completes for all immediate children, this level is processed
        logger.info(f"Finished processing items in Google Drive folder ID: {gdrive_folder_id}")
        return True  # Indicate success for this folder level

    except Exception as e:
        logger.critical(f"Critical error while listing Google Drive folder ID {gdrive_folder_id}: {e}")
        return False  # Signal critical failure for the pipeline

# --- Main ETL Pipeline Function (Runs inside Docker) ---
def main():
    """Orchestrates the file-by-file download steps inside the container."""
    logger = setup_logger()
    args = parse_args()

    # Define the base local directory where files will be temporarily downloaded
    base_download_dir = args.output

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