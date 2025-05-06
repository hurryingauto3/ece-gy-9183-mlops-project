#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import shutil
from datetime import timedelta

# Ensure dependencies are available - this check is more for local dev/testing
try:
    import gdown
    import openstack
    import keystoneauth1.exceptions.auth as ka_exceptions
except ImportError:
    print("Error: Required Python packages are not installed.")
    print("Ensure Poetry is installed, dependencies are added to pyproject.toml, and `poetry install` has been run.")
    sys.exit(1)

# --- Setup Functions ---

def setup_logger():
    """Sets up and returns a logger."""
    logger = logging.getLogger("drive_to_swift_etl")
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
    p = argparse.ArgumentParser(description="ETL pipeline to download Google Drive files and push to OpenStack Swift (file-by-file).")
    # Accept multiple folder IDs (space-separated)
    p.add_argument("folder_ids", nargs='+', help="Google Drive root folder IDs (from the URL)")
    # Default output dir inside the Docker container's WORKDIR /app. Files will be downloaded here temporarily.
    p.add_argument("-o", "--output", default="/app/downloaded_temp", help="Temporary local directory for downloads (inside container)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress gdown progress bars")
    p.add_argument("--no-cookies", action="store_false", dest="use_cookies",
                   help="Disable cookies for gdown (for public folders)")
    p.add_argument("-r", "--retry", type=int, default=3, help="Max retry attempts per file for download/upload")

    p.add_argument("--swift-container", required=True,
                   help="Name of the OpenStack Swift container to upload data to")
    p.add_argument("--clean", action="store_true", help="Clean up the root local download directory after successful upload")

    return p.parse_args()

# --- Upload Function for a Single File ---

def upload_to_swift_single_file(conn, local_file_path, swift_container, swift_object_name, logger):
    """Uploads a single local file to an OpenStack Swift container using an existing connection."""
    logger.info(f"Uploading '{local_file_path}' to Swift as '{swift_container}/{swift_object_name}'")
    try:
        # Check if the local file actually exists before trying to open
        if not os.path.exists(local_file_path):
             logger.error(f"Local file to upload not found: '{local_file_path}'")
             return False

        # Open the file in binary read mode
        with open(local_file_path, 'rb') as f:
            conn.object_store.upload_object(
                container=swift_container,
                obj=swift_object_name,
                data=f,
                # Set a timeout for upload operations (optional, defaults exist)
                timeout=3600 # 1 hour timeout per object upload
            )
        logger.debug(f"Successfully uploaded '{swift_object_name}'") # Use debug for file-by-file success details
        return True

    except openstack.exceptions.SDKException as e:
        # Log OpenStack specific errors but don't necessarily re-raise
        logger.error(f"OpenStack SDK error during upload of '{swift_object_name}': {e}")
        return False # Upload failed, indicate failure
    except Exception as e:
        # Catch any other unexpected errors during upload
        logger.error(f"An unexpected error occurred during Swift upload of '{swift_object_name}': {e}")
        return False # Upload failed, indicate failure

# --- Recursive Function to Process Folder Contents File-by-File ---

def process_folder_contents(gdrive_folder_id, swift_container, local_base_output_dir, current_local_path_rel, current_swift_path_rel, logger, quiet, use_cookies, retries, swift_conn):
    """
    Recursively processes Google Drive folder contents, downloading, uploading,
    and deleting files one by one.
    """
    # Build the full local path for the current level inside the base output directory
    full_current_local_path = os.path.join(local_base_output_dir, current_local_path_rel)
    os.makedirs(full_current_local_path, exist_ok=True) # Ensure directory exists locally

    logger.info(f"Processing Google Drive folder ID: {gdrive_folder_id} (Local path: '{full_current_local_path}', Swift path: '{current_swift_path_rel}')")

    try:
        # Use gdown to list contents of the current folder
        # This lists immediate children, we recurse for subfolders
        items = gdown.get_urls_from_gdrive_folder(
            gdrive_folder_id,
            quiet=quiet,
            use_cookies=use_cookies
        )

        if not items:
             logger.warning(f"Google Drive folder ID {gdrive_folder_id} appears empty or could not be listed.")
             # Log contents of the directory if any were partially created during a previous run
             if os.path.exists(full_current_local_path):
                 logger.debug(f"Contents of local dir '{full_current_local_path}' after listing empty GDrive folder: {os.listdir(full_current_local_path)}")
             return True # Consider an empty folder processed successfully

        for item in items:
            item_name = item.get('name')
            item_id = item.get('id')
            is_folder = item.get('isFolder', False) # Default to False if key is missing
            # Basic check for essential info
            if not item_name or not item_id:
                 logger.warning(f"Skipping item with missing name or ID in folder {gdrive_folder_id}: {item}")
                 continue

            # Construct the full local path for this item (file or subdir)
            item_local_path = os.path.join(full_current_local_path, item_name)
            # Construct the full Swift object path for this item
            # Swift object names use forward slashes regardless of OS
            item_swift_path = os.path.join(current_swift_path_rel, item_name).replace(os.sep, '/')
            # Remove leading slash if it exists (Swift object names shouldn't start with '/')
            if item_swift_path.startswith('/'):
                 item_swift_path = item_swift_path[1:]


            if is_folder:
                # Recursively process sub-folder
                logger.info(f"Found folder: '{item_name}' (ID: {item_id}). Recursing...")
                success = process_folder_contents(
                    item_id,
                    swift_container,
                    local_base_output_dir,      # Pass the *base* output dir down
                    os.path.join(current_local_path_rel, item_name), # Build the relative local path
                    item_swift_path,            # Build the relative swift path
                    logger, quiet, use_cookies, retries, swift_conn
                )
                if not success:
                    logger.error(f"Recursive processing failed for folder '{item_name}' (ID: {item_id}). Propagating failure.")
                    return False # Propagate failure up

            else: # It's a file
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
                        # gdown.download for a single file often doesn't have the 'timeout' argument in older versions
                        # Check your gdown version if you get unexpected argument errors
                        gdown.download(
                            id=item_id,
                            output=item_local_path,
                            quiet=quiet,
                            use_cookies=use_cookies,
                            # timeout=... # If your gdown version supports it, add a timeout here
                        )
                        logger.info(f"Download successful for '{item_name}'.")
                        download_successful = True
                        break # Exit retry loop on success
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
                    logger.critical(f"Failed to download file '{item_name}' (ID: {item_id}) after {retries} attempts. Skipping upload and delete for this file.")
                    continue # Move to the next item in the Google Drive folder listing

                # --- Upload the file to Swift ---
                upload_successful = False
                # Retry loop for upload
                last_upload_exception = None
                for attempt in range(1, retries + 1):
                     try:
                         logger.info(f"Upload attempt {attempt}/{retries} for file '{item_name}' (local: '{item_local_path}', swift: '{item_swift_path}')")
                         upload_successful = upload_to_swift_single_file(
                             swift_conn, swift_container, item_swift_path, logger
                         )
                         if upload_successful:
                             logger.info(f"Upload successful for '{item_name}'.")
                             break # Exit retry loop on success
                         else:
                             pass # Error already logged
                     except Exception as e:
                          last_upload_exception = e
                          logger.error(f"Unexpected error during upload attempt {attempt} for '{item_name}': {e}")

                     if attempt < retries:
                         wait_time = 15 * attempt
                         logger.info(f"Retrying upload in {wait_time}s...")
                         time.sleep(wait_time)
                     else:
                         logger.critical(f"Max upload retries ({retries}) reached for file '{item_name}'.")

                # --- Delete the local file ---
                logger.info(f"Deleting local file: '{item_local_path}'...")
                try:
                    if os.path.exists(item_local_path):
                        os.remove(item_local_path)
                        logger.debug(f"Local file deleted: '{item_local_path}'")
                except Exception as e:
                    logger.error(f"Failed to delete local file '{item_local_path}': {e}")

        # If loop completes for all immediate children, this level is processed
        logger.info(f"Finished processing items in Google Drive folder ID: {gdrive_folder_id}")
        return True # Indicate success for this folder level

    except Exception as e:
        # Catch errors during initial folder listing (e.g., permissions, network)
        logger.critical(f"Critical error while listing Google Drive folder ID {gdrive_folder_id}: {e}")
        return False # Signal critical failure for the pipeline

# --- Main ETL Pipeline Function (Runs inside Docker) ---

def main():
    """Orchestrates the file-by-file download, upload, and delete steps inside the container."""
    logger = setup_logger()
    args = parse_args()

    # Define the base local directory where files will be temporarily downloaded
    base_download_dir = args.output

    logger.info("--- ETL Pipeline Start (File-by-File) ---")
    logger.info(f"Google Drive folder IDs: {args.folder_ids}")
    logger.info(f"Swift container: {args.swift_container}")
    logger.info(f"Base temporary download path (inside container): {base_download_dir}")
    logger.info(f"Quiet mode: {args.quiet}")
    logger.info(f"Use cookies: {args.use_cookies}")
    logger.info(f"Max retries per file (download/upload): {args.retry}")
    logger.info(f"Clean up root download dir at end: {args.clean}")

    # Ensure the base local directory exists
    os.makedirs(base_download_dir, exist_ok=True)

    swift_conn = None
    pipeline_success = False

    try:
        # Establish OpenStack connection once for potentially better performance
        logger.info("Connecting to OpenStack Swift using environment variables...")
        swift_conn = openstack.connect() # Reads OS_* env vars automatically
        logger.info("OpenStack connection successful.")

        # Check/Create the Swift container once at the start
        try:
            logger.info(f"Ensuring Swift container '{args.swift_container}' exists.")
            swift_conn.object_store.create_container(args.swift_container, ignore_existing=True)
            logger.info(f"Swift container '{args.swift_container}' is ready.")
        except Exception as e:
            logger.critical(f"Fatal: Could not access or create Swift container '{args.swift_container}': {e}")
            sys.exit(1) # Exit if container setup fails

        # Iterate through all the provided Google Drive folder IDs and process them
        for folder_id in args.folder_ids:
            pipeline_success = process_folder_contents(
                folder_id,
                args.swift_container,
                base_download_dir,
                '', # Initial relative local path (relative to base_download_dir)
                '', # Initial relative swift path (root of container)
                logger, args.quiet, args.use_cookies, args.retry, swift_conn
            )

            if not pipeline_success:
                logger.critical(f"ETL pipeline failed during file/folder processing for folder ID: {folder_id}")
                sys.exit(1) # Exit with non-zero code if any folder processing fails

        logger.info("File-by-File ETL Pipeline Finished.")
        sys.exit(0) # Exit successfully if all folders processed without critical errors

    except openstack.exceptions.NotFoundException as e:
        logger.critical(f"OpenStack authentication error: Missing or invalid authentication plugin. Ensure OS_* environment variables are set correctly in the container.")
        sys.exit(1) # Exit on authentication failure
    except openstack.exceptions.SDKException as e:
        logger.critical(f"OpenStack SDK error during initial connection or container setup: {e}")
        sys.exit(1) # Exit on critical OpenStack setup failure
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during the ETL pipeline: {e}")
        sys.exit(1) # Exit on unexpected critical errors
    finally:
        # Attempt to close the connection if it was successfully opened
        if swift_conn:
            try:
                swift_conn.close()
                logger.info("OpenStack connection closed.")
            except Exception as e:
                 logger.warning(f"Error closing OpenStack connection: {e}")

        # --- Final Cleanup (of the base download directory) ---
        if args.clean and os.path.exists(base_download_dir):
            logger.info(f"Cleaning up base local download directory: '{base_download_dir}'")
            try:
                shutil.rmtree(base_download_dir, ignore_errors=False) # Fail if removal fails
                logger.info("Base local download directory cleaned up.")
            except Exception as e:
                 logger.error(f"Failed to clean up base local directory '{base_download_dir}': {e}")
                 # Consider setting the exit code to 1 here if cleanup failure is critical
        elif args.clean:
             logger.info("Cleanup requested, but base download path was not created or found.")
        else:
             logger.info("Cleanup not requested. Base local download directory retained.")

if __name__ == "__main__":
    main()