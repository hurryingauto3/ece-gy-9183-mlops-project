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
    # This will trigger if dependencies aren't installed via Poetry/pip
    print("Error: Required Python packages are not installed.")
    print("Ensure Poetry is installed, dependencies are added to pyproject.toml, and `poetry install` has been run.")
    sys.exit(1)

# --- Existing Setup Functions ---

def setup_logger():
    """Sets up and returns a logger."""
    logger = logging.getLogger("drive_to_swift_etl")
    # Prevent duplicate handlers if script is run incorrectly or module imported
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Log timestamp, level, and message
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set level based on verbosity/quietness if needed, default to INFO
    logger.setLevel(logging.INFO)
    return logger

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="ETL pipeline to download a Google Drive folder and push to OpenStack Swift.")
    p.add_argument("folder_id", help="Google Drive folder ID (from the URL)")
    # Default output dir inside the Docker container's WORKDIR /app
    p.add_argument("-o", "--output", default="/app/downloaded_data", help="Temporary local destination directory for download (inside container)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress gdown progress bars")
    p.add_argument("--no-cookies", action="store_false", dest="use_cookies",
                   help="Disable cookies for gdown (for public folders)")
    p.add_argument("-r", "--retry", type=int, default=3, help="Max retry attempts for gdown download failure")

    # Argument for OpenStack Swift container name - MUST be provided
    p.add_argument("--swift-container", required=True,
                   help="Name of the OpenStack Swift container to upload data to")
    # --clean argument is not needed internally, orchestrator cleans VM/container
    # Cleanup of the download directory itself is still useful inside the container
    p.add_argument("--clean", action="store_true", help="Clean up the local download directory after successful upload")

    return p.parse_args()

# --- Modified Download Function ---

def download_folder(folder_id, output_dir, quiet, use_cookies, retries, logger):
    """Downloads a Google Drive folder using gdown with retries. Returns the download path on success, raises Exception on failure."""
    logger.info(f"Starting download of Google Drive folder ID: {folder_id}")
    # Ensure the output directory exists *inside* the container
    os.makedirs(output_dir, exist_ok=True)
    start = time.monotonic()

    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{retries} → folder {folder_id} to '{output_dir}'")
            # Add a timeout for gdown operations for robustness
            gdown.download_folder(
                id=folder_id,
                output=output_dir,
                quiet=quiet,
                use_cookies=use_cookies,
                # timeout=900 # <--- REMOVE OR COMMENT OUT THIS LINE
            )
            duration = timedelta(seconds=int(time.monotonic() - start))
            logger.info(f"Download succeeded in {duration}")

            # summary - count files/dirs in the downloaded content within output_dir
            files = dirs = 0
            # walk starts from the output_dir itself
            for _, ds, fs in os.walk(output_dir):
                dirs += len(ds)
                files += len(fs)
            logger.info(f"Downloaded {files} files across {dirs} subdirs to '{output_dir}'")

            # Return the absolute path on success
            return os.path.abspath(output_dir)

        except Exception as e:
            last_exception = e
            logger.error(f"Download attempt {attempt} failed: {e}")
            if attempt < retries:
                wait_time = 15 * attempt # wait 15s, 30s, 45s etc.
                logger.info(f"Retrying download in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.critical(f"Max download retries ({retries}) reached.")

    # If loop finishes without returning, all attempts failed
    raise last_exception # Re-raise the last exception
# --- Upload Function (Uses OS_* env vars, same logic) ---

def upload_to_swift(local_directory, swift_container, logger):
    """Uploads the contents of a local directory to an OpenStack Swift container."""
    logger.info(f"Starting upload of local directory '{local_directory}' to Swift container '{swift_container}'")
    start = time.monotonic()
    uploaded_count = 0
    failed_uploads = []
    conn = None # Initialize conn outside try

    try:
        # Establish OpenStack connection using environment variables (OS_AUTH_URL, OS_PROJECT_NAME etc)
        logger.info("Connecting to OpenStack Swift using environment variables...")
        conn = openstack.connect() # Reads OS_* env vars automatically
        logger.info("OpenStack connection successful.")

        # Check if container exists, create if not
        try:
            # Use get_container to check existence, it raises NotFoundException if not found
            conn.object_store.get_container(swift_container)
            logger.info(f"Swift container '{swift_container}' already exists.")
        except openstack.exceptions.NotFoundException:
            logger.info(f"Swift container '{swift_container}' not found. Creating...")
            conn.object_store.create_container(swift_container)
            logger.info(f"Swift container '{swift_container}' created.")
        except Exception as e:
             # Catch other potential errors during container check/create
             logger.critical(f"Error accessing or creating Swift container '{swift_container}': {e}")
             return False # Indicate a fatal setup failure

        # Walk through the local directory containing the downloaded data
        # os.walk starts from local_directory, e.g., /app/downloaded_data
        # root will be /app/downloaded_data, /app/downloaded_data/subdir1, etc.
        # We want the Swift object name to be relative to the *contents* of local_directory.
        # Example: local_directory = /app/data, contains /app/data/subdir/file.txt
        # walk gives: root = /app/data/subdir, files = [file.txt]
        # relpath('/app/data/subdir', '/app/data') -> 'subdir'
        # swift_path_prefix becomes 'subdir/'
        # swift_object_name becomes 'subdir/file.txt'

        # Check if the download directory is empty after download (shouldn't happen if gdown succeeds but good check)
        if not os.listdir(local_directory):
             logger.warning(f"Download directory '{local_directory}' is empty. No files to upload.")
             return True # Consider this a successful (though empty) upload

        for root, _, files in os.walk(local_directory):
            # Determine the relative path from the base local_directory
            # This relative path becomes the object name prefix in Swift
            # If root is the same as local_directory, relative_path is '.'
            relative_path = os.path.relpath(root, local_directory)
            # Convert OS path separator to '/' for Swift object names
            swift_path_prefix = relative_path.replace(os.sep, '/') if relative_path != '.' else ''

            # Add a trailing slash to the prefix unless it's empty
            if swift_path_prefix and not swift_path_prefix.endswith('/'):
                 swift_path_prefix += '/'

            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                # Combine prefix and file name. If prefix is '', swift_object_name is just file_name.
                swift_object_name = swift_path_prefix + file_name

                logger.info(f"Uploading '{local_file_path}' to '{swift_container}/{swift_object_name}'")
                try:
                    # Open the file in binary read mode
                    with open(local_file_path, 'rb') as f:
                        # Optional: Get file size for logging or checks
                        # file_size = os.path.getsize(local_file_path)
                        # logger.debug(f"Uploading file size: {file_size} bytes")

                        conn.object_store.upload_object(
                            container=swift_container,
                            obj=swift_object_name,
                            data=f,
                            # Set a timeout for upload operations (optional, defaults exist)
                            timeout=3600 # 1 hour timeout per object upload
                        )
                    uploaded_count += 1
                    logger.debug(f"Successfully uploaded '{swift_object_name}'") # Use debug for file-by-file success details
                except Exception as e:
                    logger.error(f"Failed to upload '{local_file_path}' as '{swift_object_name}': {e}")
                    failed_uploads.append(local_file_path)
                    # By default, continue uploading other files even if one fails.
                    # Uncomment the line below if ANY file upload failure should stop the entire upload process.
                    # raise # Re-raise the exception immediately

        duration = timedelta(seconds=int(time.monotonic() - start))
        logger.info(f"Upload process finished in {duration}")

        if failed_uploads:
            logger.warning(f"Completed with {uploaded_count} successful uploads, but {len(failed_uploads)} failed.")
            for failed in failed_uploads:
                 logger.warning(f"  Failed: {failed}")
            # Returning True here signifies that the *upload loop completed*, even if some files failed.
            # Return False instead if any file failure should be considered a critical error for the job.
            # Let's return True for process completion, rely on logs/metrics for failure detection.
            return True
        else:
            logger.info(f"Successfully uploaded {uploaded_count} files to Swift container '{swift_container}'.")
            return True

    except ka_exceptions.MissingAuthPlugin as e:
        logger.critical(f"OpenStack authentication error: Missing or invalid authentication plugin. Ensure OS_* environment variables are set correctly in the container.")
        logger.critical(f"Details: {e}")
        return False # Indicate a critical failure before upload starts
    except openstack.exceptions.SDKException as e:
        logger.critical(f"OpenStack SDK error during connection or container operation: {e}")
        return False # Indicate a critical failure
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Swift upload: {e}")
        return False # Indicate an unexpected critical failure
    finally:
        # Attempt to close the connection if it was successfully opened
        if conn:
            try:
                conn.close()
                logger.debug("OpenStack connection closed.")
            except Exception as e:
                 logger.warning(f"Error closing OpenStack connection: {e}")


# --- Main ETL Pipeline Function (Runs inside Docker) ---

def main():
    """Orchestrates the download and upload steps inside the container."""
    logger = setup_logger()
    args = parse_args()

    # The download directory is specified via arguments, defaults to /app/downloaded_data
    download_dir = args.output

    logger.info("--- ETL Pipeline Start ---")
    logger.info(f"Google Drive folder ID: {args.folder_id}")
    logger.info(f"Swift container: {args.swift_container}")
    logger.info(f"Temporary download path (inside container): {download_dir}")
    logger.info(f"Quiet mode: {args.quiet}")
    logger.info(f"Use cookies: {args.use_cookies}")
    logger.info(f"Max download retries: {args.retry}")
    logger.info(f"Clean up download dir: {args.clean}")


    # --- Extract Stage ---
    download_successful = False
    final_download_path = None
    try:
        final_download_path = download_folder(
            folder_id=args.folder_id,
            output_dir=download_dir,
            quiet=args.quiet,
            use_cookies=args.use_cookies,
            retries=args.retry,
            logger=logger
        )
        logger.info(f"Download stage completed successfully.")
        download_successful = True # Mark download as successful
    except Exception as e:
        logger.critical(f"Download stage failed: {e}")
        # Exit with a non-zero status code to signal failure to the orchestrator
        sys.exit(1)

    # --- Load Stage ---
    # Proceed to upload only if download was successful
    upload_success = False
    if download_successful:
        # The Transform stage is implicitly handled by the data structure in the downloaded directory
        upload_success = upload_to_swift(
            local_directory=final_download_path,
            swift_container=args.swift_container,
            logger=logger
        )

        if not upload_success:
            logger.critical("Upload stage failed. Exiting.")
            # upload_to_swift logs specifics, just exit here
            sys.exit(1) # Exit with a non-zero status code

        logger.info("Upload stage completed successfully.")

    # --- Cleanup (of the temporary download directory inside the container) ---
    # Clean up the local download directory regardless of upload success if --clean is set
    if args.clean and final_download_path and os.path.exists(final_download_path):
        logger.info(f"Cleaning up local download directory: '{final_download_path}'")
        try:
            # Use ignore_errors=True to prevent failure if directory is already partially removed or has permission issues
            shutil.rmtree(final_download_path, ignore_errors=True)
            logger.info("Local download directory cleaned up.")
        except Exception as e:
            # Log the error, but don't exit as the main ETL process (download/upload) was successful
            logger.error(f"Failed to clean up local directory '{final_download_path}': {e}")
    elif args.clean:
         logger.info("Cleanup requested, but no download path was created or found.")
    else:
         logger.info("Cleanup not requested. Local download directory retained.")


    logger.info("--- ETL Pipeline Finished ---")
    # If we reached here, the download was successful, and the upload process completed (could have file errors, logged).
    # Exit with success code
    sys.exit(0)


if __name__ == "__main__":
    main()