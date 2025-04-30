import pandas as pd
# from pathlib import Path # No longer needed for local paths
import io # Needed to read string content as a file
import structlog
from typing import Dict, Any, List, Optional
import openstack # Import openstack sdk
from openstack.exceptions import ResourceNotFound, SDKException # Import specific exceptions for error handling

# Import settings
from .config import settings

# Get structlog logger instance
logger = structlog.get_logger(__name__)

# Global variable for the Swift connection
# This will be initialized in main.py's startup event
swift_conn: Optional[openstack.connection.Connection] = None

def initialize_swift_connection():
    """Initializes the global OpenStack Swift connection."""
    global swift_conn
    if swift_conn is None:
        logger.info("Initializing OpenStack Swift connection...")
        try:
            swift_conn = openstack.connect(
                auth_url=str(settings.openstack.auth_url), # Convert HttpUrl to str
                project_name=settings.openstack.project_name,
                project_domain_name=settings.openstack.project_domain_name,
                username=settings.openstack.username,
                user_domain_name=settings.openstack.user_domain_name,
                password=settings.openstack.password,
                region_name=settings.openstack.region_name,
                # Optionally add app_name and app_version for logging/debugging on OpenStack side
                # app_name='agri-yield-feature-service',
                # app_version='1.0.0',
            )
            # Verify connectivity (optional, but good practice)
            # Try listing containers or checking if the target container exists
            try:
                 swift_conn.object_store.get_container(settings.openstack.swift_container_name)
                 logger.info("OpenStack Swift connection successful and container found.", container=settings.openstack.swift_container_name)
            except ResourceNotFound:
                 logger.critical("OpenStack Swift container not found during connection initialization.", container=settings.openstack.swift_container_name)
                 swift_conn.close() # Close the connection if container check fails
                 swift_conn = None
                 raise ConnectionError(f"Swift container '{settings.openstack.swift_container_name}' not found.") # Use a standard error
            except Exception as e:
                 logger.critical("Failed to verify Swift container during connection initialization.", container=settings.openstack.swift_container_name, error=str(e), exc_info=True)
                 swift_conn.close()
                 swift_conn = None
                 raise ConnectionError(f"Failed to verify Swift connection or container: {e}")


        except Exception as e:
            logger.critical("Failed to initialize OpenStack Swift connection.", error=str(e), exc_info=True)
            swift_conn = None # Ensure it's None if connection failed
            raise ConnectionError(f"Failed to initialize OpenStack Swift connection: {e}") # Use a standard error

def close_swift_connection():
    """Closes the global OpenStack Swift connection."""
    global swift_conn
    if swift_conn is not None:
        logger.info("Closing OpenStack Swift connection...")
        try:
            swift_conn.close()
            logger.info("OpenStack Swift connection closed.")
        except Exception as e:
            logger.error("Error closing OpenStack Swift connection.", error=str(e), exc_info=True)
        finally:
            swift_conn = None # Ensure it's None after trying to close


# --- Modify the data loading function ---
def load_and_process_weather_features(fips_code: str, year: int) -> Optional[List[Dict[str, Any]]]:
    """
    Loads and processes weather data for a given FIPS and year from OpenStack Swift,
    filtering for the Apr-Oct season.
    Mirrors the data cleaning logic in CropYieldDataset.

    Args:
        fips_code (str): The FIPS code of the county.
        year (int): The year of the data.

    Returns:
        Optional[List[Dict[str, Any]]]: A list of dictionaries, where each dict is a day's
                                       weather features for the April-October season,
                                       cleaned of NaNs. Returns an empty list if the file
                                       is found but has no valid season data, or None if
                                       the object (file) is not found in Swift.
                                       Raises other exceptions for processing errors.
    Raises:
         ConnectionError: If the Swift connection is not initialized.
         SDKException: For other OpenStack SDK errors during download.
         Exception: For other unexpected errors during processing.
    """
    if swift_conn is None:
        logger.error("Swift connection not initialized.")
        raise ConnectionError("Swift connection is not available.")

    year_str = str(year) # Ensure year is a string
    # Construct the object name in Swift
    # Assuming the structure is FIPS_CODE/YEAR/WeatherTimeSeriesYEAR.csv
    object_name = f"{fips_code}/{year_str}/WeatherTimeSeries{year_str}.csv"
    container_name = settings.openstack.swift_container_name

    log = logger.bind(fips=fips_code, year=year_str, container=container_name, object=object_name)
    log.debug("Attempting to load weather data from Swift")

    try:
        # Download the object content
        # download_object returns bytes
        object_content_bytes = swift_conn.object_store.download_object(container_name, object_name, stream=False) # stream=False downloads fully to memory

        # Decode bytes to string (assuming UTF-8 encoding) and wrap in StringIO
        object_content_str = object_content_bytes.decode('utf-8')
        csv_file_like = io.StringIO(object_content_str)

        # Read the CSV content into a pandas DataFrame
        # pd.read_csv can read directly from a file-like object
        df = pd.read_csv(csv_file_like)

        # --- Data Processing (Mirroring CropYieldDataset) ---
        # Filter to growing season (April to October)
        # Ensure 'Month' column exists and is numeric before filtering
        if 'Month' not in df.columns:
             log.error("Weather data is missing 'Month' column.")
             # Treat as a data format error
             raise ValueError("'Month' column is missing in weather data from Swift.")

        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df.dropna(subset=['Month'], inplace=True) # Drop rows where Month is NaN after coerce

        df_season = df[(df['Month'] >= 4) & (df['Month'] <= 10)].copy() # Use .copy()

        if df_season.empty:
            log.warning("No weather data in Apr-Oct season after filtering")
            return [] # Return empty list if season data is missing

        # Drop non-weather columns like Year, Month, Day.
        cols_to_drop = ['Year', 'Month', 'Day'] # Don't include 'Date' unless you explicitly add it before dropping
        existing_cols_to_drop = [col for col in df_season.columns if col in cols_to_drop]

        df_season = df_season.drop(columns=existing_cols_to_drop, errors='ignore')

        # Ensure all remaining columns have numeric types
        for col in df_season.columns:
             df_season[col] = pd.to_numeric(df_season[col], errors='coerce')

        # Drop any columns that became all NaN after conversion
        df_season.dropna(axis=1, how='all', inplace=True)

        # Check if any weather features remain after dropping columns
        if df_season.shape[1] == 0:
            log.warning("No weather features remaining after cleaning columns")
            return [] # Return empty list if no feature columns remain

        # Drop rows with any NaN values (missing days within the season)
        initial_rows = len(df_season)
        df_season.dropna(axis=0, how='any', inplace=True)
        if len(df_season) < initial_rows:
             log.warning("Dropped rows with NaN weather data", dropped_count=initial_rows - len(df_season), remaining_count=len(df_season))
             if df_season.empty:
                  log.warning("No valid weather data rows left after dropping NaNs")
                  return [] # Return empty list if no valid rows left


        # --- End Data Processing ---

        # Convert DataFrame to list of dictionaries (one dict per row/day)
        weather_list_of_dicts = df_season.to_dict('records')

        log.debug("Successfully loaded and processed weather data from Swift", row_count=len(weather_list_of_dicts), features_count=df_season.shape[1])
        return weather_list_of_dicts

    except ResourceNotFound:
        # Swift object (file) not found in the container
        log.warning("Swift object not found")
        return None # Explicitly return None if the file doesn't exist

    except pd.errors.EmptyDataError:
        log.warning("CSV data from Swift object is empty")
        return [] # Treat empty CSV content as no data for the season

    except Exception as e:
        # Catch any other exceptions during download, decoding, or processing (SDKException, pandas errors, etc.)
        log.error("Error processing weather data from Swift", error=str(e), exc_info=True)
        # Re-raise the exception so the main app can catch and return 500
        raise