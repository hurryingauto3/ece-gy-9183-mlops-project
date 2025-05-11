import pandas as pd
import numpy as np # Import numpy

# from pathlib import Path # No longer needed for local paths
import io  # Needed to read string content as a file
import structlog
from typing import Dict, Any, List, Optional
import openstack  # Import openstack sdk
from openstack.exceptions import (
    ResourceNotFound,
    SDKException,
)  # Import specific exceptions for error handling

# Import settings
from .config import settings

# Get structlog logger instance
logger = structlog.get_logger(__name__)

# Global variable for the Swift connection
# This will be initialized in main.py's startup event
swift_conn: Optional[openstack.connection.Connection] = None

# Placeholder for potential crop-specific data paths or feature sets
# This might be loaded from config or be more complex
CROP_SPECIFIC_CONFIG = {
    "default": {"weather_file_suffix": "WeatherTimeSeries"},
    # "corn": {"weather_file_suffix": "WeatherTimeSeries_Corn"}, # Example
    # "soy": {"weather_file_suffix": "WeatherTimeSeries_Soy"},   # Example
}

def initialize_swift_connection() -> None:
    """Initialize the global OpenStack Swift connection and confirm the container exists."""
    global swift_conn
    if swift_conn:          # already connected
        return

    swift_conn = None       # reset on every call

    # ── build minimal, env-free auth payload ─────────────────────────────
    params = {
        "auth_url": str(settings.openstack.auth_url),
        "auth_type": settings.openstack.auth_type,           # "v3applicationcredential"
        "application_credential_id": settings.openstack.application_credential_id,
        "application_credential_secret": settings.openstack.application_credential_secret,
        "region_name": settings.openstack.region_name,
        "load_envvars": False,   # ← ignore any host OS_* variables
    }

    try:
        swift_conn = openstack.connect(**params)
    except Exception as exc:
        logger.critical("Swift auth failure during initialization. Will attempt to proceed in offline/dummy mode if applicable.", error=str(exc), exc_info=True)
        swift_conn = None # Allow service to start, swift_conn remains None
        # Do not raise ConnectionError here to allow dummy mode
        return # Exit initialization

    # ── verify the container exists (openstacksdk ≥ 2.0) ─────────────────
    # This part only runs if swift_conn was successfully created above
    try:
        # head request; raises ResourceNotFound if absent
        swift_conn.object_store.get_container_metadata(settings.swift_container_name)
        logger.info("Swift container found", container=settings.swift_container_name)

    except ResourceNotFound:
        logger.critical("Swift container not found. Will attempt to proceed in offline/dummy mode if applicable.", container=settings.swift_container_name)
        if swift_conn: swift_conn.close()
        swift_conn = None # Allow service to start
        # Do not raise ConnectionError
    except Exception as exc:
        logger.critical("Error verifying Swift container. Will attempt to proceed in offline/dummy mode if applicable.", error=str(exc), exc_info=True)
        if swift_conn: swift_conn.close()
        swift_conn = None # Allow service to start
        # Do not raise ConnectionError
    
def close_swift_connection():
    """Closes the global OpenStack Swift connection."""
    global swift_conn
    if swift_conn is not None:
        logger.info("Closing OpenStack Swift connection...")
        try:
            swift_conn.close()
            logger.info("OpenStack Swift connection closed.")
        except Exception as e:
            logger.error(
                "Error closing OpenStack Swift connection.", error=str(e), exc_info=True
            )
        finally:
            swift_conn = None  # Ensure it's None after trying to close


# --- Modify the data loading function ---
def load_and_process_weather_features(
    fips_code: str, year: int, cut_off_date: Optional[str] = None, crop: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Loads and processes weather data for a given FIPS, year, and optional cut-off date from OpenStack Swift,
    filtering for the Apr-Oct season or up to the cut-off date within that season.
    Optionally considers crop type for data fetching (currently placeholder).

    Args:
        fips_code (str): The FIPS code of the county.
        year (int): The year of the data.
        cut_off_date (Optional[str]): The cut-off date in 'YYYY-MM-DD' format. If provided,
                                      data is filtered up to this date within the season.
        crop (Optional[str]): The type of crop. (Currently placeholder for future use, e.g., fetching different data).

    Returns:
        Optional[List[Dict[str, Any]]]: A list of dictionaries, where each dict is a day's
                                       weather features, cleaned of NaNs. Returns an empty list
                                       if the file is found but has no valid data matching criteria,
                                       or None if the object (file) is not found in Swift.
    Raises:
         ConnectionError: If the Swift connection is not initialized AND dummy mode is not active.
         SDKException: For other OpenStack SDK errors during download.
         Exception: For other unexpected errors during processing.
    """
    # Check for dummy FIPS code first or if swift_conn is None (implying offline/dummy mode)
    # Define a few dummy weather features names. Ensure these match what a real model might expect.
    dummy_feature_names = ["temp_max_c", "temp_min_c", "precipitation_mm", "solar_rad_mj_m2"]
    num_dummy_days = 90 # Number of days of dummy data

    if fips_code == "00000" or fips_code.upper() == "DUMMY" or swift_conn is None:
        logger.info(f"Entering dummy data mode for FIPS: {fips_code}, Year: {year}, Crop: {crop}, Cut-off: {cut_off_date}. Swift available: {swift_conn is not None}")
        mock_weather_data = []
        start_date = pd.Timestamp(f"{year}-04-01") # Start of typical season
        
        # Determine end date for dummy data generation
        actual_end_date = pd.Timestamp(f"{year}-10-31") # End of typical season
        if cut_off_date:
            try:
                parsed_cut_off = pd.to_datetime(cut_off_date)
                if parsed_cut_off < actual_end_date:
                    actual_end_date = parsed_cut_off
            except ValueError:
                logger.warning(f"Invalid cut_off_date format '{cut_off_date}' in dummy mode, using full season.")
        
        # Ensure start_date is not after actual_end_date
        if start_date > actual_end_date:
            logger.warning(f"Start date {start_date} is after end date {actual_end_date} in dummy mode, returning no data.")
            return []
            
        current_date = start_date
        days_generated = 0
        while current_date <= actual_end_date and days_generated < num_dummy_days:
            day_features = {
                name: round(np.random.uniform(0, 30) if "temp" in name else np.random.uniform(0,10), 2) 
                for name in dummy_feature_names
            }
            # Add Year, Month, Day for potential internal processing before they are dropped
            # Although our current processing drops them based on name, not position
            day_features["Year"] = current_date.year
            day_features["Month"] = current_date.month
            day_features["Day"] = current_date.day
            mock_weather_data.append(day_features)
            current_date += pd.Timedelta(days=1)
            days_generated += 1
        
        # The rest of the processing (dropping Year/Month/Day, numeric conversion, NaN handling) 
        # will be applied to this dummy data if we pass it through the same pandas logic.
        # For simplicity here, let's assume the dummy data is already in the final desired format 
        # (list of dicts with only numeric weather features).
        # However, to be robust, we should process it minimally.
        
        if not mock_weather_data:
            return []

        df = pd.DataFrame(mock_weather_data)
        # Drop non-weather columns (Year, Month, Day)
        cols_to_drop = ["Year", "Month", "Day"]
        existing_cols_to_drop = [col for col in df.columns if col in cols_to_drop]
        df = df.drop(columns=existing_cols_to_drop, errors="ignore")
        
        # Ensure all remaining columns have numeric types (dummy data should be already)
        # for col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        # df.dropna(axis=1, how="all", inplace=True)
        # if df.empty or df.shape[1] == 0: return []
        # df.dropna(axis=0, how="any", inplace=True)
        # if df.empty: return []
        logger.info(f"Generated {len(df)} days of dummy weather data.")
        return df.to_dict("records")

    # Original logic if not in dummy mode and swift_conn is available
    if swift_conn is None: # Should have been caught by dummy mode if fips was not DUMMY, but as safeguard
        logger.error("Swift connection not initialized and not in recognized dummy mode.")
        # Depending on strictness, could raise ConnectionError or return empty/None
        # For testing, returning empty might be preferable to crashing service if DUMMY FIPS wasn't used.
        return [] 

    year_str = str(year)
    
    # Placeholder: Determine data source based on crop
    # For now, uses a default, but this could select different files or parsing
    # crop_config = CROP_SPECIFIC_CONFIG.get(crop.lower() if crop else "default", CROP_SPECIFIC_CONFIG["default"])
    # weather_file_name_base = crop_config["weather_file_suffix"]
    # For simplicity, assuming crop doesn't change the filename for now.
    weather_file_name_base = "WeatherTimeSeries"
    
    # Construct the object name in Swift
    # Assuming the structure is FIPS_CODE/YEAR/WeatherTimeSeriesYEAR.csv
    object_name = f"{fips_code}/{year_str}/{weather_file_name_base}{year_str}.csv"
    container_name = settings.swift_container_name

    log = logger.bind(
        fips=fips_code, year=year_str, container=container_name, object=object_name, crop=crop, cut_off_date=cut_off_date
    )
    log.debug("Attempting to load weather data from Swift")

    try:
        # Download the object content
        # download_object returns bytes
        object_content_bytes = swift_conn.object_store.download_object(
            container_name, object_name, stream=False
        )  # stream=False downloads fully to memory

        # Decode bytes to string (assuming UTF-8 encoding) and wrap in StringIO
        object_content_str = object_content_bytes.decode("utf-8")
        csv_file_like = io.StringIO(object_content_str)

        # Read the CSV content into a pandas DataFrame
        # pd.read_csv can read directly from a file-like object
        df = pd.read_csv(csv_file_like)

        # --- Data Processing (Mirroring CropYieldDataset) ---
        # Convert Year, Month, Day to datetime for filtering
        if not all(col in df.columns for col in ["Year", "Month", "Day"]):
            log.error("Weather data is missing Year, Month, or Day column(s).")
            raise ValueError("Year, Month, or Day column(s) are missing in weather data.")

        df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

        # Filter to growing season (April to October)
        df_season = df[(df["date"].dt.month >= 4) & (df["date"].dt.month <= 10)].copy()

        if df_season.empty:
            log.warning("No weather data in Apr-Oct season before cut-off date filter")
            return []

        # Apply cut-off date filter if provided
        if cut_off_date:
            try:
                parsed_cut_off_date = pd.to_datetime(cut_off_date)
                df_season = df_season[df_season["date"] <= parsed_cut_off_date].copy()
                log.debug(f"Applied cut-off date: {cut_off_date}")
            except ValueError:
                log.error(f"Invalid cut_off_date format: {cut_off_date}. Should be YYYY-MM-DD.")
                # Or raise an error, for now, we'll ignore invalid date and proceed without this filter
                pass # Or raise HTTPException in main.py based on a specific error type here

        if df_season.empty:
            log.warning("No weather data in Apr-Oct season after cut-off date filter")
            return []  # Return empty list if season data is missing or filtered out

        # Drop non-weather columns like Year, Month, Day, and the created 'date'
        cols_to_drop = [
            "Year",
            "Month",
            "Day",
            "date", # also drop the created date column
        ]
        existing_cols_to_drop = [
            col for col in df_season.columns if col in cols_to_drop
        ]

        df_season = df_season.drop(columns=existing_cols_to_drop, errors="ignore")

        # Ensure all remaining columns have numeric types
        for col in df_season.columns:
            df_season[col] = pd.to_numeric(df_season[col], errors="coerce")

        # Drop any columns that became all NaN after conversion
        df_season.dropna(axis=1, how="all", inplace=True)

        # Check if any weather features remain after dropping columns
        if df_season.shape[1] == 0:
            log.warning("No weather features remaining after cleaning columns")
            return []  # Return empty list if no feature columns remain

        # Drop rows with any NaN values (missing days within the season)
        initial_rows = len(df_season)
        df_season.dropna(axis=0, how="any", inplace=True)
        if len(df_season) < initial_rows:
            log.warning(
                "Dropped rows with NaN weather data",
                dropped_count=initial_rows - len(df_season),
                remaining_count=len(df_season),
            )
            if df_season.empty:
                log.warning("No valid weather data rows left after dropping NaNs")
                return []  # Return empty list if no valid rows left

        # --- End Data Processing ---

        # Convert DataFrame to list of dictionaries (one dict per row/day)
        weather_list_of_dicts = df_season.to_dict("records")

        log.debug(
            "Successfully loaded and processed weather data from Swift",
            row_count=len(weather_list_of_dicts),
            features_count=df_season.shape[1],
        )
        return weather_list_of_dicts

    except ResourceNotFound:
        # Swift object (file) not found in the container
        log.warning("Swift object not found")
        return None  # Explicitly return None if the file doesn't exist

    except pd.errors.EmptyDataError:
        log.warning("CSV data from Swift object is empty")
        return []  # Treat empty CSV content as no data for the season

    except Exception as e:
        # Catch any other exceptions during download, decoding, or processing (SDKException, pandas errors, etc.)
        log.error(
            "Error processing weather data from Swift", error=str(e), exc_info=True
        )
        # Re-raise the exception so the main app can catch and return 500
        raise
