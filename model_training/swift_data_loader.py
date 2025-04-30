# model_training/swift_data_loader.py
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import json
import os
import io # To read string content as a file
import structlog # Using structlog
from typing import Dict, Any, List, Optional, Tuple
import openstack # For Swift interaction
from openstack.exceptions import ResourceNotFound, SDKException # Import exceptions
import tqdm # For progress bars during data loading

# Get structlog logger instance
# Ensure structlog is configured early in the entrypoint script (train_job.py)
logger = structlog.get_logger(__name__)

# Define the required weather columns and their expected order.
# This list MUST match the column names and order *after* cleaning
# in your original local data loading process that was used to train the model.
# Replace with the actual list of weather feature column names:
WEATHER_FEATURE_COLUMNS = [
    'TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'AWND', 'WDF2', 'WDF5',
    'WSF2', 'WSF5', 'PGTM', 'PSUN', 'TSUN', 'EVAP', 'MNPN', 'MXPN',
    'DAPR', 'DASF', 'MDPR', 'MDSF', 'WT01', 'WT02', 'WT03', 'WT04',
    'WT05', 'WT06', 'WT07', 'WT08', 'WT09', 'WT10', 'WT11', 'WT12',
    'WT13', 'WT14', 'WT15', 'WT16', 'WT17', 'WT18', 'WT19', 'WT21',
    'WT22'
] # <-- Update this list with your actual feature columns and order!


class SwiftCropYieldDataset(Dataset):
    """
    Loads crop yield data and weather time series from OpenStack Swift.

    Data structure expected in Swift container:
    container_name/
    ├── FIPS_CODE_1/
    │   ├── crop_name.json (contains yearly yield data for this FIPS)
    │   ├── YEAR_1/
    │   │   └── WeatherTimeSeriesYEAR_1.csv (daily weather data)
    │   ├── YEAR_2/
    │   │   └── WeatherTimeSeriesYEAR_2.csv
    │   └── ...
    └── FIPS_CODE_2/
        └── ...
    """
    def __init__(self, swift_container_name: str, crop_name: str, swift_conn: openstack.connection.Connection, transform=None):
        """
        Initializes the dataset by listing objects in Swift and processing metadata.

        Args:
            swift_container_name (str): The name of the Swift container.
            crop_name (str): The name of the crop (e.g., "corn").
            swift_conn (openstack.connection.Connection): An active OpenStack Swift connection object.
            transform (callable, optional): Optional transform to be applied to weather data.
        """
        self.swift_container_name = swift_container_name
        self.crop_name = crop_name.lower()
        self.swift_conn = swift_conn # Store the provided connection
        self.transform = transform

        # Stores samples as tuples: (weather_tensor, yield_target, fips_id, year_str)
        self._samples_with_year = []
        # Dictionary to store indices of samples grouped by year (index refers to _samples_with_year list)
        self._samples_by_year_indices: Dict[str, List[int]] = {}
        # Mappings for FIPS codes
        self.fips_to_id: Dict[str, int] = {}
        self.id_to_fips: Dict[int, str] = {}
        self._next_fips_id = 0 # Counter for assigning unique integer IDs

        logger.info(f"Initializing Swift Crop Yield Dataset from container '{swift_container_name}' for crop '{self.crop_name}'")

        # --- Step 1: List objects and build metadata index ---
        # We need to find all FIPS/Year combinations that have BOTH yield JSON and weather CSV.
        fips_data_index: Dict[str, Dict[str, Dict[str, str]]] = {} # {fips: {year: {'weather_obj': name, 'yield_obj': name}}}

        try:
            logger.info(f"Listing objects in Swift container '{self.swift_container_name}'...")
            # Use delimiter='/' to list 'folders' (FIPS codes)
            # list_objects is lazy, converting to list fetches all results
            fips_folders_list = list(self.swift_conn.object_store.list_objects(self.swift_container_name, delimiter='/', limit=None))
            # common_prefix represents directories ending in '/'
            fips_prefixes = [obj.common_prefix for obj in fips_folders_list if obj.common_prefix and '/' in obj.common_prefix]

            if not fips_prefixes:
                 logger.warning("No FIPS folders found in the Swift container.")
                 # Proceed with empty dataset, or raise error? Proceed for now.

            logger.info(f"Found {len(fips_prefixes)} potential FIPS folders.")

            for fips_prefix in tqdm.tqdm(sorted(fips_prefixes), desc="Indexing Swift Objects"): # Sort for deterministic order
                # Remove trailing slash to get FIPS code
                fips_code = fips_prefix.rstrip('/')

                # Assign FIPS ID immediately for the index
                if fips_code not in self.fips_to_id:
                     self.fips_to_id[fips_code] = self._next_fips_id
                     self.id_to_fips[self._next_fips_id] = fips_code
                     self._next_fips_id += 1
                # fips_id = self.fips_to_id[fips_code] # We'll get this later when processing samples

                # List objects within this FIPS prefix recursively
                fips_objects = list(self.swift_conn.object_store.list_objects(self.swift_container_name, prefix=fips_prefix, delimiter=None, limit=None)) # No delimiter for recursive list

                yield_json_object_name = None
                year_weather_objects: Dict[str, str] = {} # {year_str: object_name}

                for obj in fips_objects:
                    obj_name = obj.name
                    # Check for yield JSON (e.g., FIPS/crop_name.json)
                    # Ensure it's directly in the FIPS folder, not a subfolder
                    if obj_name == f"{fips_code}/{self.crop_name}.json":
                        yield_json_object_name = obj_name

                    # Check for weather CSVs (e.g., FIPS/YEAR/WeatherTimeSeriesYEAR.csv)
                    parts = obj_name.split('/')
                    # Object name must have 3 parts: FIPS/YEAR/FILE
                    if len(parts) == 3 and parts[0] == fips_code and parts[2] == f"WeatherTimeSeries{parts[1]}.csv":
                        year_str = parts[1]
                        # Basic year validation (check if it looks like a year)
                        if year_str.isdigit() and len(year_str) == 4:
                             year_weather_objects[year_str] = obj_name
                        else:
                             logger.debug(f"Skipping object '{obj_name}': Year part '{year_str}' does not look like a 4-digit year.")


                # If yield JSON exists for this FIPS and we found weather data for some years
                if yield_json_object_name and year_weather_objects:
                    fips_data_index[fips_code] = {}
                    # Store object names for each valid year found
                    for year_str, weather_obj_name in year_weather_objects.items():
                         fips_data_index[fips_code][year_str] = {
                             'weather_obj': weather_obj_name,
                             'yield_obj': yield_json_object_name # Same yield JSON for all years of this FIPS
                         }
                    logger.debug(f"Indexed data for FIPS {fips_code}: found {len(year_weather_objects)} years with weather and yield JSON.")
                elif yield_json_object_name:
                     logger.debug(f"Skipping FIPS {fips_code}: Found yield JSON but no weather CSVs in expected YEAR subfolders.")
                else:
                     logger.debug(f"Skipping FIPS {fips_code}: No yield JSON found for crop '{self.crop_name}'.")


        except SDKException as e:
             logger.critical(f"OpenStack SDK error during Swift object listing/indexing: {e}", exc_info=True)
             raise ConnectionError(f"Failed to list Swift objects: {e}") from e # Raise connection error
        except Exception as e:
             logger.critical(f"An unexpected error occurred during Swift object listing/indexing: {e}", exc_info=True)
             raise RuntimeError(f"Error indexing Swift data: {e}") from e # Raise generic runtime error

        logger.info(f"Finished indexing Swift objects. Data candidates found for {len(fips_data_index)} FIPS codes across various years.")

        # --- Step 2: Download and Process Data for Each Sample ---
        # Iterate through the indexed data, download, process, and store samples in memory.
        processed_sample_count = 0 # This will be the index in _samples_with_year list
        # Sort FIPS codes and years for deterministic loading order
        sorted_fips = sorted(fips_data_index.keys())

        logger.info("Starting download and processing of Swift data...")
        for fips_code in tqdm.tqdm(sorted_fips, desc="Processing Swift Data"):
            fips_id = self.fips_to_id[fips_code] # Get the assigned ID

            # Sort years for this FIPS code for deterministic order
            sorted_years = sorted(fips_data_index[fips_code].keys())

            # Download yield JSON once per FIPS
            yield_data = None
            yield_obj_name_for_fips = None
            if sorted_years: # Only download if there are years to process for this FIPS
                 yield_obj_name_for_fips = fips_data_index[fips_code][sorted_years[0]]['yield_obj'] # Get the yield obj name
                 try:
                     yield_content_bytes = self.swift_conn.object_store.download_object(self.swift_container_name, yield_obj_name_for_fips, stream=False)
                     yield_content_str = yield_content_bytes.decode('utf-8') # Assuming UTF-8
                     yield_data = json.loads(yield_content_str)
                     # logger.debug(f"Downloaded yield JSON for FIPS {fips_code}", obj=yield_obj_name_for_fips)
                 except ResourceNotFound:
                      logger.error(f"Yield JSON object '{yield_obj_name_for_fips}' not found during processing for FIPS {fips_code}. Skipping samples for this FIPS.", obj=yield_obj_name_for_fips)
                      yield_data = None # Ensure yield_data is None if download fails
                 except Exception as e:
                      logger.error(f"Error processing yield data object '{yield_obj_name_for_fips}' for FIPS {fips_code}: {e}. Skipping samples for this FIPS.", exc_info=True)
                      yield_data = None # Ensure yield_data is None if processing fails


            # Process weather data for each year
            for year_str in sorted_years:
                sample_info = fips_data_index[fips_code][year_str]
                weather_obj_name = sample_info['weather_obj']

                # Skip if yield data failed to load for this FIPS
                if yield_data is None:
                     continue # Move to the next year/fips

                # --- Extract Yield for this specific year ---
                yield_value = None
                if year_str in yield_data and 'yield' in yield_data[year_str]:
                    try:
                         yield_value = float(yield_data[year_str]['yield'])
                         # logger.debug(f"Extracted yield {yield_value} for {fips_code}/{year_str}")
                    except (ValueError, TypeError):
                         logger.warning(f"Yield value for year {year_str} in JSON is not a valid number for FIPS {fips_code}. Skipping sample.")
                         continue # Skip if yield value is invalid
                else:
                     logger.debug(f"Yield data not found for year {year_str} in JSON for FIPS {fips_code}. Skipping sample.")
                     continue # Skip if yield data for the specific year is missing


                # --- Download and Process Weather CSV ---
                weather_tensor = None
                try:
                    # Download weather object
                    weather_content_bytes = self.swift_conn.object_store.download_object(self.swift_container_name, weather_obj_name, stream=False)
                    weather_content_str = weather_content_bytes.decode('utf-8') # Assuming UTF-8
                    csv_file_like = io.StringIO(weather_content_str)

                    # Read and process CSV (mirroring CropYieldDataset logic)
                    df = pd.read_csv(csv_file_like)

                    # Ensure 'Month' column exists and is numeric
                    if 'Month' not in df.columns:
                         logger.warning(f"Weather data missing 'Month' column for {fips_code}/{year_str}. Skipping.", obj=weather_obj_name)
                         continue

                    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                    df.dropna(subset=['Month'], inplace=True) # Drop rows where Month is NaN

                    # Filter to growing season (April to October)
                    df_season = df[(df['Month'] >= 4) & (df['Month'] <= 10)].copy()

                    if df_season.empty:
                         logger.debug(f"No weather data in Apr-Oct season for {fips_code}/{year_str}. Skipping.", obj=weather_obj_name)
                         continue

                    # Drop non-feature columns like Year, Month, Day.
                    cols_to_drop = ['Year', 'Month', 'Day']
                    existing_cols_to_drop = [col for col in df_season.columns if col in cols_to_drop]
                    df_season = df_season.drop(columns=existing_cols_to_drop, errors='ignore')

                    # Ensure all remaining columns have numeric types
                    for col in df_season.columns:
                         df_season[col] = pd.to_numeric(df_season[col], errors='coerce')

                    # Drop any columns that became all NaN after conversion
                    df_season.dropna(axis=1, how='all', inplace=True)

                    # Check if expected features remain and reorder
                    if len(WEATHER_FEATURE_COLUMNS) > 0: # Check if we have a defined list of expected features
                        # Keep only the columns in WEATHER_FEATURE_COLUMNS that exist in df_season AND are actual features
                        # Use a robust check against potential leftover non-feature columns even after dropping common ones
                        current_feature_cols = [col for col in df_season.columns if col in WEATHER_FEATURE_COLUMNS]

                        if len(current_feature_cols) != len(WEATHER_FEATURE_COLUMNS):
                             missing_cols = [col for col in WEATHER_FEATURE_COLUMNS if col not in current_feature_cols]
                             extra_cols = [col for col in df_season.columns if col not in WEATHER_FEATURE_COLUMNS and col not in cols_to_drop] # Consider cols_to_drop again
                             logger.warning(f"Feature mismatch for {fips_code}/{year_str}. Expected {len(WEATHER_FEATURE_COLUMNS)}, found {len(current_feature_cols)} relevant cols. Missing: {missing_cols}. Extra: {extra_cols}. Skipping sample.", obj=weather_obj_name)
                             continue # Skip if feature set doesn't match expectation

                        # Reorder the DataFrame columns
                        df_season_ordered = df_season[current_feature_cols].copy()
                    else:
                        # If WEATHER_FEATURE_COLUMNS is empty or not used, use current columns.
                        # This is less safe for model input consistency.
                        logger.warning("WEATHER_FEATURE_COLUMNS is not defined. Relying on DataFrame column order from source data. This may lead to consistency issues.")
                        df_season_ordered = df_season.copy() # Use all remaining columns


                    # Drop rows with any NaN values *after* feature selection/reordering
                    initial_rows = len(df_season_ordered)
                    df_season_ordered.dropna(axis=0, how='any', inplace=True)
                    if len(df_season_ordered) < initial_rows:
                        logger.debug(f"Dropped {initial_rows - len(df_season_ordered)} rows with NaN weather data for {fips_code}/{year_str}. Remaining: {len(df_season_ordered)}.", obj=weather_obj_name)
                        if df_season_ordered.empty:
                            logger.debug(f"No valid weather data rows left after dropping NaNs for {fips_code}/{year_str}. Skipping sample.", obj=weather_obj_name)
                            continue

                    # Convert the processed DataFrame to a PyTorch tensor
                    weather_tensor = torch.tensor(df_season_ordered.values, dtype=torch.float32)

                    # Final check: ensure tensor is not empty (can happen if sequence length becomes 0)
                    if weather_tensor.size(0) == 0:
                         logger.debug(f"Weather tensor is empty for {fips_code}/{year_str} after processing. Skipping sample.", obj=weather_obj_name)
                         continue


                except ResourceNotFound:
                     logger.warning(f"Weather object '{weather_obj_name}' not found during processing. Skipping sample.", obj=weather_obj_name)
                     continue # Skip this sample
                except pd.errors.EmptyDataError:
                     logger.debug(f"Weather CSV object '{weather_obj_name}' is empty. Skipping sample.", obj=weather_obj_name)
                     continue
                except Exception as e:
                    logger.error(f"Error processing weather data object '{weather_obj_name}': {e}. Skipping sample.", exc_info=True)
                    continue # Skip this sample on error


                # --- Store the Valid Sample ---
                # If we reached here, we have valid weather_tensor and yield_value
                yield_target = torch.tensor(yield_value, dtype=torch.float32) # Convert yield value to tensor

                self._samples_with_year.append((weather_tensor, yield_target, fips_id, year_str))

                # Store the index grouped by year
                if year_str not in self._samples_by_year_indices:
                    self._samples_by_year_indices[year_str] = []
                self._samples_by_year_indices[year_str].append(processed_sample_count) # Use the list index
                processed_sample_count += 1


        logger.info(f"\nFinished processing Swift data.")
        logger.info(f"Loaded {len(self._samples_with_year)} total samples for crop '{self.crop_name}'.")
        logger.info(f"Found {len(self.fips_to_id)} unique FIPS codes indexed (potential).") # Total FIPS found during indexing
        logger.info(f"Samples collected from {len(self._samples_by_year_indices)} years: {sorted(self._samples_by_year_indices.keys()) if self._samples_by_year_indices else 'None'}")

        # Final check on FIPS mapping - only keep IDs for FIPS that actually have samples loaded
        actual_fips_ids_with_samples = set(item[2] for item in self._samples_with_year)
        # Rebuild mappings based on actual samples
        self.fips_to_id = {fips: id for fips, id in self.fips_to_id.items() if id in actual_fips_ids_with_samples}
        # Rebuild id_to_fips based on the new fips_to_id
        self.id_to_fips = {id: fips for fips, id in self.fips_to_id.items()} # Correct way to rebuild
        logger.info(f"Adjusted FIPS mapping to {len(self.fips_to_id)} FIPS codes with actual samples loaded.")
        if len(self.fips_to_id) == 0 and len(self._samples_with_year) > 0:
             # This case should ideally not happen if indexing/mapping works correctly
             logger.error("FIPS mapping is empty but samples were loaded. Potential mapping error.")


        # Check if any samples were loaded at all
        if len(self._samples_with_year) == 0:
             logger.warning("No samples were loaded after processing data from Swift.")
             # Depending on strictness, you might raise an error here
             # For now, an empty dataset is possible, but training will likely fail later


    def __len__(self):
        """Returns the total number of samples loaded into memory."""
        return len(self._samples_with_year)

    def __getitem__(self, idx):
        """Retrieves a single sample by index. Returns weather, yield, fips_id."""
        # Retrieve the sample tuple - it's already processed and tensored
        weather_tensor, yield_target, fips_id, _ = self._samples_with_year[idx]
        # Apply transform only to the weather tensor if specified
        if self.transform:
            weather_tensor = self.transform(weather_tensor)
        # Return the sample components needed for the model and collate_fn
        return weather_tensor, yield_target, fips_id

    def get_fips_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Returns the dictionaries mapping FIPS codes to IDs and vice-versa for FIPS with samples."""
        return self.fips_to_id, self.id_to_fips

    def get_num_fips(self) -> int:
        """Returns the number of unique FIPS codes that have samples loaded."""
        return len(self.fips_to_id)

    def get_years(self) -> List[str]:
        """Returns a sorted list of all unique years present in the loaded samples."""
        return sorted(self._samples_by_year_indices.keys())

    def get_sample_indices_by_year(self) -> Dict[str, List[int]]:
        """Returns a dictionary mapping year (str) to a list of sample indices for that year."""
        return self._samples_by_year_indices

    # You might want a method to get metadata for logging/debugging
    def get_sample_metadata(self, idx) -> Tuple[int, str]:
        """Returns (fips_id, year_str) for a sample index."""
        _, _, fips_id, year_str = self._samples_with_year[idx]
        return fips_id, year_str