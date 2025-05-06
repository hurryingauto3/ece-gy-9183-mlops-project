import os
import json
import pandas as pd
from openstack import connection
from pathlib import Path
import logging
from datetime import datetime

# Setup logger
logger = logging.getLogger("data_processing_pipeline")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Block Storage Directory Path (This could be a mounted path like /mnt/block)
block_storage_dir = "/mnt/block/data_lake"  # Adjust as per your block storage mount

# Swift Connection Setup
def connect_to_swift():
    conn = connection.Connection(
        auth_url=os.getenv("OS_AUTH_URL"),
        project_name=os.getenv("OS_PROJECT_NAME"),
        username=os.getenv("OS_USERNAME"),
        password=os.getenv("OS_PASSWORD"),
        region_name=os.getenv("OS_REGION_NAME"),
        user_domain_name=os.getenv("OS_USER_DOMAIN_NAME"),
        project_domain_name=os.getenv("OS_PROJECT_DOMAIN_NAME")
    )
    return conn

# Download data from Swift
def download_data_from_swift(conn, container_name, object_name, local_path):
    """Download file from Swift to local directory."""
    logger.info(f"Downloading {object_name} from Swift container {container_name}")
    try:
        conn.object_store.download_object(container_name, object_name, destination=local_path)
        logger.info(f"Successfully downloaded {object_name} to {local_path}")
    except Exception as e:
        logger.error(f"Error downloading {object_name}: {e}")
        raise

# Preprocess HRRR weather data
def preprocess_hrrr_data(weather_data_path):
    """Preprocess HRRR weather data (remove missing values, handle columns)."""
    try:
        df = pd.read_csv(weather_data_path)
        # Filter only relevant columns (e.g., weather variables)
        df_cleaned = df[['Year', 'Month', 'Day', 'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)', 
                         'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)']]
        df_cleaned.dropna(inplace=True)  # Drop rows with missing values
        return df_cleaned
    except Exception as e:
        logger.error(f"Error preprocessing HRRR data: {e}")
        raise

# Preprocess USDA crop yield data
def preprocess_usda_data(crop_data_path):
    """Preprocess USDA crop yield data and return a dictionary with crop types."""
    try:
        df = pd.read_csv(crop_data_path)
        # Filter relevant columns (state, county, crop yield, etc.)
        df_filtered = df[['year', 'state_name', 'county_name', 'commodity_desc', 
                          'PRODUCTION, MEASURED IN BU', 'YIELD, MEASURED IN BU / ACRE']]

        # Create a dictionary to map FIPS codes to yield data, categorized by crop type
        yield_data_by_crop = {}

        for _, row in df_filtered.iterrows():
            crop = row['commodity_desc'].lower()
            key = (row['state_name'], row['county_name'], row['year'])
            if crop not in yield_data_by_crop:
                yield_data_by_crop[crop] = {}
            yield_data_by_crop[crop][key] = {
                'production': row['PRODUCTION, MEASURED IN BU'],
                'yield': row['YIELD, MEASURED IN BU / ACRE']
            }
        return yield_data_by_crop
    except Exception as e:
        logger.error(f"Error preprocessing USDA data: {e}")
        raise

# Save preprocessed weather data to block storage
def save_weather_to_block_storage(fips_code, year, weather_df):
    """Save the preprocessed weather data to block storage."""
    year_dir = Path(block_storage_dir) / f"{fips_code}" / f"{year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    weather_file_path = year_dir / f"WeatherTimeSeries{year}.csv"
    weather_df.to_csv(weather_file_path, index=False)
    logger.info(f"Weather data saved to {weather_file_path}")

# Save crop yield data to block storage (for all crops)
def save_crop_yield_to_block_storage(fips_code, crop_data_by_crop):
    """Save the preprocessed crop yield data to block storage."""
    for crop, yield_data in crop_data_by_crop.items():
        crop_file_path = Path(block_storage_dir) / f"{fips_code}" / f"{crop}.json"
        with open(crop_file_path, 'w') as f:
            json.dump(yield_data, f)
        logger.info(f"Crop yield data for '{crop}' saved to {crop_file_path}")

# Main function to orchestrate data processing
def process_and_store_data():
    # Connect to Swift
    conn = connect_to_swift()

    # List all available years and states (from HRRR data) in Swift
    logger.info("Fetching available HRRR data from Swift...")
    weather_container = "weather-container"  # Example Swift container name for weather data
    weather_prefix = "hrrr/data"  # The prefix where HRRR data is stored
    weather_objects = conn.object_store.objects(container=weather_container, prefix=weather_prefix)
    
    available_years = set()
    available_states = set()
    for obj in weather_objects:
        path_parts = obj.name.split('/')
        if len(path_parts) >= 4:  # HRRR data follows hrrr/data/year/state
            year = path_parts[2]
            state = path_parts[3]
            available_years.add(year)
            available_states.add(state)
    
    # Fetch USDA crop yield data (state, year)
    logger.info("Fetching available USDA crop yield data from Swift...")
    crop_container = "crop-yield-container"  # Example Swift container name for crop yield data
    crop_prefix = "usda/data/crop"  # The prefix where USDA data is stored
    crop_objects = conn.object_store.objects(container=crop_container, prefix=crop_prefix)

    available_crop_years = set()
    for obj in crop_objects:
        path_parts = obj.name.split('/')
        if len(path_parts) >= 3:  # USDA data follows usda/data/crop/year
            year = path_parts[2]
            available_crop_years.add(year)
    
    # Loop through available years and states and process the data
    for year in available_years:
        for state in available_states:
            try:
                # Download HRRR weather data
                weather_object_name = f"hrrr/data/{year}/{state}/weather.csv"  # Example path in Swift
                weather_local_path = f"/tmp/{state}_{year}_weather.csv"
                download_data_from_swift(conn, weather_container, weather_object_name, weather_local_path)

                # Preprocess the weather data
                weather_df = preprocess_hrrr_data(weather_local_path)

                # Save weather data to block storage
                fips_code = "19001"  # Example FIPS code for the county, this should be mapped dynamically
                save_weather_to_block_storage(fips_code, year, weather_df)

                # Download USDA crop yield data
                if year in available_crop_years:
                    crop_object_name = f"usda/data/crop/{year}/yield.csv"  # Example path in Swift
                    crop_local_path = f"/tmp/{state}_{year}_yield.csv"
                    download_data_from_swift(conn, crop_container, crop_object_name, crop_local_path)

                    # Preprocess the crop yield data
                    yield_data_by_crop = preprocess_usda_data(crop_local_path)

                    # Save crop yield data to block storage
                    save_crop_yield_to_block_storage(fips_code, yield_data_by_crop)

            except Exception as e:
                logger.error(f"Error processing data for year {year}, state {state}: {e}")

    # Close Swift connection
    conn.close()

# Run the data processing pipeline
if __name__ == "__main__":
    process_and_store_data()