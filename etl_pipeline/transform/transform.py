import os
import json
import pandas as pd
from pathlib import Path
import logging
from openstack import connection

# Setup logger
logger = logging.getLogger("data_processing_pipeline")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Block Storage Directory Path (Mounted block storage path)
block_storage_dir = "/mnt/project4/data_lake"  # Adjust this to the correct mounted block storage path

# VM Temporary Directory for Local Downloads
temp_download_dir = "/app/downloaded_temp"  # Same as extract.py's default

# Ensure the temporary download directory exists
Path(temp_download_dir).mkdir(parents=True, exist_ok=True)

# Swift Connection Setup (we'll use it only if needed)
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

# Preprocess HRRR weather data
def preprocess_hrrr_data(weather_data_path):
    """Preprocess HRRR weather data (remove missing values, handle columns)."""
    try:
        df = pd.read_csv(weather_data_path)
        # Filter only relevant columns (e.g., weather variables)
        df_cleaned = df[['Year', 'Month', 'Day', 'State', 'County', 'FIPS Code', 
                         'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)', 
                         'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)', 
                         'Wind Speed (m s**-1)']]
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
        df_filtered = df[['year', 'state_ansi', 'county_ansi', 'commodity_desc', 
                          'PRODUCTION, MEASURED IN 480 LB BALES', 'YIELD, MEASURED IN LB / ACRE']]

        # Create a dictionary to map FIPS codes to yield data, categorized by crop type
        yield_data_by_fips = {}

        # Generate FIPS code by combining state_ansi and county_ansi
        for _, row in df_filtered.iterrows():
            state_ansi = row['state_ansi']
            county_ansi = row['county_ansi']
            crop = row['commodity_desc'].lower()

            # Combine state_ansi and county_ansi to get FIPS code (state_ansi + county_ansi)
            fips_code = f"{int(state_ansi):02d}{int(county_ansi):03d}"

            if fips_code not in yield_data_by_fips:
                yield_data_by_fips[fips_code] = {}

            if crop not in yield_data_by_fips[fips_code]:
                yield_data_by_fips[fips_code][crop] = []

            yield_data_by_fips[fips_code][crop].append({
                'year': row['year'],
                'yield': row['YIELD, MEASURED IN LB / ACRE']
            })

        return yield_data_by_fips
    except Exception as e:
        logger.error(f"Error preprocessing USDA data: {e}")
        raise

# Save HRRR data by FIPS code
def save_weather_data_by_fips(weather_df, fips_code):
    """Save the preprocessed weather data to block storage based on FIPS code."""
    # Save data per FIPS per year
    for year in weather_df['Year'].unique():
        year_df = weather_df[weather_df['Year'] == year]
        year_dir = Path(block_storage_dir) / f"{fips_code}" / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        weather_file_path = year_dir / f"WeatherTimeSeries{year}.csv"
        year_df.to_csv(weather_file_path, index=False)
        logger.info(f"Weather data for FIPS {fips_code} and year {year} saved to {weather_file_path}")

# Save crop yield data to block storage
def save_crop_yield_to_block_storage(fips_code, crop_data_by_fips):
    """Save the preprocessed crop yield data to block storage."""
    for crop, yield_data in crop_data_by_fips[fips_code].items():
        crop_file_path = Path(block_storage_dir) / f"{fips_code}" / f"{crop}.json"
        with open(crop_file_path, 'w') as f:
            json.dump(yield_data, f)
        logger.info(f"Crop yield data for '{crop}' saved to {crop_file_path}")

# Main function to orchestrate data processing
def process_and_store_data():
    logger.info("Processing HRRR and USDA data from local VM storage...")

    available_fips_codes = set()

    # Loop through the temp download directory to list the available files for HRRR and USDA
    for root, dirs, files in os.walk(temp_download_dir):
        for file in files:
            if file.endswith("weather.csv"):  # Filter weather data files
                # Extract year, state, and FIPS info from the file path (assuming the format 'state_year_weather.csv')
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    state = path_parts[-2]
                    year = path_parts[-1].split("_")[1]
                    fips_code = pd.read_csv(os.path.join(root, file))['FIPS Code'].iloc[0]  # Read FIPS code from the HRRR file
                    available_fips_codes.add(fips_code)

            elif file.endswith("yield.csv"):  # Filter USDA data files
                # Extract year, state, and county ANSI codes from the USDA file
                crop_local_path = os.path.join(root, file)
                yield_data_by_fips = preprocess_usda_data(crop_local_path)
                
                for fips_code in yield_data_by_fips:
                    # Process the USDA data for the FIPS code and store in block storage
                    save_crop_yield_to_block_storage(fips_code, yield_data_by_fips)

    # Process each available FIPS code for weather data
    for fips_code in available_fips_codes:
        try:
            # Process HRRR weather data for the FIPS code
            weather_local_path = os.path.join(temp_download_dir, f"{state}_{year}_weather.csv")
            weather_df = preprocess_hrrr_data(weather_local_path)
            save_weather_data_by_fips(weather_df, fips_code)
        except Exception as e:
            logger.error(f"Error processing HRRR data for FIPS code {fips_code}: {e}")

    logger.info("Data processing finished.")

# Run the data processing pipeline
if __name__ == "__main__":
    process_and_store_data()
