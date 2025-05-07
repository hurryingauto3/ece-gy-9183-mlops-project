import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

def build_features_csv(
    data_lake_dir="data_lake_organized/",
    crop_name="corn",
    output_csv_path="features.csv"
):
    """
    Builds a features.csv that aggregates weather statistics per FIPS/year
    and joins with USDA yield data.
    """
    # Storage for final dataset
    all_rows = []

    data_lake_path = Path(data_lake_dir)

    # List all FIPS folders
    fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

    for fips_folder in tqdm(fips_folders, desc="Processing FIPS codes"):
        fips_code = fips_folder.name
        
        # Load the USDA yield JSON for the selected crop
        crop_json_path = fips_folder / f"{crop_name.lower()}.json"
        if not crop_json_path.exists():
            continue  # Skip if no USDA data
        
        with open(crop_json_path, 'r') as f:
            yield_data = json.load(f)
        
        # Go through each year available inside the FIPS folder
        year_folders = [y for y in fips_folder.iterdir() if y.is_dir()]
        
        for year_folder in year_folders:
            year = year_folder.name
            weather_csv = year_folder / f"WeatherTimeSeries{year}.csv"
            
            if not weather_csv.exists():
                continue  # Skip if weather file missing
            
            if year not in yield_data:
                continue  # Skip if USDA yield missing for this year
            
            # Load the daily weather time series
            df = pd.read_csv(weather_csv)
            
            # Basic aggregation: mean, std, max, min per feature
            feature_row = {
                'FIPS': fips_code,
                'Year': year,
                'Yield (bu/acre)': yield_data[year]['yield'],
                'Production (bu)': yield_data[year]['production'],
            }

            # For each weather feature, calculate statistics
            for col in [
                'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                'Precipitation (kg m**-2)', 'Relative Humidity (%)',
                'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                'Downward Shortwave Radiation Flux (W m**-2)',
                'Vapor Pressure Deficit (kPa)'
            ]:
                if col in df.columns:
                    feature_row[f"{col} Mean"] = df[col].mean()
                    feature_row[f"{col} Std"] = df[col].std()
                    feature_row[f"{col} Max"] = df[col].max()
                    feature_row[f"{col} Min"] = df[col].min()
            
            # Save the final feature row
            all_rows.append(feature_row)

    # Combine everything into a DataFrame
    final_df = pd.DataFrame(all_rows)

    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Features CSV created: {output_csv_path}")

# Example usage:
build_features_csv(crop_name="corn")