import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from scipy.stats import linregress  # For trend/slope features

def build_advanced_features_csv(
    data_lake_dir="data_lake_organized/",
    crop_name="corn",
    output_csv_path="features_advanced.csv"
):
    """
    Builds an advanced features.csv that includes:
    - Basic weather stats
    - Growing season only (April–October)
    - Extreme event counts
    - Weather trends
    - Joined with USDA crop yield
    """
    all_rows = []
    data_lake_path = Path(data_lake_dir)
    fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

    for fips_folder in tqdm(fips_folders, desc="Processing FIPS codes"):
        fips_code = fips_folder.name
        
        crop_json_path = fips_folder / f"{crop_name.lower()}.json"
        if not crop_json_path.exists():
            continue
        
        with open(crop_json_path, 'r') as f:
            yield_data = json.load(f)
        
        year_folders = [y for y in fips_folder.iterdir() if y.is_dir()]
        
        for year_folder in year_folders:
            year = year_folder.name
            weather_csv = year_folder / f"WeatherTimeSeries{year}.csv"
            
            if not weather_csv.exists():
                continue
            if year not in yield_data:
                continue
            
            df = pd.read_csv(weather_csv)
            
            # ✅ Filter for growing season: April (4) to October (10)
            df = df[(df['Month'] >= 4) & (df['Month'] <= 10)]

            feature_row = {
                'FIPS': fips_code,
                'Year': year,
                'Yield (bu/acre)': yield_data[year]['yield'],
                'Production (bu)': yield_data[year]['production'],
            }

            # List of weather columns to process
            weather_columns = [
                'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                'Precipitation (kg m**-2)', 'Relative Humidity (%)',
                'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                'Downward Shortwave Radiation Flux (W m**-2)',
                'Vapor Pressure Deficit (kPa)'
            ]
            
            # ✅ Basic aggregation: mean, std, min, max
            for col in weather_columns:
                if col in df.columns:
                    feature_row[f"{col} Mean"] = df[col].mean()
                    feature_row[f"{col} Std"] = df[col].std()
                    feature_row[f"{col} Max"] = df[col].max()
                    feature_row[f"{col} Min"] = df[col].min()

            # ✅ Extreme event counts
            if 'Max Temperature (K)' in df.columns:
                hot_days = (df['Max Temperature (K)'] > 308.15).sum()  # 35°C in Kelvin
                cold_days = (df['Min Temperature (K)'] < 278.15).sum()  # 5°C in Kelvin
                feature_row['Hot Days >35C'] = hot_days
                feature_row['Cold Days <5C'] = cold_days
            
            if 'Precipitation (kg m**-2)' in df.columns:
                heavy_rain_days = (df['Precipitation (kg m**-2)'] > 10).sum()
                dry_days = (df['Precipitation (kg m**-2)'] < 1).sum()
                feature_row['Heavy Rain Days >10mm'] = heavy_rain_days
                feature_row['Dry Days <1mm'] = dry_days

            # ✅ Trend features (Slope over time)
            if 'Avg Temperature (K)' in df.columns:
                x = range(len(df))
                slope, intercept, r_value, p_value, std_err = linregress(x, df['Avg Temperature (K)'])
                feature_row['Avg Temp Trend (slope)'] = slope

            if 'Precipitation (kg m**-2)' in df.columns:
                x = range(len(df))
                slope, intercept, r_value, p_value, std_err = linregress(x, df['Precipitation (kg m**-2)'])
                feature_row['Precipitation Trend (slope)'] = slope

            all_rows.append(feature_row)

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Advanced features CSV created: {output_csv_path}")

# Example usage:
build_advanced_features_csv(crop_name="corn", output_csv_path="features_corn_advanced.csv")