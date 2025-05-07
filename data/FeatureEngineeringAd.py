import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from scipy.stats import linregress  # For trend/slope features

def build_advanced_features_csv_all_crops(
    data_lake_dir="data_lake_organized/",
    output_csv_path="features_all_crops.csv"
):
    """
    Builds a features.csv for all 4 crops:
    - Aggregates weather stats (April–October)
    - Adds yield from corn, soybeans, cotton, winterwheat (if present)
    - Result: One row per (FIPS, Year) with up to 4 yield targets
    """
    crops = ["corn", "soybeans", "cotton", "winterwheat"]
    all_rows = []
    data_lake_path = Path(data_lake_dir)
    fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

    for fips_folder in tqdm(fips_folders, desc="Processing FIPS codes"):
        fips_code = fips_folder.name

        # Load all USDA crop yield data into one dict
        crop_yield = {crop: {} for crop in crops}
        for crop in crops:
            crop_path = fips_folder / f"{crop}.json"
            if crop_path.exists():
                with open(crop_path, 'r') as f:
                    crop_yield[crop] = json.load(f)

        # Process weather files for each year
        year_folders = [y for y in fips_folder.iterdir() if y.is_dir()]
        for year_folder in year_folders:
            year = year_folder.name
            weather_csv = year_folder / f"WeatherTimeSeries{year}.csv"
            if not weather_csv.exists():
                continue
            
            df = pd.read_csv(weather_csv)
            df = df[(df['Month'] >= 4) & (df['Month'] <= 10)]  # Growing season filter

            feature_row = {
                'FIPS': fips_code,
                'Year': year,
            }

            # Add available yields
            for crop in crops:
                if year in crop_yield[crop]:
                    feature_row[f"Yield_{crop}"] = crop_yield[crop][year]['yield']
                    feature_row[f"Production_{crop}"] = crop_yield[crop][year]['production']
                else:
                    feature_row[f"Yield_{crop}"] = None
                    feature_row[f"Production_{crop}"] = None

            # Weather features to summarize
            weather_columns = [
                'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                'Precipitation (kg m**-2)', 'Relative Humidity (%)',
                'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                'Downward Shortwave Radiation Flux (W m**-2)',
                'Vapor Pressure Deficit (kPa)'
            ]

            for col in weather_columns:
                if col in df.columns:
                    feature_row[f"{col} Mean"] = df[col].mean()
                    feature_row[f"{col} Std"] = df[col].std()
                    feature_row[f"{col} Max"] = df[col].max()
                    feature_row[f"{col} Min"] = df[col].min()

            # Extreme events
            if 'Max Temperature (K)' in df.columns:
                feature_row['Hot Days >35C'] = (df['Max Temperature (K)'] > 308.15).sum()
                feature_row['Cold Days <5C'] = (df['Min Temperature (K)'] < 278.15).sum()
            if 'Precipitation (kg m**-2)' in df.columns:
                feature_row['Heavy Rain Days >10mm'] = (df['Precipitation (kg m**-2)'] > 10).sum()
                feature_row['Dry Days <1mm'] = (df['Precipitation (kg m**-2)'] < 1).sum()

            # Trends
            if 'Avg Temperature (K)' in df.columns:
                x = range(len(df))
                slope, *_ = linregress(x, df['Avg Temperature (K)'])
                feature_row['Avg Temp Trend (slope)'] = slope
            if 'Precipitation (kg m**-2)' in df.columns:
                x = range(len(df))
                slope, *_ = linregress(x, df['Precipitation (kg m**-2)'])
                feature_row['Precipitation Trend (slope)'] = slope

            all_rows.append(feature_row)

    # Combine and export
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Multi-crop features CSV created: {output_csv_path}")
    
build_advanced_features_csv_all_crops()