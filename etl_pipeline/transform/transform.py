import os
import json
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

# --- Logging setup ---
logger = logging.getLogger("data_processing_pipeline")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# --- Directory paths ---
raw_data_root = "/mnt/swift_store/raw_data"
transformed_data_root = "/mnt/swift_store/transformed_data"

# --- Supported crops ---
crops = ["Corn", "Soybeans", "Cotton", "WinterWheat"]
commodity_map = {"WinterWheat": "Wheat"}  # USDA uses 'Wheat' for WinterWheat

Path(transformed_data_root).mkdir(parents=True, exist_ok=True)

# --- Preprocess HRRR Weather Files ---
def preprocess_hrrr_all_months(file_paths):
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read HRRR file {path}: {e}")

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    df_clean = df_all[[
        'Year', 'Month', 'Day', 'State', 'County', 'FIPS Code',
        'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
        'Precipitation (kg m**-2)', 'Relative Humidity (%)',
        'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)'
    ]].dropna()

    df_clean.rename(columns={
        'FIPS Code': 'FIPS',
        'Avg Temperature (K)': 'Avg Temp (K)',
        'Max Temperature (K)': 'Max Temp (K)',
        'Min Temperature (K)': 'Min Temp (K)',
        'Precipitation (kg m**-2)': 'Precip (kg/m²)',
        'Relative Humidity (%)': 'Humidity (%)',
        'Wind Gust (m s**-1)': 'Wind Gust (m/s)',
        'Wind Speed (m s**-1)': 'Wind Speed (m/s)'
    }, inplace=True)

    return df_clean

# --- Preprocess USDA Yield Files ---
def preprocess_usda_data(path, crop_name):
    try:
        df = pd.read_csv(path)

        df.rename(columns={
            "commodity_desc": "Crop Type",
            "year": "Year",
            "state_ansi": "State ANSI",
            "county_ansi": "County ANSI",
            "YIELD, MEASURED IN LB / ACRE": "Yield (lb/acre)"
        }, inplace=True)

        crop_key = commodity_map.get(crop_name, crop_name).lower()
        df_filtered = df[df["Crop Type"].str.lower() == crop_key]

        yield_by_fips = defaultdict(list)
        for _, row in df_filtered.iterrows():
            fips = f"{int(row['State ANSI']):02d}{int(row['County ANSI']):03d}"
            yield_by_fips[fips].append({
                "year": int(row["Year"]),
                "yield": row["Yield (lb/acre)"]
            })

        return yield_by_fips
    except Exception as e:
        logger.error(f"Error preprocessing USDA data at {path}: {e}")
        return {}

# --- Save Weather ---
def save_weather_data(df, fips_code, year):
    out_dir = Path(transformed_data_root) / fips_code / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"WeatherTimeSeries{year}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved weather data: {out_path}")

# --- Save Crop Yield ---
def save_crop_yield(fips_code, crop, records):
    out_dir = Path(transformed_data_root) / fips_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{crop.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved yield data: {out_path}")

# --- HRRR Processor ---
def process_weather_data():
    hrrr_base = Path(raw_data_root) / "HRRR" / "data"
    if not hrrr_base.exists():
        logger.warning("No HRRR data found.")
        return

    for year_dir in hrrr_base.iterdir():
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        for state_dir in year_dir.iterdir():
            if not state_dir.is_dir():
                continue
            csv_files = list(state_dir.glob("*.csv"))
            if not csv_files:
                continue

            logger.info(f"Processing HRRR: Year={year}, State={state_dir.name}")
            df = preprocess_hrrr_all_months(csv_files)
            if df is None:
                continue

            for fips_code in df["FIPS"].unique():
                fips_df = df[df["FIPS"] == fips_code]
                save_weather_data(fips_df, str(fips_code), year)

# --- USDA Processor ---
def process_usda_data():
    usda_base = Path(raw_data_root) / "USDA" / "data"
    if not usda_base.exists():
        logger.warning("No USDA data found.")
        return

    for crop in crops:
        crop_dir = usda_base / crop
        if not crop_dir.exists():
            continue
        for year_dir in crop_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for file in year_dir.glob("*.csv"):
                yield_by_fips = preprocess_usda_data(file, crop)
                for fips_code, records in yield_by_fips.items():
                    save_crop_yield(fips_code, crop, records)

# --- Main Orchestration ---
def main():
    logger.info("=== Transform Pipeline Start ===")
    process_weather_data()
    process_usda_data()
    logger.info("=== Transform Pipeline Complete ===")

if __name__ == "__main__":
    main()
