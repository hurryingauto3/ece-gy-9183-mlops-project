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
Path(transformed_data_root).mkdir(parents=True, exist_ok=True)

# --- Supported crops ---
crops = ["Corn", "Soybeans", "Cotton", "WinterWheat"]
commodity_map = {"WinterWheat": "Wheat"}  # USDA uses 'Wheat' for WinterWheat

# --- In-memory storage ---
combined_records = []

# --- Preprocess HRRR Weather Files ---
def preprocess_hrrr_all_months(file_paths):
    dfs = []
    for path in file_paths:
        if path.is_dir():
            logger.warning(f"Skipping directory in file list: {path}")
            continue
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read HRRR file {path}: {e}")

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    df_all.rename(columns={
        'FIPS Code': 'FIPS',
        'Avg Temperature (K)': 'Avg Temp (K)',
        'Max Temperature (K)': 'Max Temp (K)',
        'Min Temperature (K)': 'Min Temp (K)',
        'Precipitation (kg m**-2)': 'Precip (kg/m²)',
        'Relative Humidity (%)': 'Humidity (%)',
        'Wind Gust (m s**-1)': 'Wind Gust (m/s)',
        'Wind Speed (m s**-1)': 'Wind Speed (m/s)',
        'U Component of Wind (m s**-1)': 'U Wind (m/s)',
        'V Component of Wind (m s**-1)': 'V Wind (m/s)',
        'Downward Shortwave Radiation Flux (W m**-2)': 'Solar Flux (W/m²)',
        'Vapor Pressure Deficit (kPa)': 'VPD (kPa)'
    }, inplace=True)

    weather_cols = [
        'Avg Temp (K)', 'Max Temp (K)', 'Min Temp (K)', 'Precip (kg/m²)',
        'Humidity (%)', 'Wind Gust (m/s)', 'Wind Speed (m/s)',
        'U Wind (m/s)', 'V Wind (m/s)', 'Solar Flux (W/m²)', 'VPD (kPa)'
    ]
    df_ffill = df_all[weather_cols].ffill()
    df_bfill = df_all[weather_cols].bfill()
    df_all[weather_cols] = ((df_ffill + df_bfill) / 2)

    agg_df = df_all.groupby(["FIPS", "Year", "Month", "Day"]).agg({
        'Avg Temp (K)': 'mean',
        'Max Temp (K)': 'max',
        'Min Temp (K)': 'min',
        'Precip (kg/m²)': ['mean', 'min', 'max'],
        'Humidity (%)': ['mean', 'min', 'max'],
        'Wind Gust (m/s)': ['mean', 'min', 'max'],
        'Wind Speed (m/s)': ['mean', 'min', 'max'],
        'U Wind (m/s)': ['mean', 'min', 'max'],
        'V Wind (m/s)': ['mean', 'min', 'max'],
        'Solar Flux (W/m²)': ['mean', 'min', 'max'],
        'VPD (kPa)': ['mean', 'min', 'max'],
    })

    # Flatten multi-index columns
    agg_df.columns = [' '.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    agg_df[["Year", "Month", "Day"]] = agg_df[["Year", "Month", "Day"]].astype(int)
    return agg_df

# --- Preprocess USDA Yield Files ---
def preprocess_usda_data(path, crop_name):
    try:
        df = pd.read_csv(path)
        df.rename(columns={
            "commodity_desc": "Crop Type",
            "year": "Year",
            "state_ansi": "State ANSI",
            "county_ansi": "County ANSI"
        }, inplace=True)

        crop_key = commodity_map.get(crop_name, crop_name).lower()
        df_filtered = df[df["Crop Type"].str.lower() == crop_key]

        yield_col = None
        for col in df_filtered.columns:
            if "yield" in col.lower() and "acre" in col.lower():
                yield_col = col
                break

        if not yield_col:
            logger.error(f"Yield column not found in file: {path}")
            return {}

        df_filtered.rename(columns={yield_col: "Yield"}, inplace=True)
        df_filtered.dropna(inplace=True)

        yield_by_fips = defaultdict(list)
        for _, row in df_filtered.iterrows():
            fips = f"{int(row['State ANSI']):02d}{int(row['County ANSI']):03d}"
            yield_by_fips[fips].append({
                "year": int(row["Year"]),
                "yield": row["Yield"]
            })

        return yield_by_fips

    except Exception as e:
        logger.error(f"Error preprocessing USDA data at {path}: {e}")
        return {}

# --- Gather All USDA Yields ---
def get_all_usda_yields():
    usda_base = Path(raw_data_root) / "USDA Crop Dataset" / "data"
    yield_by_fips_crop = defaultdict(lambda: defaultdict(list))

    if not usda_base.exists():
        logger.warning("No USDA data found.")
        return yield_by_fips_crop

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
                    yield_by_fips_crop[fips_code][crop].extend(records)

    return yield_by_fips_crop

# --- Process Weather and Collect Records ---
def process_weather_data():
    hrrr_base = Path(raw_data_root) / "WRF-HRRR Computed dataset" / "data"
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

            df["FIPS"] = df["FIPS"].astype(int).apply(lambda x: f"{x:05d}")
            df["Year"] = df["Year"].astype(int)

            for fips_code in df["FIPS"].unique():
                fips_df = df[df["FIPS"] == fips_code].copy()
                for crop in crops:
                    fips_df[f"yield_{crop.lower()}"] = 0
                combined_records.append(fips_df)

# --- Build Dataset ---
def build_final_dataset_in_memory():
    logger.info("Building final dataset from in-memory records")
    if not combined_records:
        logger.warning("No weather records to process.")
        return

    full_df = pd.concat(combined_records, ignore_index=True)
    yield_lookup = get_all_usda_yields()

    for i, row in full_df.iterrows():
        fips = row["FIPS"]
        year = int(row["Year"])
        for crop in crops:
            crop_yields = yield_lookup.get(fips, {}).get(crop, [])
            for entry in crop_yields:
                if entry["year"] == year:
                    full_df.at[i, f"yield_{crop.lower()}"] = entry["yield"]
                    break

    training_years = {"2017", "2018", "2019", "2020"}
    val_test_year = "2021"

    training_df = full_df[full_df["Year"].astype(str).isin(training_years)]
    year_2022_df = full_df[full_df["Year"].astype(str) == val_test_year]

    val_df = year_2022_df[year_2022_df["Day"] % 2 == 1]
    test_df = year_2022_df[year_2022_df["Day"] % 2 == 0]

    # Output datasets
    training_df.to_csv(Path(transformed_data_root) / "training.csv", index=False)
    test_df.to_csv(Path(transformed_data_root) / "test.csv", index=False)

    prod_dir = Path(transformed_data_root) / "production" 
    prod_dir.mkdir(parents=True, exist_ok=True)
    val_df[val_df["Day"] % 3 == 0].to_csv(prod_dir / "dev.csv", index=False)
    val_df[val_df["Day"] % 3 == 1].to_csv(prod_dir / "staging.csv", index=False)
    val_df[val_df["Day"] % 3 == 2].to_csv(prod_dir / "canary.csv", index=False)

    logger.info("Saved training.csv, test.csv, and production split files.")

# --- Main ---
def main():
    logger.info("=== Transform Pipeline Start ===")
    process_weather_data()
    build_final_dataset_in_memory()
    logger.info("=== Transform Pipeline Complete ===")

if __name__ == "__main__":
    main()