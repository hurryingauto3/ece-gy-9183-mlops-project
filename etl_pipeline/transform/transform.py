import os
import json
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import shutil



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
      if os.path.isdir(path):
        print(f"Directory found in list: {path}")
        # try:
      df = pd.read_csv(path)
      dfs.append(df)
        # except Exception as e:
            # logger.error(f"Failed to read HRRR file {path}: {e}")

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
def process_weather_data():
    hrrr_base = Path(raw_data_root) / "WRF-HRRR Computed dataset" / "data"
    logger.info(f"HRRR base path: {hrrr_base}")

    if not hrrr_base.exists():
        logger.warning("No HRRR data found.")
        return

    for year_dir in sorted(hrrr_base.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        logger.info(f"=== Processing Year: {year} ===")

        for state_dir in sorted(year_dir.iterdir()):
            if not state_dir.is_dir():
                continue
            csv_files = list(state_dir.glob("*.csv"))
            logger.info(f"  State={state_dir.name}, CSV files found: {len(csv_files)}")

            if not csv_files:
                continue

            try:
                df = preprocess_hrrr_all_months(csv_files)
                if df is None or df.empty:
                    logger.warning(f"  No valid data for state: {state_dir.name}")
                    continue

                for fips_code in df["FIPS"].unique():
                    fips_df = df[df["FIPS"] == fips_code]
                    save_weather_data(fips_df, str(fips_code), year)

            except Exception as e:
                logger.error(f"  Failed processing state={state_dir.name}, year={year}: {e}")

        logger.info(f"=== Finished Year: {year} ===\n")


def preprocess_hrrr_all_months(file_paths):
    dfs = []
    for path in file_paths:
        if os.path.isdir(path):
            logger.warning(f"Directory instead of file: {path}")
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

    # Fill missing weather values using average of ffill and bfill
    weather_cols = [
        'Avg Temp (K)', 'Max Temp (K)', 'Min Temp (K)', 'Precip (kg/m²)',
        'Humidity (%)', 'Wind Gust (m/s)', 'Wind Speed (m/s)',
        'U Wind (m/s)', 'V Wind (m/s)', 'Solar Flux (W/m²)', 'VPD (kPa)'
    ]

    df_ffill = df_all[weather_cols].ffill()
    df_bfill = df_all[weather_cols].bfill()
    df_all[weather_cols] = ((df_ffill + df_bfill) / 2)

    # Drop critical missing values before aggregation
    df_all.dropna(subset=["FIPS", "Year", "Month", "Day"], inplace=True)

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

    # Cast date columns to int cleanly
    agg_df[["Year", "Month", "Day"]] = agg_df[["Year", "Month", "Day"]].astype(int)

    return agg_df



# --- Save Weather ---
def save_weather_data(df, fips_code, year):
    out_dir = Path(transformed_data_root) / fips_code / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"WeatherTimeSeries{year}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved weather data: {out_path}")

# --- Save Crop Yield ---
def save_crop_yield(fips_code, crop, new_records):
    out_dir = Path(transformed_data_root) / fips_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{crop.lower()}.json"

    existing = []
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing file {out_path}: {e}")

    # Merge and deduplicate by year
    all_records = {r["year"]: r for r in existing}
    for r in new_records:
        all_records[r["year"]] = r  # This will update or add

    merged_records = sorted(all_records.values(), key=lambda x: x["year"])

    with open(out_path, "w") as f:
        json.dump(merged_records, f, indent=2)

    logger.info(f"Updated yield data: {out_path}")

# --- HRRR Processor ---


# --- USDA Processor ---
def process_usda_data():
    usda_base = Path(raw_data_root) / "USDA Crop Dataset" / "data"
    print(usda_base)
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

def build_final_dataset():
    all_records = []

    for fips_dir in Path(transformed_data_root).iterdir():
        if not fips_dir.is_dir():
            continue

        for year_dir in fips_dir.iterdir():
            if not year_dir.is_dir():
                continue

            weather_file = year_dir / f"WeatherTimeSeries{year_dir.name}.csv"
            if not weather_file.exists():
                continue

            try:
                df = pd.read_csv(weather_file)
                df["FIPS"] = fips_dir.name
                year = int(year_dir.name)

                for crop in crops:
                    crop_file = fips_dir / f"{crop.lower()}.json"
                    yield_val = 0
                    if crop_file.exists():
                        with open(crop_file) as f:
                            crop_yields = json.load(f)
                            for entry in crop_yields:
                                if entry["year"] == year:
                                    yield_val = entry["yield"]
                                    break
                    df[f"yield_{crop.lower()}"] = yield_val

                all_records.append(df)

            except Exception as e:
                logger.error(f"Failed processing {weather_file}: {e}")

    if not all_records:
        logger.warning("No combined data to process.")
        return

    full_df = pd.concat(all_records, ignore_index=True)

    training_years = {"2017", "2018", "2019", "2020"}
    val_test_year = "2021"

    training_df = full_df[full_df["Year"].astype(str).isin(training_years)]
    year_2022_df = full_df[full_df["Year"].astype(str) == val_test_year]

    val_df = year_2022_df[year_2022_df["Day"] % 2 == 1]  # Odd days
    test_df = year_2022_df[year_2022_df["Day"] % 2 == 0]  # Even days


    # Save final datasets
    training_path = Path(transformed_data_root) / "training.csv"
    val_path = Path(transformed_data_root) / "val.csv"
    test_path = Path(transformed_data_root) / "test.csv"

    training_df.to_csv(training_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Datasets saved: training.csv, val.csv, test.csv")

    # --- Cleanup ---
    for item in Path(transformed_data_root).iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.name not in {"training.csv", "val.csv", "test.csv"}:
            item.unlink()

    logger.info("Cleanup complete. Only final datasets retained.")



# --- Main Orchestration ---
def main():
    logger.info("=== Transform Pipeline Start ===")
    process_weather_data()
    process_usda_data()
    build_final_dataset()
    logger.info("=== Transform Pipeline Complete ===")

if __name__ == "__main__":
    main()
