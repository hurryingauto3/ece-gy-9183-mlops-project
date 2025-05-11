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
raw_data_root = Path("/mnt/swift_store/raw_data")
transformed_data_root = Path("/mnt/swift_store/transformed_data")
crops = ["Corn", "Soybeans", "Cotton", "WinterWheat"]
commodity_map = {"WinterWheat": "Wheat"} # USDA uses 'Wheat' for WinterWheat

transformed_data_root.mkdir(parents=True, exist_ok=True)


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
        'Year', 'Month', 'Day', 'FIPS Code',
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
    df_clean["FIPS"] = df_clean["FIPS"].astype(int).apply(lambda x: f"{x:05d}")
    return df_clean

# --- Preprocess USDA ---
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

        yield_col = next((col for col in df_filtered.columns if "yield" in col.lower() and "acre" in col.lower()), None)
        if not yield_col:
            logger.error(f"Yield column not found in file: {path}")
            return {}

        df_filtered.rename(columns={yield_col: "Yield"}, inplace=True)

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

# --- Save Weather ---
def save_weather_data(df, fips_code, year):
    out_dir = transformed_data_root / fips_code / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"WeatherTimeSeries{year}.csv"
    df.to_csv(out_path, index=False)

# --- Save Yield ---
def save_crop_yield(fips_code, crop, new_records):
    out_dir = transformed_data_root / fips_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{crop.lower()}.json"

    existing = []
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
        except:
            pass

    all_records = {r["year"]: r for r in existing}
    for r in new_records:
        all_records[r["year"]] = r
    merged_records = sorted(all_records.values(), key=lambda x: x["year"])

    with open(out_path, "w") as f:
        json.dump(merged_records, f, indent=2)

# --- HRRR Processor ---
def process_weather_data():
    hrrr_base = raw_data_root / "HRRR" / "data"
    if not hrrr_base.exists():
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
            df = preprocess_hrrr_all_months(csv_files)
            if df is None:
                continue
            for fips_code in df["FIPS"].unique():
                save_weather_data(df[df["FIPS"] == fips_code], fips_code, year)

# --- USDA Processor ---
def process_usda_data():
    usda_base = raw_data_root / "USDA" / "data"
    if not usda_base.exists():
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

# --- Combine Weather + Yield + Split ---
def build_final_dataset():
    all_records = []

    for fips_dir in transformed_data_root.iterdir():
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
                            for entry in json.load(f):
                                if entry["year"] == year:
                                    yield_val = entry["yield"]
                                    break
                    df[f"yield_{crop.lower()}"] = yield_val

                all_records.append(df)

            except Exception as e:
                logger.error(f"Failed processing {weather_file}: {e}")

    if not all_records:
        return

    full_df = pd.concat(all_records, ignore_index=True)
    training_df = full_df[full_df["Year"].isin([2017, 2018, 2019, 2020])]
    val_test_df = full_df[full_df["Year"] == 2021].sample(frac=1, random_state=42)
    val_df = val_test_df.iloc[:len(val_test_df)//2]
    test_df = val_test_df.iloc[len(val_test_df)//2:]

    # Save outputs
    training_df.to_csv(transformed_data_root / "training.csv", index=False)
    val_df.to_csv(transformed_data_root / "val.csv", index=False)
    test_df.to_csv(transformed_data_root / "test.csv", index=False)
    logger.info("Saved training.csv, val.csv, test.csv")

    # Cleanup transformed directory (except final .csvs)
    for item in transformed_data_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.suffix != ".csv":
            item.unlink()

    # Cleanup raw data directory entirely
    shutil.rmtree(raw_data_root, ignore_errors=True)
    logger.info("Cleaned up raw_data and intermediate transformed files.")

# --- Main ---
def main():
    logger.info("=== Transform Pipeline Start ===")
    process_weather_data()
    process_usda_data()
    build_final_dataset()
    logger.info("=== Transform Pipeline Complete ===")

if __name__ == "__main__":
    main()
