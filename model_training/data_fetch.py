import os
import json
import io
import pandas as pd
import openstack
from datetime import datetime


def connect_to_swift():
    required_envs = [
        "OS_AUTH_URL", "OS_AUTH_TYPE",
        "OS_APPLICATION_CREDENTIAL_ID", "OS_APPLICATION_CREDENTIAL_SECRET"
    ]
    missing = [key for key in required_envs if key not in os.environ]
    if missing:
        raise EnvironmentError(f"Missing required OpenStack credentials: {missing}")

    return openstack.connect()


def get_yield_mapping(json_bytes: bytes) -> dict:
    yield_data = json.loads(json_bytes.decode("utf-8"))
    return {str(entry["year"]): entry["yield"] for entry in yield_data}


def process_fips_crop(fips_code: str, crop_name: str, output_dir="output"):
    conn = connect_to_swift()
    container_name = os.environ.get("FS_OPENSTACK_SWIFT_CONTAINER_NAME")
    if not container_name:
        raise EnvironmentError("FS_OPENSTACK_SWIFT_CONTAINER_NAME not set.")

    os.makedirs(output_dir, exist_ok=True)

    fips_prefix = f"{fips_code}/"
    print(f"[INFO] Listing objects in container '{container_name}' under prefix '{fips_prefix}'...")
    all_objects = list(conn.object_store.objects(container=container_name, prefix=fips_prefix))

    # Locate the crop yield JSON file
    crop_json_name = f"{fips_code}/{crop_name.lower()}.json"
    crop_json_obj = next((obj for obj in all_objects if obj.name == crop_json_name), None)
    if not crop_json_obj:
        raise FileNotFoundError(f"Yield JSON file not found: {crop_json_name}")

    crop_json_bytes = conn.object_store.download_object(container=container_name, obj=crop_json_obj.name)
    yield_mapping = get_yield_mapping(crop_json_bytes)

    weather_dataframes = {}
    for obj in all_objects:
        parts = obj.name.split("/")
        if len(parts) == 3 and parts[0] == fips_code and parts[2] == f"WeatherTimeSeries{parts[1]}.csv":
            year = parts[1]
            if year not in yield_mapping:
                print(f"[WARN] Skipping {year} — no yield in JSON.")
                continue

            print(f"[INFO] Downloading weather CSV for year {year}...")
            csv_bytes = conn.object_store.download_object(container=container_name, obj=obj.name)
            df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))

            if "Year" not in df.columns:
                print(f"[WARN] CSV '{obj.name}' is missing 'Year' column. Skipping.")
                continue

            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df = df[df["Year"] == int(year)]

            if df.empty:
                print(f"[WARN] No rows in {obj.name} with Year == {year}. Skipping.")
                continue

            df["Yield"] = yield_mapping[year]
            weather_dataframes[year] = df

    if not weather_dataframes:
        raise ValueError("No usable weather + yield data found.")

    all_years = sorted(weather_dataframes.keys())
    latest_year = all_years[-1]
    train_years = all_years[:-1]

    df_train = pd.concat([weather_dataframes[y] for y in train_years], ignore_index=True) if train_years else pd.DataFrame()
    df_test = weather_dataframes[latest_year]

    base_name = f"{fips_code}_{crop_name.lower()}"
    train_path = os.path.join(output_dir, f"{base_name}_training_data.csv")
    test_path = os.path.join(output_dir, f"{base_name}_testing_data.csv")

    if not df_train.empty:
        df_train.to_csv(train_path, index=False)
        print(f"[SUCCESS] Training data saved: {train_path}")
    else:
        print(f"[INFO] No training data created (only one year present).")

    df_test.to_csv(test_path, index=False)
    print(f"[SUCCESS] Testing data saved: {test_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare local training/testing CSVs from Swift.")
    parser.add_argument("--fips", required=True, help="FIPS code, e.g., 19001")
    parser.add_argument("--crop", required=True, help="Crop name, e.g., corn")
    args = parser.parse_args()

    process_fips_crop(args.fips, args.crop)
