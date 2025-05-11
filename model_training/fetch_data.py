# fetch_data.py
import os
import json
import io
import pandas as pd
import openstack

def connect_to_swift():
    required_envs = [
        "OS_AUTH_URL", "OS_APPLICATION_CREDENTIAL_ID", "OS_APPLICATION_CREDENTIAL_SECRET"
    ]
    missing = [key for key in required_envs if key not in os.environ]
    if missing:
        raise EnvironmentError(f"Missing required OpenStack creds: {missing}")
    return openstack.connect()

def get_yield_mapping(json_bytes: bytes) -> dict:
    yield_data = json.loads(json_bytes.decode("utf-8"))
    return {str(entry["year"]): entry["yield"] for entry in yield_data}

def process_fips_crop(fips_code: str,
                      crop_name: str,
                      output_dir: str = "output"):
    conn = connect_to_swift()
    container = os.environ.get("OS_SWIFT_CONTAINER_NAME")
    if not container:
        raise EnvironmentError("OS_SWIFT_CONTAINER_NAME not set.")

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{fips_code}/"
    objs = list(conn.object_store.objects(container=container, prefix=prefix))

    # 1) load yields
    json_path = f"{fips_code}/{crop_name.lower()}.json"
    obj = next((o for o in objs if o.name == json_path), None)
    if not obj:
        raise FileNotFoundError(f"No yield JSON at {json_path}")
    jm = conn.object_store.download_object(container, obj.name)
    yield_map = get_yield_mapping(jm)

    # 2) load weather CSV per year
    frames = {}
    for o in objs:
        parts = o.name.split("/")
        if len(parts)==3 and parts[0]==fips_code and parts[2]==f"WeatherTimeSeries{parts[1]}.csv":
            yr = parts[1]
            if yr not in yield_map:
                print(f"[WARN] skip {yr} (no yield)")
                continue
            raw = conn.object_store.download_object(container, o.name)
            df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
            df = df[df["Year"]==int(yr)]
            if df.empty:
                continue
            df["Yield"] = yield_map[yr]
            df["FIPS Code"] = fips_code
            frames[yr] = df

    if len(frames) < 3:
        raise ValueError("Need at least 3 years of data to create train/eval/test split.")

    yrs = sorted(frames.keys())
    train_years = yrs[:-2]
    eval_year = yrs[-2]
    test_year = yrs[-1]

    df_train = pd.concat([frames[y] for y in train_years], ignore_index=True)
    df_eval  = frames[eval_year]
    df_test  = frames[test_year]

    base = f"{fips_code}_{crop_name.lower()}"
    df_train.to_csv(f"{output_dir}/{base}_training_data.csv", index=False)
    df_eval.to_csv(f"{output_dir}/{base}_eval_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/{base}_testing_data.csv", index=False)

    print(f"[OK] wrote training_data.csv")
    print(f"[OK] wrote eval_data.csv")
    print(f"[OK] wrote testing_data.csv")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Fetch weather+yield from Swift → two CSVs"
    )
    p.add_argument("--fips",  required=True)
    p.add_argument("--crop",  required=True)
    p.add_argument("--out",   default="output")
    args = p.parse_args()
    process_fips_crop(args.fips, args.crop, args.out)
