import os
import openstack
import io
import pandas as pd

def connect_to_swift():
    required_envs = [
        "OS_AUTH_URL", "OS_APPLICATION_CREDENTIAL_ID", "OS_APPLICATION_CREDENTIAL_SECRET"
    ]
    missing = [key for key in required_envs if key not in os.environ]
    if missing:
        raise EnvironmentError(f"Missing required OpenStack creds: {missing}")
    return openstack.connect()

def download_csvs_from_swift(output_dir: str = "output"):
    conn = connect_to_swift()
    container = os.environ.get("OS_SWIFT_CONTAINER_NAME")
    if not container:
        raise EnvironmentError("OS_SWIFT_CONTAINER_NAME not set.")

    os.makedirs(output_dir, exist_ok=True)

    filenames = {
        "training": "training.csv",
        "eval": "eval.csv",
        "test": "test.csv"
    }

    for key, name in filenames.items():
        try:
            obj = conn.object_store.get_object(container, name)
            data = conn.object_store.download_object(obj)
            df = pd.read_csv(io.StringIO(data.decode("utf-8")))
            df.to_csv(os.path.join(output_dir, name), index=False)
            print(f"[OK] downloaded {name}")
        except Exception as e:
            print(f"[ERROR] could not fetch {name}: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch 3 CSVs (train/eval/test) from Swift")
    p.add_argument("--out", default="output", help="Output directory")
    args = p.parse_args()
    download_csvs_from_swift(args.out)
