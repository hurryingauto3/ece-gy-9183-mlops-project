# predict.py
import os
import argparse
import mlflow.pytorch
import torch
import pandas as pd

from model import LSTMTCNRegressor
from load_data import WEATHER_FEATURE_COLUMNS

def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop yield using a deployed MLflow model")
    parser.add_argument("--stage", type=str, required=True, help="MLflow stage to load model from: Production, Staging, or Canary")
    parser.add_argument("--crop-id", type=int, required=True, help="Crop ID (int, from training mapping)")
    parser.add_argument("--fips-id", type=int, required=True, help="FIPS ID (int, from training mapping)")
    parser.add_argument("--csv", type=str, required=True, help="Path to weather CSV file for a county-year")
    return parser.parse_args()

def main():
    args = parse_args()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MLFLOW_MODEL_NAME"]
    stage = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"[INFO] Loading model: {model_name} from stage: {args.stage}")
    model = mlflow.pytorch.load_model(f"models:/{model_name}/{args.stage}")
    model.eval()

    # Load weather features
    df = pd.read_csv(args.csv)
    df_weather = df[WEATHER_FEATURE_COLUMNS].dropna()
    if df_weather.empty:
        raise ValueError("No valid weather data in CSV.")
    
    x_weather = torch.tensor(df_weather.values, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    fips_id = torch.tensor([args.fips_id], dtype=torch.long)
    crop_id = torch.tensor([args.crop_id], dtype=torch.long)

    with torch.no_grad():
        pred = model(x_weather, fips_id, crop_id)

    print(f"[RESULT] Predicted yield: {pred.item():.2f}")

if __name__ == "__main__":
    main()