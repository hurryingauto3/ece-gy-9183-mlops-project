# inference.py
import os
import argparse

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pytorch

from load_data import WEATHER_FEATURE_COLUMNS

def load_features(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    missing = [c for c in WEATHER_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    arr = df[WEATHER_FEATURE_COLUMNS].dropna().values
    if arr.shape[0] == 0:
        raise ValueError("No valid rows after dropping NaNs")
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",  required=True)
    p.add_argument("--fips-id",    type=int, default=0)
    p.add_argument("--model-name", default=os.environ.get("MLFLOW_MODEL_NAME","AgriYieldPredictor"))
    p.add_argument("--model-stage",default=os.environ.get("MLFLOW_MODEL_STAGE","Production"))
    p.add_argument("--num-samples",type=int, default=500)
    p.add_argument("--output-fig", default="model_training/output/histogram.png")
    args = p.parse_args()

    # 1) load model
    uri = f"models:/{args.model_name}/{args.model_stage}"
    model = mlflow.pytorch.load_model(uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    # 2) load data
    x = load_features(args.input_csv).to(device)
    fips = torch.tensor([args.fips_id], dtype=torch.long, device=device)

    # 3) sample
    preds = []
    with torch.no_grad():
        for _ in range(args.num_samples):
            y = model(x, fips)
            preds.append(y.item())
    preds = np.array(preds)

    # 4) plot
    bins = np.linspace(preds.min(), preds.max(), 21)
    counts, edges = np.histogram(preds, bins=bins)
    plt.figure(figsize=(8,6))
    plt.bar(edges[:-1], counts, width=np.diff(edges), edgecolor='black', align='edge')
    plt.xlabel("Predicted Yield")
    plt.ylabel("Frequency")
    plt.title(f"{args.model_name} MC-Dropout Dist. (N={args.num_samples})")
    plt.grid(axis='y', alpha=0.5)

    os.makedirs(os.path.dirname(args.output_fig), exist_ok=True)
    plt.savefig(args.output_fig)
    print(f"[OK] Saved histogram to {args.output_fig}")
    print(f"Mean={preds.mean():.2f}, Std={preds.std():.2f}, Min={preds.min():.2f}, Max={preds.max():.2f}")

if __name__ == "__main__":
    main()
