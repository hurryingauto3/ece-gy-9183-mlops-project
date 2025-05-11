# train.py
import os
import argparse

import pandas as pd
import torch
import mlflow.pytorch

from load_data import LocalCropYieldDataset
from model import LSTMTCNRegressor
from utils import collate_fn, train_model, evaluate_model

def parse_args():
    p = argparse.ArgumentParser(description="Train or evaluate the crop‐yield model")
    p.add_argument("--train-csv",   required=True,
                   help="Path to the training CSV (with Year, FIPS Code, Yield + weather features)")
    p.add_argument("--test-csv",    required=True,
                   help="Path to the testing CSV (same columns)")
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--mlflow",      action="store_true",
                   help="If set, logs params/metrics/artifact to MLflow using env $MLFLOW_*")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load datasets
    train_ds = LocalCropYieldDataset(os.path.abspath(args.train_csv))
    test_ds  = LocalCropYieldDataset(os.path.abspath(args.test_csv))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 2) build model
    input_dim = len(train_ds._samples_with_year[0][0][0]) if len(train_ds) else 0
    num_fips  = train_ds.get_num_fips()

    model = LSTMTCNRegressor(
        input_dim=input_dim,
        num_fips=num_fips,
        fips_embedding_dim=16,
        hidden_dim=64,
        lstm_layers=1,
        tcn_channels=[64, 32],
        dropout_rate=0.1
    )

    # 3) optional MLflow run
    if args.mlflow:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "CropYield"))
        mlflow.start_run()
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
        })

    # 4) train
    model = train_model(
        model,
        train_loader,
        test_loader,      # using test as validation here
        num_epochs=args.epochs,
        lr=args.lr
    )

    # 5) evaluate
    rmse, mae = evaluate_model(model, test_loader)

    if args.mlflow:
        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae})
        mlflow.pytorch.log_model(
            model,
            artifact_path="crop_yield_model",
            registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "AgriYieldPredictor")
        )
        mlflow.end_run()

if __name__ == "__main__":
    main()
