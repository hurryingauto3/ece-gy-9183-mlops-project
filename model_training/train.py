# train.py
import os
import argparse
import pandas as pd
import torch
import mlflow.pytorch

from data_loader import LocalCropYieldDataset
from model       import LSTMTCNRegressor
from utils       import collate_fn, train_model, evaluate_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv",  required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--mlflow",     action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load datasets
    train_ds = LocalCropYieldDataset(args.train_csv)
    test_ds  = LocalCropYieldDataset(args.test_csv)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 2) build model
    input_dim = len(LocalCropYieldDataset.__dict__['WEATHER_FEATURE_COLUMNS'])
    num_fips  = train_ds.get_num_fips()

    model = LSTMTCNRegressor(
        input_dim=input_dim,
        num_fips=num_fips,
        fips_embedding_dim=16,
        hidden_dim=64,
        lstm_layers=1,
        tcn_channels=[64,32],
        dropout_rate=0.1
    )

    # 3) (optional) MLflow start
    if args.mlflow:
        mlflow.set_experiment("CropYield")
        mlflow.start_run()

    # 4) train + eval
    model = train_model(model, train_loader,
                        num_epochs=args.epochs,
                        lr=args.lr)
    rmse, mae = evaluate_model(model, test_loader)

    if args.mlflow:
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae",  mae)
        mlflow.pytorch.log_model(
            model,
            artifact_path="crop_yield_model",
            registered_model_name="AgriYieldPredictor"
        )
        mlflow.end_run()

if __name__=="__main__":
    main()
