import os
import argparse
import torch
import mlflow
import mlflow.pytorch
from load_data import MultiCropYieldDataset
from model import LSTMTCNRegressor
from utils import collate_fn, train_model, evaluate_model
import subprocess
from config import config

def parse_args():
    p = argparse.ArgumentParser(description="Train & register multi-crop yield model with MLflow")
    p.add_argument("--train-csv", required=True, help="Path to training CSV")
    p.add_argument("--eval-csv", required=True, help="Path to eval CSV")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load datasets
    train_ds = MultiCropYieldDataset(os.path.abspath(args.train_csv))
    eval_ds  = MultiCropYieldDataset(os.path.abspath(args.eval_csv))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 2) Build model
    input_dim = train_ds[0][0].shape[1]
    num_fips = train_ds.get_num_fips()
    num_crops = train_ds.get_num_crops()

    model = LSTMTCNRegressor(
        input_dim=input_dim,
        num_fips=num_fips,
        num_crops=num_crops,
        fips_embedding_dim=config["fips_embedding_dim"],
        hidden_dim=config["hidden_dim"],
        lstm_layers=config["lstm_layers"],
        tcn_channels=config["tcn_channels"],
        dropout_rate=config["dropout_rate"]
    )

    # 3) MLflow logging
    if args.mlflow:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
        model_name = os.environ["MLFLOW_MODEL_NAME"]

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        mlflow.start_run(log_system_metrics=True)

        mlflow.log_params({
            k: v if not isinstance(v, list) else str(v)
            for k, v in config.items()
        })

        # Log GPU info
        try:
            gpu_info = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
            mlflow.log_text(gpu_info, "gpu-info.txt")
        except Exception as e:
            print("Could not log GPU info:", e)

    # 4) Train model
    model = train_model(model, train_loader, eval_loader, num_epochs=args.epochs, lr=args.lr)

    # 5) Evaluate
    rmse, mae = evaluate_model(model, eval_loader)

    if args.mlflow:
        mlflow.log_metrics({"eval_rmse": rmse, "eval_mae": mae})
        mlflow.pytorch.log_model(model, "model")

        # Register the model under versioned name
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"[MLflow] Registered model as version: {result.version}")

        mlflow.end_run()

if __name__ == "__main__":
    main()