This module handles the full training pipeline for crop yield prediction using a deep learning model (LSTM + TCN). It includes data loading, model training, experiment tracking with MLflow, and model promotion.

> ## Assumptions (Before this begins)
---
Processed weather + yield data is already available as local CSVs via Swift mount:

```bash
/mnt/swift_store/transformed_data/train.csv

/mnt/swift_store/transformed_data/eval.csv

/mnt/swift_store/transformed_data/test.csv
```

MLflow server is already running (http://mlflow:5001) and set up to track experiments.

All credentials (OpenStack + MLflow) are defined in .env.jupyter.

Docker with GPU and Poetry is installed on the Chameleon node.

---

> ## Files

```bash
model_training/
├── 01_control_centre.ipynb      # Main notebook to drive training, logging, promotion
├── train.py                     # CLI version of training script
├── model.py                     # LSTM + TCN model definition
├── load_data.py                 # Dataset loading utilities
├── promote_model.py             # Promotes model from "None" to Staging/Canary/Production
├── utils.py                     # train_model(), evaluate_model(), collate_fn
├── Dockerfile.jupyter           # Builds Jupyter container for interactive GPU work
├── .env.jupyter                 # Stores env vars for Swift and MLflow access
├── pyproject.toml / poetry.lock # Poetry environment files
```
🧠 Model Overview

> Inputs: daily weather time series + FIPS + crop ID

> Output: predicted yield for that crop

>Model: LSTM + TCN, with embedding layers for region/crop

🚀 Workflow Summary

- Start Jupyter Server (on Chameleon GPU node)

```bash
# On GPU node
docker compose --profile gpu up jupyter
```
2. Open Notebook and Configure Hyperparameters

Edit config = {...} in 01_control_centre.ipynb.

3. Run Training + Logging

Loads train/eval data from Swift mount

Builds and trains model

Logs everything to MLflow: metrics, params, model file

4. Promote Best Model

export TARGET_STAGE=Production
python promote_model.py

Or run the corresponding cell in the notebook.

🔧 Environment Setup

Make sure .env.jupyter is present in model_training/ with:

MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_MODEL_NAME=AgriYieldPredictor
MLFLOW_EXPERIMENT_NAME=Crop Yield Training
OS_AUTH_URL=... # and other OpenStack credentials

✅ What This Covers

Interactive notebook-based training

CLI-based retraining (train.py)

MLflow experiment tracking

Manual promotion from "None" → Staging/Production

🧪 What Happens After This

Inference API pulls model from MLflow Registry based on stage

Prediction/inference handled by separate model-serving container (not in this repo)

🧼 Cleanup

Old models can be removed from MLflow UI manually

CSVs are not re-uploaded or transformed by this module

