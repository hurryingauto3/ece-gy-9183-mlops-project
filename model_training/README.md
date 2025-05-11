# AgriYieldPredictor: Model Training Operations Manual

This manual provides instructions for **setting up, running, logging, and promoting** crop yield prediction models using the AgriYieldPredictor training system. The system uses MLflow for experiment tracking and model lifecycle management. This document assumes that data ingestion and serving infrastructure are handled by other subsystems.

---

## 0. Pre-Conditions

Before using this module:

* Processed data is available in an OpenStack Swift container as `features.csv` with crop yields and weather data.
* All relevant environment variables from `.env` are loaded.
* A Docker environment is available to run this module locally or on a Chameleon GPU node.

---

## 1. Folder Structure

```
model_training/
├── data_loader.py         # Loads multi-crop CSV data
├── fetch_data.py          # Downloads data from Swift, splits to train/eval/test
├── model.py               # LSTM + TCN model class
├── train.py               # Trains the model and logs to MLflow
├── predict.py             # Runs inference using a model in the registry
├── promote_model.py       # Promotes a model version to a stage
├── utils.py               # Common training utilities
├── requirements.txt
└── Dockerfile             # Environment to be containerized for training
```

---

## 2. Required Environment Variables

Ensure the following are set in your shell or container via `.env`:

```bash
# MLflow config
MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_MODEL_NAME=AgriYieldPredictor
MLFLOW_MODEL_STAGE=Production
MLFLOW_EXPERIMENT_NAME="Crop Yield Training"

# OpenStack Swift config
FS_OPENSTACK_SWIFT_CONTAINER_NAME=object-persist-project-4
OS_AUTH_URL=...
OS_APPLICATION_CREDENTIAL_ID=...
OS_APPLICATION_CREDENTIAL_SECRET=...
OS_REGION_NAME=CHI@TACC
```

---

## 3. Operational Commands

### 3.1 Fetch Data from Swift

```bash
python fetch_data.py --fips <FIPS_CODE> --crop <crop>
```

Outputs 3 CSVs into the `output/` directory:

* `<fips>_<crop>_training_data.csv`
* `<fips>_<crop>_eval_data.csv`
* `<fips>_<crop>_testing_data.csv`

### 3.2 Train and Log Model to MLflow

```bash
python train.py \
  --train-csv output/<fips>_<crop>_training_data.csv \
  --eval-csv output/<fips>_<crop>_eval_data.csv \
  --mlflow
```

Logs the following to MLflow:

* Parameters
* RMSE/MAE metrics
* GPU diagnostics
* Model artifact
* Registers model version to `AgriYieldPredictor`

### 3.3 Promote Model to Registry Stage

```bash
export TARGET_STAGE=Staging  # Options: Staging, Canary, Production
python promote_model.py
```

This moves the latest model from "None" stage to the specified one.

### 3.4 Run Prediction from Registry

```bash
python predict.py \
  --stage Production \
  --fips-id <fips_id> \
  --crop-id <crop_id> \
  --csv /path/to/input.csv
```

This returns the yield prediction for the given crop and county-year.

---

## 4. Valid MLflow Stages

* `None`: unpromoted version
* `Staging`: approved for evaluation
* `Canary`: under live testing
* `Production`: used in serving endpoint

---

## 5. Accessing MLflow & Artifacts

* MLflow UI: `http://<your-floating-ip>:5001`
* MinIO Browser: `http://<your-floating-ip>:9001`

---

## 6. GPU Jupyter Training Environment (To Be Provisioned)

You will deploy a GPU Jupyter container using:

* `model_training/Dockerfile`
* SSH tunnel to expose Jupyter on port 8888

Launch instructions will be added once the Terraform/Ansible pipeline is ready.

---

## 7. Maintenance

* Old model versions can be deleted from the MLflow UI if no longer needed.
* `output/` files can be cleaned up after promotion.

---

## 8. Pre-Submission Checklist

* [ ] Is `.env` loaded correctly in this shell/container?
* [ ] Did `train.py` log metrics and artifacts to MLflow?
* [ ] Was the model successfully registered?
* [ ] Is the model visible under the expected stage?
* [ ] Did `predict.py` return a valid scalar prediction?

---

## Maintainers

This training module is maintained by the Model Team for the Agri MLOps Project.
Contact: @model-lead or refer to internal wiki for escalation.
