# promote_model.py
import os
from mlflow.tracking import MlflowClient

model_name = os.environ["MLFLOW_MODEL_NAME"]
target_stage = os.environ["TARGET_STAGE"]  # e.g., Staging, Canary, Production

client = MlflowClient()

# Find the latest version in "None" stage (unpromoted models)
versions = client.get_latest_versions(model_name, stages=["None"])
if not versions:
    raise Exception("No models in 'None' stage to promote.")

latest = versions[0]

print(f"Promoting model version {latest.version} to stage {target_stage}")
client.transition_model_version_stage(
    name=model_name,
    version=latest.version,
    stage=target_stage
)