import os
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
LOCAL_MLFLOW_DIR = "/root/huy/Deep-Skin-Lesion-Classification/mlruns"
REMOTE_TRACKING_URI = (
    "https://dagshub.com/huytrnq/Deep-Skin-Lesion-Classification.mlflow"
)
USERNAME = "huytrnq"
PASSWORD = "HuyinEU1997@@"

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_URI"] = REMOTE_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD

# Initialize clients
local_client = MlflowClient(tracking_uri=f"file://{LOCAL_MLFLOW_DIR}")
remote_client = MlflowClient(tracking_uri=REMOTE_TRACKING_URI)

# List and log experiments
local_experiments = local_client.search_experiments()

print(local_experiments)

for exp in local_experiments:
    print(f"Processing Experiment: {exp.name}")

    # Ensure the remote experiment exists
    if remote_client.get_experiment_by_name(exp.name) is None:
        remote_client.create_experiment(exp.name)
        print(f"Created remote experiment: {exp.name}")

    remote_experiment = remote_client.get_experiment_by_name(exp.name)
    remote_experiment_id = (
        remote_experiment.experiment_id if remote_experiment else None
    )

    # Process runs
    local_runs = local_client.search_runs(exp.experiment_id)
    for run_info in local_runs:
        with mlflow.start_run(
            experiment_id=remote_experiment_id, run_name=run_info.info.run_name
        ):
            # Log parameters
            for key, value in run_info.data.params.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key in run_info.data.metrics.keys():
                # Retrieve full metric history
                metric_history = local_client.get_metric_history(
                    run_info.info.run_id, key
                )
                for record in metric_history:
                    # Log each step and value
                    mlflow.log_metric(key, record.value, step=record.step)

            # Log artifacts
            artifacts_path = os.path.join(
                LOCAL_MLFLOW_DIR, exp.experiment_id, run_info.info.run_id, "artifacts"
            )
            if os.path.exists(artifacts_path):
                mlflow.log_artifacts(artifacts_path)

        print(f"Logged Run ID: {run_info.info.run_id} to DagsHub")
