# src/train.py
import os
import mlflow
import subprocess
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_git_info():
    def run(cmd):
        return subprocess.check_output(cmd, shell=True).decode().strip()

    try:
        commit = run("git rev-parse HEAD")
        branch = run("git rev-parse --abbrev-ref HEAD")
        status = run("git status --porcelain")
        dirty = len(status) > 0
    except Exception:
        commit = "N/A"
        branch = "N/A"
        dirty = False

    return commit, branch, dirty

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Parcel-Damage-Classification")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        commit, branch, dirty = get_git_info()
        mlflow.log_param("git_commit", commit)
        mlflow.log_param("git_branch", branch)
        mlflow.log_param("git_dirty", dirty)
        mlflow.log_param("epochs", 76)
        mlflow.log_param("imgsz", 640)
        mlflow.log_param("dataset_path", "data")

        model = YOLO("yolo11n-cls.pt")

        results = model.train(
            data="data",
            epochs=1,
            imgsz=640,
            project="runs/classify",
            name=run_id,   
        )

        # Log metrics if available
        if hasattr(results, "metrics"):
            for k, v in results.metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

        # Log best weights
        best_weights = f"runs/classify/{run_id}/weights/best.pt"
        if os.path.exists(best_weights):
            mlflow.log_artifact(best_weights)

        # Log all training artifacts
        results_dir = f"runs/classify/{run_id}"
        if os.path.exists(results_dir):
            mlflow.log_artifacts(results_dir)

        print("Training completed and logged to MLflow.")

if __name__ == "__main__":
    main()