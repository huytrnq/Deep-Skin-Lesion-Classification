"""
This script tests a trained deep learning model for skin lesion classification.
It uses PyTorch for model evaluation.

Run the script using the following command:
    python exp.py --batch_size 64 --workers 4
"""

import os
import argparse
from pathlib import Path

import torch
import mlflow.pytorch
from torch.utils.data import DataLoader

from utils.dataset import SkinDataset
from utils.utils import test, load_data_file, load_config, build_transforms
from utils.metric import MetricsMonitor


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Testing Script"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--tta", default=False, help="Use Test Time Augmentation", action='store_true')
    parser.add_argument(
        "--data_root", type=str, default="/root/huy/datasets/Binary", help="Path to data directory"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    
    # Constants
    RUN_ID = '328ea8d3d27f48bfa57b8caa09698749'
    MODEL_URI = f"runs:/{RUN_ID}/skin_lesion_model"
    ARTIFACT_PATH = "config/config.json"  # Path to the artifact in the run

    CONFIG_PATH = "config.json"
    # Download the configuration file from the run
    local_path = mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path=ARTIFACT_PATH)
    if  Path(local_path).is_file():
        print(f"Config file downloaded to: {local_path}")
        CONFIG_PATH = local_path

    CLASSES = ["nevus", "others"]
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Data Transformations
    # Load the configuration file
    config = load_config(CONFIG_PATH)
    # Build train and test transforms from the configuration
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])

    # Load test data paths and labels
    test_names, test_labels = load_data_file("datasets/val.txt")

    # Create test dataset and dataloader
    if args.tta:
        print("================== Using Test Time Augmentation ==================")
        test_dataset = SkinDataset(args.data_root, 'val', test_names, test_labels, train_transform)
    else:
        test_dataset = SkinDataset(args.data_root, 'val', test_names, test_labels, test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
    )

    print("================== Test dataset Info: ==================\n", test_dataset)

    # Load Model
    model = mlflow.pytorch.load_model(MODEL_URI)
    model = model.to(DEVICE)

    # Monitors
    test_monitor = MetricsMonitor(metrics=["accuracy"])

    # Test Phase
    test(model, test_loader, None, DEVICE, test_monitor)
    test_acc = test_monitor.compute_average("accuracy")

    print(f"Test Accuracy: {test_acc:.4f}")
