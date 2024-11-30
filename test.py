"""
This script tests a trained deep learning model for skin lesion classification.
It uses PyTorch for model evaluation.

Run the script using the following command:
    python exp.py --batch_size 64 --workers 4
"""

import os
import argparse
import torch
import mlflow.pytorch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import SkinDataset
from utils.utils import test, load_data_file
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
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    return parser.parse_args()


if __name__ == "__main__":
    MODEL_URI = "runs:/11d6baa8742d485baddbb12fdeb9530e/skin_lesion_model"

    args = arg_parser()

    # Data Transformations
    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Constants
    CLASSES = ["nevus", "others"]
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load test data paths and labels
    test_path, test_labels = load_data_file("datasets/val.txt")

    # Create test dataset and dataloader
    test_dataset = SkinDataset(test_path, test_labels, transform)
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