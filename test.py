"""
This script tests a trained deep learning model for skin lesion classification.
It uses PyTorch for model evaluation.

Run the script using the following command:
    python exp.py --tta
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import mlflow.pytorch
from torch.nn.functional import softmax
from torchvision import transforms

from utils.dataset import SkinDataset
from utils.utils import load_data_file, load_config, build_transforms
from utils.metric import MetricsMonitor


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Testing Script"
    )
    parser.add_argument(
        "--tta", default=False, help="Use Test Time Augmentation", action="store_true"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/huy/datasets/Binary",
        help="Path to data directory",
    )
    return parser.parse_args()


def tta_step(model, image, transform, num_tta=5, device="cpu"):
    """
    Perform a single TTA step on an image.

    Args:
        model (torch.nn.Module): Trained model.
        image (PIL.Image or torch.Tensor): Input image.
        transform (callable): Transformation to apply for TTA.
        num_tta (int): Number of TTA iterations.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Averaged predictions across TTA iterations.
    """
    model.eval()
    tta_outputs = []

    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for _ in range(num_tta):
            # Apply augmentation
            # Convert tensor to PIL Image if needed
            if isinstance(image, torch.Tensor):
                image = to_pil(image)
            augmented_image = transform(image)
            augmented_image = augmented_image.unsqueeze(0).to(
                device
            )  # Add batch dimension

            # Perform inference
            output = model(augmented_image)
            tta_outputs.append(softmax(output, dim=1))

    # Average predictions across TTA iterations
    avg_output = torch.stack(tta_outputs).mean(dim=0)
    return avg_output


if __name__ == "__main__":
    args = arg_parser()

    # Constants
    RUN_ID = "9bee903dfb44461aa10eb8386f19c6fa"
    MODEL_URI = f"runs:/{RUN_ID}/skin_lesion_model"
    ARTIFACT_PATH = "config/config.json"  # Path to the artifact in the run

    CONFIG_PATH = "config.json"
    # Download the configuration file from the run
    local_path = mlflow.artifacts.download_artifacts(
        run_id=RUN_ID, artifact_path=ARTIFACT_PATH
    )
    if Path(local_path).is_file():
        print(f"Config file downloaded to: {local_path}")
        CONFIG_PATH = local_path

    CLASSES = ["nevus", "others"]
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Data Transformations
    config = load_config(CONFIG_PATH)
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])
    basic_transform = transforms.Compose([transforms.ToTensor()])

    # Load test data paths and labels
    test_names, test_labels = load_data_file("datasets/val.txt")

    # Create test dataset
    test_dataset = SkinDataset(
        args.data_root, "val", test_names, test_labels, basic_transform
    )
    print("================== Test dataset Info: ==================\n", test_dataset)

    # Load Model
    model = mlflow.pytorch.load_model(MODEL_URI)
    model = model.to(DEVICE)

    # Testing Phase
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            image, label = test_dataset[idx]
            label = torch.tensor([label]).to(DEVICE)

            if args.tta:
                # Perform TTA
                tta_output = tta_step(
                    model, image, train_transform, num_tta=5, device=DEVICE
                )
                pred = tta_output.argmax(dim=1)
            else:
                # Standard Inference
                image = image.unsqueeze(0).to(DEVICE)  # Add batch dimension
                output = model(image)
                pred = output.argmax(dim=1)

            all_preds.append(pred)
            all_labels.append(label)

    # Calculate Accuracy
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc = (all_preds == all_labels).float().mean().item()

    print(f"Test Accuracy: {test_acc:.4f}")
