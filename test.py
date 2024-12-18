import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import dagshub
import mlflow.pytorch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import SkinDataset
from utils.utils import (
    load_data_file,
    load_config,
    build_transforms,
    export_predictions,
)
from sklearn.metrics import cohen_kappa_score


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: Command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Testing Script"
    )
    parser.add_argument(
        "--tta", default=False, help="Use Test Time Augmentation", action="store_true"
    )
    parser.add_argument(
        "--num_tta",
        type=int,
        default=10,
        help="Number of TTA iterations",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() // 2,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/huy/datasets/Binary",
        help="Path to data directory",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/Binary/val.txt",
        help="Path to the file containing test data paths and labels",
    )
    parser.add_argument(
        "--log_kappa",
        default=False,
        help="Log Cohen's Kappa Score",
        action="store_true",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="aca333832cbf492981651b12b6f27c84",
        help="MLflow run ID",
    )
    return parser.parse_args()


def tta_step_batch(model, images, transform, num_tta=5, device="cpu"):
    """Perform Test Time Augmentation (TTA) for a batch of images."""
    model.eval()
    tta_outputs = []

    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for _ in range(num_tta):
            augmented_images = []
            for img in images:
                pil_img = to_pil(img.cpu())
                augmented_images.append(transform(pil_img))
            augmented_images = torch.stack(augmented_images).to(device)

            # Perform inference
            output = model(augmented_images)
            tta_outputs.append(softmax(output, dim=1))

    # Average predictions across TTA iterations
    avg_output = torch.stack(tta_outputs).mean(dim=0)
    return avg_output


def test(
    model,
    config,
    data_file,
    data_root,
    batch_size,
    num_workers,
    device="cuda",
    tta=False,
    num_tta=5,
    log_kappa=False,
):
    """Test a trained model on a dataset."""
    # Data Transformations
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])
    basic_transform = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor()]
    )

    # Load test data
    test_names, test_labels = load_data_file(data_file)

    # Create test dataset and dataloader
    test_dataset = SkinDataset(
        data_root,
        "val",
        test_names,
        test_labels,
        basic_transform if tta else test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("================== Test dataset Info: ==================\n", test_dataset)

    # Testing Phase
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(
            test_loader, desc="Testing TTA" if tta else "Testing"
        ):
            batch_labels = batch_labels.to(device)

            if tta:
                # Perform TTA
                tta_output = tta_step_batch(
                    model,
                    batch_images,
                    train_transform,
                    num_tta=num_tta,
                    device=device,
                )
                batch_probs = tta_output
            else:
                # Standard Inference
                batch_images = batch_images.to(device)
                output = model(batch_images)
                batch_probs = softmax(output, dim=1)

            all_probs.append(batch_probs)
            all_labels.append(batch_labels)

    # Concatenate all probabilities and labels
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Calculate discrete predictions
    all_preds = all_probs.argmax(axis=1)

    # Calculate accuracy
    test_acc = (all_preds == all_labels).mean()

    # Calculate and log Kappa Score if enabled
    kappa_score = None
    if log_kappa:
        kappa_score = cohen_kappa_score(all_labels, all_preds)

    return test_acc, kappa_score, all_probs


def load_model_and_config(run_id, artifact_path="config.json", device="cuda"):
    """Load the trained model and configuration."""
    model_uri = f"runs:/{run_id}/skin_lesion_model"
    config_path = "config.json"

    # Download the configuration file from the run
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    if Path(local_path).is_file():
        print(f"Config file downloaded to: {local_path}")
        config_path = local_path

    # Load model
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device)

    # Load config
    config = load_config(config_path)
    return model, config


def main(args):
    """Main function to test the model."""
    # Constants
    RUN_ID = args.run_id
    ARTIFACT_PATH = "config/config.json"

    dagshub.init(
        repo_owner="huytrnq", repo_name="Deep-Skin-Lesion-Classification", mlflow=True
    )

    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {DEVICE}")
    # Load model and configuration
    model, config = load_model_and_config(RUN_ID, ARTIFACT_PATH, DEVICE)

    with mlflow.start_run(run_id=RUN_ID):
        run_name = mlflow.active_run().info.run_name
        print(f"MLflow run: {run_name}")
        # Test the model
        test_acc, kappa_score, prediction_probs = test(
            model,
            config,
            args.data_file,
            args.data_root,
            args.batch_size,
            args.num_workers,
            DEVICE,
            tta=args.tta,
            num_tta=args.num_tta,
            log_kappa=args.log_kappa,
        )
        print(f"Test Accuracy: {test_acc:.4f}")
        if args.log_kappa:
            print(f"Cohen's Kappa Score: {kappa_score:.4f}")

        ## Save predictions
        export_name = (
            "prediction_probs.npy" if not args.tta else "tta_prediction_probs.npy"
        )
        export_path = (
            f"results/Binary/{export_name}"
            if "Binary" in args.data_root
            else f"results/Multiclass/{export_name}"
        )
        export_predictions(prediction_probs, export_path)

        ## Log metrics
        mlflow.log_metric(
            "test_accuracy" if not args.tta else "test_accuracy_tta", test_acc
        )
        if args.log_kappa:
            mlflow.log_metric(
                "test_kappa_score" if args.tta else "test_kappa_score_tta", kappa_score
            )
        mlflow.log_artifact(export_path, artifact_path="results")


if __name__ == "__main__":
    args = arg_parser()
    os.environ["MLFLOW_TRACKING_URI"] = (
        "https://dagshub.com/huytrnq/Deep-Skin-Lesion-Classification.mlflow"
    )
    main(args)
