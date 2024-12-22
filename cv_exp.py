"""
This script trains and validates a deep learning model for skin lesion classification with cross-validation.
It includes an ensemble across folds for final predictions.
"""

import os
import argparse
import dagshub
import mlflow
import mlflow.pytorch
import mlflow

from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B6_Weights

from models.resnet import ResNetLoRA
from test import test
from utils.dataset import SkinDataset
from utils.metric import MetricsMonitor
from utils.utils import (
    train,
    validate,
    load_data_file,
    load_config,
    build_transforms,
    freeze_layers,
    export_predictions,
    compute_class_weights_from_dataset,
    log_first_batch_images,
)

dagshub.init(
    repo_owner="huytrnq", repo_name="Deep-Skin-Lesion-Classification", mlflow=True
)
mlflow.start_run(
    run_name="Skin Lesion Classification with Cross-Validation and Ensemble"
)


def arg_parser():
    """Arg parser"""
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Experiment with Cross-Validation"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer names to freeze (e.g., 'layer1,layer2,fc')",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0, help="Number of warm-up epochs"
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=0.00005, help="Learning rate during warm-up"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() // 2, help="Number of workers"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/huy/datasets/Binary",
        help="Path to data directory",
    )
    parser.add_argument(
        "--num_tta",
        type=int,
        default=10,
        help="Number of TTA iterations",
    )
    parser.add_argument(
        "--kfold", type=int, default=5, help="Number of folds for cross-validation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    # Constants
    CONFIG_PATH = "config.json"
    CLASSES = ["nevus", "others"]
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WORKERS = args.workers
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Data Transformations
    config = load_config(CONFIG_PATH)
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])

    # Load data paths and labels
    all_names, all_labels = load_data_file("datasets/Binary/all.txt")

    # Cross-validation setup
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    fold_idx = 1

    all_fold_predictions = []

    for train_index, val_index in kf.split(all_names):
        print(f"========== Fold {fold_idx}/{args.kfold} ==========")

        train_names = [all_names[i] for i in train_index]
        train_labels = [all_labels[i] for i in train_index]
        val_names = [all_names[i] for i in val_index]
        val_labels = [all_labels[i] for i in val_index]

        # Create datasets and dataloaders
        train_dataset = SkinDataset(
            args.data_root, "train", train_names, train_labels, train_transform
        )
        val_dataset = SkinDataset(
            args.data_root, "val", val_names, val_labels, test_transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
        )

        print(
            "================== Train dataset Info: ==================\n", train_dataset
        )
        print("================== Val dataset Info: ==================\n", val_dataset)

        # Model
        model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, len(CLASSES)
        )
        model = model.to(DEVICE)

        # Loss and Optimizer
        class_weights = compute_class_weights_from_dataset(train_dataset, len(CLASSES))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

        warmup_optimizer = getattr(torch.optim, args.optimizer)(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.warmup_lr,
        )
        main_optimizer = getattr(torch.optim, args.optimizer)(
            [param for param in model.parameters() if param.requires_grad], lr=LR
        )

        # Scheduler
        scheduler = CosineAnnealingLR(main_optimizer, T_max=EPOCHS, eta_min=0.00001)

        # Warm-up Phase
        for epoch in range(args.warmup_epochs):
            print(f"Warm-up Epoch {epoch + 1}/{args.warmup_epochs}")
            train(
                model,
                train_loader,
                criterion,
                warmup_optimizer,
                DEVICE,
                train_monitor=MetricsMonitor(metrics=["loss", "accuracy"]),
            )
            validate(
                model,
                val_loader,
                criterion,
                DEVICE,
                val_monitor=MetricsMonitor(metrics=["loss", "accuracy"]),
            )

        # Training Phase
        for epoch in range(args.warmup_epochs, EPOCHS):
            print(f"Training Fold {fold_idx} - Epoch {epoch + 1}/{EPOCHS}")
            train(
                model,
                train_loader,
                criterion,
                main_optimizer,
                DEVICE,
                train_monitor=MetricsMonitor(metrics=["loss", "accuracy"]),
            )
            validate(
                model,
                val_loader,
                criterion,
                DEVICE,
                val_monitor=MetricsMonitor(metrics=["loss", "accuracy"]),
            )
            scheduler.step()

        # Save predictions for ensemble
        _, _, fold_prediction_probs = test(
            model=model,
            config=config,
            data_file="datasets/Binary/val.txt",
            data_root=args.data_root,
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            device=DEVICE,
            tta=True,
            num_tta=args.num_tta,
        )
        all_fold_predictions.append(fold_prediction_probs)

        fold_idx += 1

    # Final ensemble
    print("Combining predictions from all folds...")
    ensemble_predictions = sum(all_fold_predictions) / args.kfold
    final_predictions = ensemble_predictions.argmax(axis=1)

    # Export ensemble predictions
    export_predictions(final_predictions, "results/ensemble_predictions.npy")
    mlflow.log_artifact("results/ensemble_predictions.npy", artifact_path="results")
    print("Ensemble predictions exported.")
