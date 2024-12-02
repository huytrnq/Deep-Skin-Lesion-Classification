"""
This script trains and validates a deep learning model for skin lesion classification.
It uses MLflow for tracking experiments and PyTorch for model training.

Run the script using the following command:
    python exp.py --batch_size 64 --epochs 100 --lr 0.001 --workers 4 --warmup_epochs 10 --warmup_lr 0.00005
"""

import os
import argparse
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import swin_t, Swin_T_Weights

from models.resnet import ResNetLoRA
from utils.dataset import SkinDataset
from utils.metric import MetricsMonitor
from utils.utils import (
    train,
    validate,
    test,
    load_data_file,
    load_config,
    build_transforms,
    freeze_layers,
)


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Experiment"
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

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # MLflow Experiment Setup
    mlflow.set_experiment("Skin Lesion Classification")

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
    # Load the configuration file
    config = load_config(CONFIG_PATH)
    # Build train and test transforms from the configuration
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])
    print("Train Transformations:", train_transform)
    print("Test Transformations:", test_transform)

    # Load data paths and labels
    train_names, train_labels = load_data_file("datasets/train.txt")
    train_names, val_names, train_labels, val_labels = train_test_split(
        train_names, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    test_names, test_labels = load_data_file("datasets/val.txt")

    # Create datasets and dataloaders
    # Split the data into train, validation and using validation data as test data
    train_dataset = SkinDataset(
        args.data_root, "train", train_names, train_labels, train_transform
    )
    val_dataset = SkinDataset(
        args.data_root, "train", val_names, val_labels, test_transform
    )
    test_dataset = SkinDataset(
        args.data_root, "val", test_names, test_labels, test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS
    )

    print("================== Train dataset Info: ==================\n", train_dataset)
    print("================== Val dataset Info: ==================\n", val_dataset)
    print("================== Test dataset Info: ==================\n", test_dataset)

    # Model
    # model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # model.fc = torch.nn.Linear(2048, len(CLASSES))

    # model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(1536, len(CLASSES))
    # model = ResNetLoRA(weights=ResNet50_Weights.DEFAULT, num_classes=len(CLASSES), rank=64)
    # Load pre-trained ViT-B-16 model
    # Load the ViT-L/16 model with pretrained weights
    # Load the pre-trained Swin Transformer model
    # weights = Swin_T_Weights.DEFAULT
    # model = swin_t(weights=weights)

    # # Modify the classifier head for your specific task
    # num_classes = len(CLASSES)  # Replace with the number of output classes
    # model.head = torch.nn.Sequential(
    #     torch.nn.Linear(model.head.in_features, 512),  # Intermediate dense layer
    #     torch.nn.ReLU(),  # Non-linear activation
    #     torch.nn.Dropout(0.5),  # Regularization
    #     torch.nn.Linear(512, num_classes),  # Final classification layer
    # )
    model = model.to(DEVICE)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Monitors
    train_monitor = MetricsMonitor(metrics=["loss", "accuracy"])
    val_monitor = MetricsMonitor(
        metrics=["loss", "accuracy"], patience=args.patience, mode="max"
    )
    test_monitor = MetricsMonitor(metrics=["loss", "accuracy"])

    # Warm-up settings
    WARMUP_EPOCHS = args.warmup_epochs  # Number of warm-up epochs
    WARMUP_LR = args.warmup_lr  # Learning rate during warm-up

    # Training Loop
    with mlflow.start_run():
        # Log Artifacts
        mlflow.log_artifact(CONFIG_PATH, artifact_path="config")
        # Log Parameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("warmup_lr", WARMUP_LR)
        mlflow.log_param("warmup_epochs", WARMUP_EPOCHS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("classes", CLASSES)
        mlflow.log_param("model", model.__class__.__name__)

        # Optimizer setup for warm-up phase
        warmup_optimizer = getattr(torch.optim, args.optimizer)(
            [param for param in model.parameters() if param.requires_grad], lr=WARMUP_LR
        )
        main_optimizer = getattr(torch.optim, args.optimizer)(
            [param for param in model.parameters() if param.requires_grad], lr=LR
        )

        # Scheduler: Cosine Annealing
        scheduler = CosineAnnealingLR(main_optimizer, T_max=EPOCHS, eta_min=0.00001)

        # Warm-up Phase
        print("====================== Warm-up phase ======================")
        for epoch in range(WARMUP_EPOCHS):
            print(f"Warm-up Epoch {epoch + 1}/{WARMUP_EPOCHS}")
            # Unfreeze all layers during warm-up
            for param in model.parameters():
                param.requires_grad = True

            train(
                model,
                train_loader,
                criterion,
                warmup_optimizer,
                DEVICE,
                train_monitor,
            )

            # Validation during warm-up
            validate(model, val_loader, criterion, DEVICE, val_monitor)

            # Log Metrics
            train_loss = train_monitor.compute_average("loss")
            train_acc = train_monitor.compute_average("accuracy")
            val_loss = val_monitor.compute_average("loss")
            val_acc = val_monitor.compute_average("accuracy")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Training Phase
        print("====================== Training phase ======================")
        # Freeze layers

        freeze_layers(model, args.freeze_layers)
        for epoch in range(WARMUP_EPOCHS, EPOCHS):
            print(f"Training Epoch {epoch + 1}/{EPOCHS}")

            train(
                model,
                train_loader,
                criterion,
                main_optimizer,
                DEVICE,
                train_monitor,
            )

            validate(model, val_loader, criterion, DEVICE, val_monitor)

            # Adjust learning rate with cosine scheduler
            scheduler.step()  # Update the learning rate based on the scheduler

            # Log Metrics
            train_loss = train_monitor.compute_average("loss")
            train_acc = train_monitor.compute_average("accuracy")
            val_loss = val_monitor.compute_average("loss")
            val_acc = val_monitor.compute_average("accuracy")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # Early Stopping
            if val_monitor.early_stopping_check(val_acc, model):
                print("Early stopping triggered.")
                break

        # Test Phase
        test(model, test_loader, criterion, DEVICE, test_monitor)
        test_loss = test_monitor.compute_average("loss")
        test_acc = test_monitor.compute_average("accuracy")
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log the Model
        mlflow.pytorch.log_model(model, artifact_path="skin_lesion_model")
        print("Model logged to MLflow.")
