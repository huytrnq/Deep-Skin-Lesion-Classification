"""
This script trains and validates a deep learning model for skin lesion classification.
It uses MLflow for tracking experiments and PyTorch for model training.
"""

import os
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from utils.dataset import SkinDataset
from utils.utils import train, validate, test, load_data_file
from utils.metric import MetricsMonitor

# MLflow Experiment Setup
mlflow.set_experiment("Skin Lesion Classification")

# Data Transformations
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
        ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Constants
CLASSES = ["nevus", "others"]
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.01
WORKERS = os.cpu_count()
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

if __name__ == "__main__":

    # Load data paths and labels
    train_path, train_labels = load_data_file("datasets/train.txt")
    train_path, val_path, train_labels, val_labels = train_test_split(
        train_path, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    test_path, test_labels = load_data_file("datasets/val.txt")

    # Create datasets and dataloaders
    train_dataset = SkinDataset(train_path, train_labels, transform)
    val_dataset = SkinDataset(val_path, val_labels, transform)
    test_dataset = SkinDataset(test_path, test_labels, transform)

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
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Monitors
    train_monitor = MetricsMonitor(metrics=["loss", "accuracy"])
    val_monitor = MetricsMonitor(metrics=["loss", "accuracy"], patience=5, mode="max")
    test_monitor = MetricsMonitor(metrics=["loss", "accuracy"])

    # MLflow Tracking
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("device", DEVICE)

        # Training Loop
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train(model, train_loader, criterion, optimizer, DEVICE, train_monitor)
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
        mlflow.pytorch.log_model(model, "skin_lesion_model")
        print("Model logged to MLflow.")
