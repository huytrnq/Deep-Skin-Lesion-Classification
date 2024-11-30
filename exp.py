"""
This script trains and validates a deep learning model for skin lesion classification.
It uses MLflow for tracking experiments and PyTorch for model training.

Run the script using the following command:
    python exp.py --batch_size 64 --epochs 100 --lr 0.001 --workers 4 --img_size 400 --warmup_epochs 10 --warmup_lr 0.00005
"""

import os
import argparse
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.efficientnet import EfficientNet_B1_Weights, EfficientNet_B2_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR 

from utils.dataset import SkinDataset
from utils.utils import train, validate, test, load_data_file
from utils.metric import MetricsMonitor
from utils.transform import ResizeKeepRatio

def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Experiment"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, default=400, help="Image size for training"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warm-up epochs"
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=0.00005, help="Learning rate during warm-up"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    parser.add_argument(
        "--data_root", type=str, default="/root/huy/datasets/Binary", help="Path to data directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # MLflow Experiment Setup
    mlflow.set_experiment("Skin Lesion Classification")

    # Data Transformations
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
            ),
            # transforms.ColorJitter(
            #     brightness=(0.7, 1.3),  # Random brightness factor
            #     contrast=(0.7, 1.3),    # Random contrast factor
            #     saturation=(0.7, 1.3),  # Random saturation factor
            # ),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Constants
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

    # Load data paths and labels
    train_names, train_labels = load_data_file("datasets/train.txt")
    train_names, val_names, train_labels, val_labels = train_test_split(
        train_names, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    test_names, test_labels = load_data_file("datasets/val.txt")

    # Create datasets and dataloaders
    # Split the data into train, validation and using validation data as test data
    train_dataset = SkinDataset(args.data_root, 'train', train_names, train_labels, transform)
    val_dataset = SkinDataset(args.data_root, 'train', val_names, val_labels, transform)
    test_dataset = SkinDataset(args.data_root, 'val', test_names, test_labels, test_transform)

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
    # model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(1408, len(CLASSES))
    model = model.to(DEVICE)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Monitors
    train_monitor = MetricsMonitor(metrics=["loss", "accuracy"])
    val_monitor = MetricsMonitor(metrics=["loss", "accuracy"], patience=5, mode="max")
    test_monitor = MetricsMonitor(metrics=["loss", "accuracy"])

    # Warm-up settings
    WARMUP_EPOCHS = args.warmup_epochs  # Number of warm-up epochs
    WARMUP_LR = args.warmup_lr  # Learning rate during warm-up

    # Training Loop
    with mlflow.start_run():
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
        warmup_optimizer = torch.optim.Adam(model.parameters(), lr=WARMUP_LR)
        main_optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad], lr=LR
        )

        # Scheduler: Cosine Annealing
        scheduler = CosineAnnealingLR(main_optimizer, T_max=EPOCHS, eta_min=0.00001)
        
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")

            # Warm-up phase
            if epoch < WARMUP_EPOCHS:
                print("====================== Warm-up phase ======================")
                # Unfreeze all layers
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
            else:
                print("====================== Training phase ======================")
                # Freeze feature extraction layers after warm-up
                for name, param in model.named_parameters():
                    if "features" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

                # Training with main optimizer
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