"""Utility functions for training and testing the model."""

import json

import numpy as np

import torch
from torchvision.transforms import transforms
from sklearn.metrics import cohen_kappa_score

from utils.transform import GaussianNoiseInjection

CUSTOM_TRANSFORMS = {
    "GaussianNoiseInjection": GaussianNoiseInjection,
}


def freeze_layers(model, layers):
    """
    Freeze the specified layers of the model.

    Args:
        model: The PyTorch model.
        layers (list): List of layer names to freeze.
    """
    # Parse the freeze layers argument
    # Freeze specified layers
    if layers:
        layers = layers.split(",")
        print(f"Freezing specified layers: {layers}")
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
            else:
                param.requires_grad = True


def load_data_file(input_file):
    """Load image names and labels from txt file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        list: List of image names.
        list: List of labels.
    """
    names, labels = [], []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, class_id = line.rsplit(" ", 1)
            # Append the path and label
            names.append(path)
            labels.append(int(class_id))

    return names, labels


def load_config(config_path):
    """Load the configuration file.
    Args: config_path (str): Path to the configuration file.
    Returns: dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def build_transforms(transform_config):
    """Build a transform pipeline dynamically from the configuration."""
    transform_list = []
    for transform_name, params in transform_config.items():
        # Check if the transform is a torchvision transform
        if hasattr(transforms, transform_name):
            transform_class = getattr(transforms, transform_name)
        # Check if it's a custom transform
        elif transform_name in CUSTOM_TRANSFORMS:
            transform_class = CUSTOM_TRANSFORMS[transform_name]
        else:
            raise ValueError(f"Unknown transform: {transform_name}")

        # Add the transform with parameters if provided
        if isinstance(params, dict):
            transform_list.append(transform_class(**params))
        else:
            transform_list.append(transform_class())
    return transforms.Compose(transform_list)


def compute_class_weights_from_dataset(dataset, num_classes):
    """
    Compute class weights based on the class distribution in the dataset.

    Args:
        dataset: Dataset object that has a `_get_class_distribution` method.
        num_classes: Total number of classes.

    Returns:
        torch.Tensor: Computed class weights.
    """
    if not hasattr(dataset, "_get_class_distribution"):
        raise AttributeError("Dataset must have a '_get_class_distribution' method.")

    # Get class distribution from the dataset
    class_distribution = dataset._get_class_distribution()

    # Total samples in the dataset
    total_samples = sum(class_distribution.values())

    # Compute weights inversely proportional to class frequencies
    class_weights = [
        total_samples / (num_classes * class_distribution.get(i, 1))
        for i in range(num_classes)
    ]
    return torch.tensor(class_weights, dtype=torch.float32)


def export_predictions(
    predictions,
    export_path,
):
    """
    Export the predictions to a npy file.

    Args:
        predictions (list): List of predicted class labels.
        export_path (str): Path to the npy file to save the predictions.
    """
    # Save the predictions to a npy file
    with open(export_path, "wb") as f:
        np.save(f, predictions)


def train(model, dataloader, criterion, optimizer, device, monitor, log_kappa=False):
    """
    Train the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer for the model.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.train()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    for iteration, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)

        # Collect predictions and labels for kappa score
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        monitor.update("loss", loss.item(), count=labels.size(0))
        monitor.update("accuracy", accuracy, count=labels.size(0))
        # Compute Kappa Score if enabled
        if log_kappa:
            kappa = cohen_kappa_score(predicted.cpu().numpy(), labels.cpu().numpy())
            monitor.update("kappa", kappa, count=len(all_labels))
        monitor.print_iteration(iteration, total_iterations, phase="Train")

    monitor.print_final(phase="Train")


def validate(model, dataloader, criterion, device, monitor, log_kappa=False):
    """
    Validate the model on the validation dataset.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.eval()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for iteration, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Metrics
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Collect predictions and labels for kappa score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            monitor.update("loss", loss.item(), count=labels.size(0))
            monitor.update("accuracy", accuracy, count=labels.size(0))
            # Compute Kappa Score if enabled
            if log_kappa:
                kappa = cohen_kappa_score(predicted.cpu().numpy(), labels.cpu().numpy())
                monitor.update("kappa", kappa, count=len(all_labels))
            monitor.print_iteration(iteration, total_iterations, phase="Validation")

    monitor.print_final(phase="Validation")


def test(model, dataloader, criterion, device, monitor, log_kappa=False):
    """
    Test the model on the test dataset.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the test data.
        criterion: Loss function.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.eval()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for iteration, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)

            # Metrics
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Collect predictions and labels for kappa score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            monitor.update("accuracy", accuracy, count=labels.size(0))
            if criterion is not None:
                monitor.update("loss", loss.item(), count=labels.size(0))
            # Compute Kappa Score if enabled
            if log_kappa:
                kappa = cohen_kappa_score(predicted.cpu().numpy(), labels.cpu().numpy())
                monitor.update("kappa", kappa, count=len(all_labels))
            monitor.print_iteration(iteration, total_iterations, phase="Test")

    monitor.print_final(phase="Test")
