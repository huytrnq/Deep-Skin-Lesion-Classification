"""
This module contains a custom PyTorch dataset class for the skin lesion dataset.
It is used to load the images and their corresponding labels (if available) from the disk.
"""

from collections import Counter
from torch.utils.data import Dataset
from PIL import Image


class SkinDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None, inference=False):
        """
        Args:
            paths (list): List of file paths to the images.
            labels (list, optional): List of corresponding class IDs for the images.
                                    Required if inference=False.
            transform (callable, optional): Optional transform to apply to the images.
            inference (bool): If True, the dataset will not expect labels (default=False).
        """
        self.paths = paths
        self.labels = labels if not inference else None
        self.transform = transform
        self.inference = inference

        if not self.inference and self.labels is None:
            raise ValueError("Labels must be provided if inference=False.")
        if not self.inference and len(paths) != len(labels):
            raise ValueError("Paths and labels must have the same length.")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.paths)

    def __str__(self):
        """Return a string representation of the dataset."""
        str_repr = f"Dataset: {len(self)} samples\n"
        if not self.inference:
            str_repr += f"Class distribution: {self._get_class_distribution()}\n"
        return str_repr

    def _get_class_distribution(self):
        """Calculate the class distribution."""

        if self.labels:
            class_counts = Counter(self.labels)
            return dict(class_counts)
        return {}

    def __getitem__(self, idx):
        """Fetch the image (and label if available) at the given index."""
        img_path = self.paths[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if needed

        # Apply transformations, if provided
        if self.transform:
            image = self.transform(image)

        if self.inference:
            return image

        label = self.labels[idx]
        return image, label
