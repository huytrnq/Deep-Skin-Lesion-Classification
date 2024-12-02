"""
This module contains custom transformations for data augmentation.
"""

import torch
from torchvision.transforms import functional as F


class ResizeKeepRatio:
    """Resize an image while maintaining the aspect ratio by adding padding."""

    def __init__(self, size, fill_color=(0, 0, 0)):
        """
        Resize an image while maintaining the aspect ratio by adding padding.

        Args:
            size (int): Desired size of the shorter side.
            fill_color (tuple): RGB color for padding (default: black).
        """
        self.size = size
        self.fill_color = fill_color

    def __call__(self, image):
        # Get original dimensions
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        # Resize while keeping the aspect ratio
        if original_width > original_height:
            new_width = self.size
            new_height = int(self.size / aspect_ratio)
        else:
            new_height = self.size
            new_width = int(self.size * aspect_ratio)

        image = F.resize(image, (new_height, new_width))

        # Add padding to make the image square
        delta_width = self.size - new_width
        delta_height = self.size - new_height
        padding = (
            delta_width // 2,
            delta_height // 2,
            delta_width - delta_width // 2,
            delta_height - delta_height // 2,
        )
        image = F.pad(image, padding, fill=self.fill_color)

        return image


class GaussianNoiseInjection:
    """Add Gaussian noise to an image."""

    def __init__(self, mean=0, std=0.1):
        """
        Add Gaussian noise to an image.

        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Convert image to tensor
        image = F.to_tensor(image)

        # Add noise
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = image + noise

        # Clip the image to [0, 1]
        noisy_image = torch.clamp(noisy_image, 0, 1)

        # Convert tensor back to image
        noisy_image = F.to_pil_image(noisy_image)

        return noisy_image
