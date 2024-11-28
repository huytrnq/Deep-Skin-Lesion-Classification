"""
This module contains custom transformations for data augmentation.
"""
import torch
from torchvision.transforms import functional as F
from PIL import Image

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
        padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
        image = F.pad(image, padding, fill=self.fill_color)

        return image