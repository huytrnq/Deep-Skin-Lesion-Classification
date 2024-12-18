"""
This module contains custom transformations for data augmentation.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image

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


class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = ""):
        """
        Augments an image by overlaying random hair images.

        Args:
            hairs (int): Maximum number of hairs to overlay.
            hairs_folder (str): Folder containing hair PNG images.
        """
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Apply hair augmentation to the given image.

        Args:
            img (PIL.Image or np.ndarray): The input image.

        Returns:
            PIL.Image: The augmented image.
        """
        import random
        import cv2
        import numpy as np
        from PIL import Image
        import os

        n_hairs = random.randint(0, self.hairs)

        if n_hairs == 0:
            return img

        # Convert PIL Image to OpenCV image
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        height, width, _ = img.shape  # Target image dimensions
        hair_images = [
            im for im in os.listdir(self.hairs_folder) if im.endswith(".png")
        ]

        for _ in range(n_hairs):
            hair_path = os.path.join(self.hairs_folder, random.choice(hair_images))
            hair = cv2.imread(
                hair_path, cv2.IMREAD_UNCHANGED
            )  # Read with alpha channel

            # Skip if hair image is too large for the target image
            if hair.shape[0] > height or hair.shape[1] > width:
                continue

            # Apply random transformations to the hair
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))  # Flip randomly
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))  # Rotate randomly

            # Extract dimensions and create ROI
            h_height, h_width, _ = hair.shape
            roi_ho = random.randint(0, height - h_height)
            roi_wo = random.randint(0, width - h_width)
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            # Create mask and overlay hair
            if hair.shape[2] == 4:  # Check if hair has an alpha channel
                alpha_channel = hair[:, :, 3]  # Extract alpha channel
                _, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                hair_rgb = hair[:, :, :3]  # Extract RGB channels

                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                hair_fg = cv2.bitwise_and(hair_rgb, hair_rgb, mask=mask)

                dst = cv2.add(img_bg, hair_fg)
                img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        # Convert image back to PIL format
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img
