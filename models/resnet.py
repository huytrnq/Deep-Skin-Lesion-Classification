"""
This is implementation of ResNet with Low-Rank Adaptation (LoRA) 
applied to convolutional layers in each block.
"""

import torch
import torch.nn as nn
from torchvision import models


class LoRAConv2d(nn.Module):
    """Low-Rank Adaptation for convolutional layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        rank=4,
        bias=True,
    ):
        """
        Low-Rank Adaptation for convolutional layers.
        """
        super().__init__()
        self.rank = rank

        # Base convolutional layer (frozen during training)
        self.base_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        for param in self.base_layer.parameters():
            param.requires_grad = False  # Freeze original weights

        # Low-rank adaptation
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_channels, kernel_size, kernel_size)
        )
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1, 1))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=1)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass with Low-Rank Adaptation.
        
        Args: x (torch.Tensor): Input tensor.
        Returns: torch.Tensor: Output tensor.
        """
        base_output = self.base_layer(x)  # Original output (frozen weights)
        lora_update = nn.functional.conv2d(
            x,
            self.lora_A,
            bias=None,
            stride=self.base_layer.stride,
            padding=self.base_layer.padding,
        )
        lora_update = nn.functional.conv2d(lora_update, self.lora_B, bias=None)
        return base_output + lora_update


class ResNetLoRA(nn.Module):
    """ResNet model with Low-Rank Adaptation (LoRA) applied to convolutional layers in each block."""

    def __init__(self, base_model_name="resnet50", rank=4, weights=None):
        """
        ResNet model with Low-Rank Adaptation (LoRA) applied to convolutional layers in each block.
        Args:
            base_model_name (str): Name of the ResNet model to use (e.g., "resnet50").
            rank (int): Rank for LoRA layers.
            weights (str): Pretrained weights to load.
        """
        super().__init__()
        self.rank = rank
        self.base_model = getattr(models, base_model_name)(weights=weights)
        self.apply_lora_to_blocks()

    def apply_lora_to_blocks(self):
        """
        Apply LoRA to the convolutional layers in each block of ResNet.
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):  # Target convolutional layers
                # Extract parameters for replacement
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                bias = module.bias is not None

                # Replace with LoRA-enhanced layer
                lora_layer = LoRAConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    rank=self.rank,
                    bias=bias,
                )

                # Replace the module in the parent
                parent_name = name.rsplit(".", 1)[0]
                parent = self.base_model.get_submodule(parent_name)
                setattr(parent, name.split(".")[-1], lora_layer)

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    from torchvision.models.resnet import ResNet50_Weights

    # Instantiate ResNet with LoRA applied to each block
    resnet_lora = ResNetLoRA(
        base_model_name="resnet50", rank=4, weights=ResNet50_Weights
    )

    # Freeze all weights except LoRA components
    for name, param in resnet_lora.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True  # Trainable
        else:
            param.requires_grad = False  # Frozen

    # Verify which parameters are trainable
    for name, param in resnet_lora.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    # Define optimizer for only LoRA parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet_lora.parameters()), lr=1e-4
    )

    # Print model to verify
    print(resnet_lora)
