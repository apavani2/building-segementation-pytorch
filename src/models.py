import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from typing import Any


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for semantic segmentation.

    Attributes:
        model (nn.Module): The DeepLabV3+ model with a ResNet-101 backbone.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        """
        Initializes the DeepLabV3+ model.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary segmentation.
            pretrained (bool): Whether to use pretrained weights. Default is True.
        """
        super(DeepLabV3Plus, self).__init__()
        self.model = segmentation.deeplabv3_resnet101(
            weights=(
                segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            )
        )

        # Modify the classifier to output the desired number of classes
        self.model.classifier[-1] = nn.Conv2d(
            self.model.classifier[-1].in_channels, num_classes, kernel_size=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with sigmoid activation applied.
        """
        return torch.sigmoid(self.model(x)["out"])


def get_model(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """
    Creates and returns an instance of the DeepLabV3Plus model.

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary segmentation.
        pretrained (bool): Whether to use pretrained weights. Default is True.

    Returns:
        nn.Module: The DeepLabV3Plus model.
    """
    model = DeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)
    return model
