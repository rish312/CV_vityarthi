"""
Model Module
=============
Defines the PlantDiseaseNet model architecture using transfer learning
with ResNet50 pretrained on ImageNet.
"""

import os
import sys

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PlantDiseaseNet(nn.Module):
    """
    Plant Disease Classification Network.

    Architecture:
        - Base: ResNet50 pretrained on ImageNet (feature extractor)
        - Head: Global Average Pool → Dropout → FC(256) → ReLU → Dropout → FC(38)

    Training Strategy:
        - Stage 1: Freeze all base layers, train only the custom head
        - Stage 2: Unfreeze top N layers of base and fine-tune with lower LR
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super(PlantDiseaseNet, self).__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet50(weights=weights)

        # Get the number of features from the last FC layer
        num_features = self.base_model.fc.in_features  # 2048 for ResNet50

        # Replace the final FC layer with our custom classifier head
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(256, num_classes),
        )

        # Store architecture info
        self.num_classes = num_classes
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.base_model(x)

    def freeze_base(self):
        """
        Freeze all layers except the custom classification head.
        Used in Stage 1 training.
        """
        # Freeze all parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the custom FC head
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        frozen = sum(1 for p in self.base_model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.base_model.parameters() if p.requires_grad)
        print(f"🔒 Base frozen: {frozen} layers frozen, {trainable} layers trainable")

    def unfreeze_top_layers(self, num_layers: int = config.STAGE2_UNFREEZE_LAYERS):
        """
        Unfreeze the top N layers of the base model for fine-tuning.
        Used in Stage 2 training.

        Args:
            num_layers: Number of layers to unfreeze from the end.
        """
        # First, freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get all named parameters as a list
        all_params = list(self.base_model.named_parameters())

        # Unfreeze the last num_layers parameters
        for name, param in all_params[-num_layers:]:
            param.requires_grad = True

        # Always unfreeze the FC head
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        trainable = sum(1 for p in self.base_model.parameters() if p.requires_grad)
        total = sum(1 for p in self.base_model.parameters())
        print(f"🔓 Fine-tuning: {trainable}/{total} layers trainable")

    def get_trainable_params(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        """Print a summary of the model architecture."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable

        print("=" * 50)
        print("  PlantDiseaseNet - Model Summary")
        print("=" * 50)
        print(f"  Base Model:       ResNet50 (ImageNet)")
        print(f"  Number of Classes: {self.num_classes}")
        print(f"  Total Parameters:  {total:,}")
        print(f"  Trainable:         {trainable:,}")
        print(f"  Frozen:            {frozen:,}")
        print("=" * 50)


def load_model(model_path: str, device: torch.device = None) -> PlantDiseaseNet:
    """
    Load a saved model from disk.

    Args:
        model_path: Path to the saved .pth file.
        device: Device to load the model on.

    Returns:
        Loaded PlantDiseaseNet model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlantDiseaseNet(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle both direct state_dict and checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


def save_model(
    model: PlantDiseaseNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    path: str,
):
    """
    Save model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        epoch: Current epoch number.
        val_accuracy: Validation accuracy at this point.
        path: File path to save to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
        },
        path,
    )
    print(f"💾 Model saved to {path} (val_acc: {val_accuracy:.4f})")


if __name__ == "__main__":
    # Quick test
    print("Testing PlantDiseaseNet...")

    model = PlantDiseaseNet()
    model.summary()

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents probabilities over {output.shape[1]} classes")

    # Test freeze/unfreeze
    model.freeze_base()
    print(f"Trainable after freeze: {model.get_trainable_params():,}")

    model.unfreeze_top_layers(20)
    print(f"Trainable after unfreeze: {model.get_trainable_params():,}")
