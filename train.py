"""
Training Module
================
Handles the two-stage training process:
  Stage 1: Train classification head only (base frozen)
  Stage 2: Fine-tune top layers + head (with lower learning rate)
"""

import os
import sys
import time
import json
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import PlantDiseaseNet, save_model


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"⏹️  Early stopping triggered (patience={self.patience})")
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_loader,
        desc=f"  Epoch {epoch}/{total_epochs} [Train]",
        leave=False,
        ncols=100,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.1f}%",
        )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        val_loader,
        desc=f"  Epoch {epoch}/{total_epochs} [Val]  ",
        leave=False,
        ncols=100,
    )

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: PlantDiseaseNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage: int = 1,
) -> Dict[str, List[float]]:
    """
    Train the model using the specified stage configuration.

    Args:
        model: The PlantDiseaseNet model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device to train on.
        stage: Training stage (1 = head only, 2 = fine-tune).

    Returns:
        Dictionary containing training history.
    """
    # Stage-specific configuration
    if stage == 1:
        epochs = config.STAGE1_EPOCHS
        lr = config.STAGE1_LR
        model.freeze_base()
        print(f"\n{'='*60}")
        print(f"  🚀 STAGE 1: Training Classification Head")
        print(f"     Epochs: {epochs} | LR: {lr}")
        print(f"     Trainable params: {model.get_trainable_params():,}")
        print(f"{'='*60}\n")
    else:
        epochs = config.STAGE2_EPOCHS
        lr = config.STAGE2_LR
        model.unfreeze_top_layers(config.STAGE2_UNFREEZE_LAYERS)
        print(f"\n{'='*60}")
        print(f"  🔬 STAGE 2: Fine-Tuning Top Layers")
        print(f"     Epochs: {epochs} | LR: {lr}")
        print(f"     Trainable params: {model.get_trainable_params():,}")
        print(f"{'='*60}\n")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
        verbose=True,
    )
    early_stopping = EarlyStopping()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, epochs
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(
            f"  Epoch {epoch:2d}/{epochs} │ "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} │ "
            f"LR: {current_lr:.2e} │ "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, epoch, val_acc, config.BEST_MODEL_PATH)
            print(f"  ⭐ New best model! Val Acc: {val_acc:.4f}")

        # Early stopping check
        if early_stopping(val_loss):
            break

    total_time = time.time() - start_time
    print(f"\n  ✅ Stage {stage} complete in {total_time:.1f}s")
    print(f"  📊 Best validation accuracy: {best_val_acc:.4f}\n")

    return history


def full_training_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = None,
) -> Tuple[PlantDiseaseNet, Dict[str, List[float]]]:
    """
    Execute the full two-stage training pipeline.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device to train on.

    Returns:
        Tuple of (trained_model, combined_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥️  Training device: {device}")

    # Initialize model
    model = PlantDiseaseNet().to(device)
    model.summary()

    # Stage 1: Train head only
    history_1 = train_model(model, train_loader, val_loader, device, stage=1)

    # Stage 2: Fine-tune
    history_2 = train_model(model, train_loader, val_loader, device, stage=2)

    # Combine histories
    combined_history = {
        key: history_1[key] + history_2[key]
        for key in history_1.keys()
    }

    # Save final model
    save_model(
        model,
        optim.Adam(model.parameters()),  # dummy optimizer for final save
        config.STAGE1_EPOCHS + config.STAGE2_EPOCHS,
        combined_history["val_acc"][-1],
        config.FINAL_MODEL_PATH,
    )

    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"📈 Training history saved to {history_path}")

    return model, combined_history


if __name__ == "__main__":
    from src.data_loader import download_dataset, create_data_loaders

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = download_dataset()
    train_loader, val_loader, test_loader, class_names = create_data_loaders(data_dir)

    # Train
    model, history = full_training_pipeline(train_loader, val_loader, device)

    print("\n🎉 Training complete!")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
