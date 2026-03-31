"""
Utility Functions
==================
Helper functions for visualization, image processing, and general utilities.
"""

import os
import sys
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from PIL import Image
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("🖥️  Using CPU")
    return device


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize an ImageNet-normalized tensor back to [0, 1] range."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    return np.clip(image, 0, 1)


def plot_sample_images(data_dir: str = None, save_path: str = None, n_per_class: int = 1, n_classes: int = 12):
    """Plot sample images from random classes."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "sample_images.png")

    dataset = datasets.ImageFolder(root=data_dir)
    classes = random.sample(dataset.classes, min(n_classes, len(dataset.classes)))

    cols = 4
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, cls_name in enumerate(classes):
        cls_idx = dataset.class_to_idx[cls_name]
        cls_images = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
        sample_idx = random.choice(cls_images)
        image, _ = dataset[sample_idx]

        ax = axes[idx]
        ax.imshow(image)
        display = config.get_display_name(cls_name)
        ax.set_title(display, fontsize=8, fontweight="bold")
        ax.axis("off")

    for idx in range(len(classes), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Sample images saved to {save_path}")


def plot_class_distribution(data_dir: str = None, save_path: str = None):
    """Plot the class distribution of the dataset."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")

    dataset = datasets.ImageFolder(root=data_dir)
    class_counts = {}
    for _, label in dataset.samples:
        name = dataset.classes[label]
        class_counts[name] = class_counts.get(name, 0) + 1

    names = [config.get_display_name(n) for n in class_counts.keys()]
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    y_pos = range(len(names))
    ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for i, count in enumerate(counts):
        ax.text(count + 10, i, str(count), va="center", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Class distribution saved to {save_path}")


def plot_augmentation_samples(data_dir: str = None, save_path: str = None):
    """Show original vs augmented versions of the same image."""
    from src.data_loader import get_train_transforms
    if data_dir is None:
        data_dir = config.DATA_DIR
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "augmentation_samples.png")

    dataset = datasets.ImageFolder(root=data_dir)
    sample_idx = random.randint(0, len(dataset) - 1)
    original_image, label = dataset[sample_idx]
    cls_name = dataset.classes[label]

    augmenter = get_train_transforms()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0][0].imshow(original_image)
    axes[0][0].set_title("Original", fontweight="bold")
    axes[0][0].axis("off")

    for i in range(1, 8):
        row, col = i // 4, i % 4
        augmented = augmenter(original_image)
        axes[row][col].imshow(denormalize(augmented))
        axes[row][col].set_title(f"Augmented #{i}", fontsize=10)
        axes[row][col].axis("off")

    plt.suptitle(f"Data Augmentation: {config.get_display_name(cls_name)}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🔄 Augmentation samples saved to {save_path}")
