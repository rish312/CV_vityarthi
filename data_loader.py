"""
Data Loader Module
==================
Handles dataset downloading, preprocessing, augmentation,
and creating PyTorch DataLoaders for training, validation, and testing.
"""

import os
import sys
import random
import shutil
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_train_transforms() -> transforms.Compose:
    """
    Returns data augmentation transforms for training.
    Includes random flips, rotation, color jitter, and normalization.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225]      # ImageNet std
        ),
    ])


def get_val_test_transforms() -> transforms.Compose:
    """
    Returns transforms for validation and testing.
    Only resize and normalize — no augmentation.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Returns transforms for single-image inference (web app).
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def download_dataset(data_dir: str = None) -> str:
    """
    Downloads the PlantVillage dataset from HuggingFace and organizes
    it into a directory-per-class structure.

    Args:
        data_dir: Target directory for the dataset.

    Returns:
        Path to the organized dataset directory.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    # Check if dataset already exists
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        num_classes = len([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        if num_classes >= 30:
            print(f"✅ Dataset already exists at {data_dir} ({num_classes} classes)")
            return data_dir

    print("📥 Downloading PlantVillage dataset from HuggingFace...")
    print("   This may take several minutes on first run.\n")

    try:
        from datasets import load_dataset

        dataset = load_dataset("mohanty/PlantVillage", split="train")
        os.makedirs(data_dir, exist_ok=True)

        # Get class names from features
        label_names = dataset.features["label"].names

        # Create directories for each class
        for name in label_names:
            os.makedirs(os.path.join(data_dir, name), exist_ok=True)

        # Save images
        print(f"📂 Organizing {len(dataset)} images into {len(label_names)} classes...")
        for idx, sample in enumerate(dataset):
            image = sample["image"]
            label = label_names[sample["label"]]
            save_path = os.path.join(data_dir, label, f"{idx}.jpg")

            if not os.path.exists(save_path):
                image.save(save_path)

            if (idx + 1) % 5000 == 0:
                print(f"   Processed {idx + 1}/{len(dataset)} images...")

        print(f"\n✅ Dataset ready: {len(dataset)} images in {len(label_names)} classes")
        return data_dir

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📋 Manual download instructions:")
        print("   1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   2. Download and extract to:", data_dir)
        print("   3. Ensure folder structure: data/PlantVillage/<class_name>/<images>")
        raise


def create_data_loaders(
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Creates train, validation, and test DataLoaders with stratified splitting.

    Args:
        data_dir: Path to the dataset directory.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    print(f"📊 Creating DataLoaders from {data_dir}...")

    # Load full dataset (with basic transforms first for splitting)
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    targets = full_dataset.targets

    print(f"   Found {len(full_dataset)} images in {len(class_names)} classes")

    # Stratified train/val/test split
    indices = list(range(len(full_dataset)))

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=targets,
        random_state=config.RANDOM_SEED,
    )

    # Second split: val vs test
    temp_targets = [targets[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=temp_targets,
        random_state=config.RANDOM_SEED,
    )

    print(f"   Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    # Create datasets with appropriate transforms
    train_dataset = TransformSubset(full_dataset, train_indices, get_train_transforms())
    val_dataset = TransformSubset(full_dataset, val_indices, get_val_test_transforms())
    test_dataset = TransformSubset(full_dataset, test_indices, get_val_test_transforms())

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("✅ DataLoaders created successfully\n")
    return train_loader, val_loader, test_loader, class_names


class TransformSubset(torch.utils.data.Dataset):
    """
    A Subset wrapper that applies custom transforms.
    This allows different transforms for train vs. val/test splits
    from the same base ImageFolder dataset.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]

        # The dataset returns PIL images (since no transform was set on ImageFolder)
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.indices)


def get_class_distribution(data_loader: DataLoader) -> Dict[int, int]:
    """
    Computes the class distribution for a DataLoader.

    Args:
        data_loader: A PyTorch DataLoader.

    Returns:
        Dictionary mapping class index to count.
    """
    distribution = {}
    for _, labels in data_loader:
        for label in labels.numpy():
            distribution[label] = distribution.get(label, 0) + 1
    return distribution


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("  PlantVillage Dataset Loader Test")
    print("=" * 60)

    data_dir = download_dataset()
    train_loader, val_loader, test_loader, class_names = create_data_loaders(data_dir)

    # Print a batch shape
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {[class_names[l] for l in labels[:5].tolist()]}")
