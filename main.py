#!/usr/bin/env python3
"""
Main Entry Point — Plant Leaf Disease Detection
=================================================
Runs the complete pipeline: download data → train → evaluate → visualize.

Usage:
    python main.py                  # Full pipeline
    python main.py --download-only  # Just download the dataset
    python main.py --evaluate-only  # Evaluate a trained model
    python main.py --eda-only       # Generate EDA visualizations only
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils import set_seed, get_device
from src.data_loader import download_dataset, create_data_loaders
from src.model import PlantDiseaseNet, load_model
from src.train import full_training_pipeline
from src.evaluate import evaluate_model, plot_training_history
from src.gradcam import visualize_gradcam, generate_gradcam_grid
from src.utils import plot_sample_images, plot_class_distribution, plot_augmentation_samples


def run_eda(data_dir: str):
    """Run Exploratory Data Analysis and generate visualizations."""
    print("\n" + "=" * 60)
    print("  📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60 + "\n")

    plot_class_distribution(data_dir)
    plot_sample_images(data_dir)
    plot_augmentation_samples(data_dir)

    print("\n✅ EDA complete! Check results/ for plots.")


def run_training(train_loader, val_loader, device):
    """Run the full training pipeline."""
    model, history = full_training_pipeline(train_loader, val_loader, device)
    plot_training_history(history)
    return model, history


def run_evaluation(model, test_loader, class_names, device):
    """Evaluate the model on the test set."""
    metrics = evaluate_model(model, test_loader, class_names, device)
    return metrics


def run_gradcam(model, data_dir, device, class_names):
    """Generate Grad-CAM visualizations on sample test images."""
    print("\n" + "=" * 60)
    print("  🔥 GRAD-CAM EXPLAINABILITY")
    print("=" * 60 + "\n")

    # Get sample images from different classes
    import random
    sample_paths = []
    for cls_dir in random.sample(os.listdir(data_dir), min(8, len(os.listdir(data_dir)))):
        cls_path = os.path.join(data_dir, cls_dir)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                sample_paths.append(os.path.join(cls_path, random.choice(images)))

    if sample_paths:
        save_path = os.path.join(config.RESULTS_DIR, "gradcam_grid.png")
        generate_gradcam_grid(sample_paths, model, device, class_names, save_path)

        # Individual visualizations for first 3
        for i, path in enumerate(sample_paths[:3]):
            ind_save = os.path.join(config.RESULTS_DIR, f"gradcam_sample_{i+1}.png")
            visualize_gradcam(path, model, device, class_names, ind_save)

    print("✅ Grad-CAM visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description="Plant Leaf Disease Detection System")
    parser.add_argument("--download-only", action="store_true", help="Only download the dataset")
    parser.add_argument("--eda-only", action="store_true", help="Only run EDA")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate a trained model")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation")
    args = parser.parse_args()

    print("🌿" * 30)
    print("\n  PLANT LEAF DISEASE DETECTION SYSTEM")
    print("  Using ResNet50 Transfer Learning\n")
    print("🌿" * 30)

    # Setup
    set_seed()
    device = get_device()

    # Step 1: Download dataset
    data_dir = download_dataset()

    if args.download_only:
        print("\n✅ Dataset download complete!")
        return

    # Step 2: EDA
    run_eda(data_dir)

    if args.eda_only:
        return

    # Step 3: Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(data_dir)

    if args.evaluate_only:
        # Load existing model
        model = load_model(config.BEST_MODEL_PATH, device)
        run_evaluation(model, test_loader, class_names, device)
        if not args.no_gradcam:
            run_gradcam(model, data_dir, device, class_names)
        return

    # Step 4: Train
    model, history = run_training(train_loader, val_loader, device)

    # Step 5: Evaluate
    model = load_model(config.BEST_MODEL_PATH, device)  # Load best model
    run_evaluation(model, test_loader, class_names, device)

    # Step 6: Grad-CAM
    if not args.no_gradcam:
        run_gradcam(model, data_dir, device, class_names)

    print("\n" + "🎉" * 20)
    print("\n  ALL DONE! Your results are in the results/ directory.")
    print(f"  Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"\n  To launch the web app:")
    print(f"    streamlit run app.py\n")
    print("🎉" * 20)


if __name__ == "__main__":
    main()
