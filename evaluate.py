"""
Evaluation Module
==================
Handles model evaluation on the test set, generates metrics,
confusion matrix, classification report, and training curve plots.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions from the model on a data loader.

    Returns:
        Tuple of (all_preds, all_labels, all_probs)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(data_loader, desc="  Evaluating", ncols=80):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list,
    device: torch.device,
    save_dir: str = None,
) -> Dict:
    """
    Comprehensive model evaluation with metrics and visualizations.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        class_names: List of class names.
        device: Device for inference.
        save_dir: Directory to save results.

    Returns:
        Dictionary containing all metrics.
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  📊 MODEL EVALUATION")
    print("=" * 60)

    # Get predictions
    preds, labels, probs = get_predictions(model, test_loader, device)

    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    print(f"\n  Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro Precision:   {precision:.4f}")
    print(f"  Macro Recall:      {recall:.4f}")
    print(f"  Macro F1-Score:    {f1:.4f}")

    # Per-class metrics
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
    )

    # Save classification report
    report_text = classification_report(
        labels, preds,
        target_names=class_names,
    )
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Plant Disease Detection - Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text)
    print(f"\n  📄 Classification report saved to {report_path}")

    # Generate visualizations
    plot_confusion_matrix(labels, preds, class_names, save_dir)
    plot_top_bottom_classes(report, class_names, save_dir)

    # Save metrics as JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "num_test_samples": len(labels),
        "num_classes": len(class_names),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1-score": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
            if name in report
        },
    }

    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📊 Metrics saved to {metrics_path}")

    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    save_dir: str,
):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create a simplified version with shorter names
    short_names = []
    for name in class_names:
        parts = name.split("___")
        crop = parts[0].replace("_", " ").split("(")[0].strip()
        condition = parts[1].replace("_", " ")[:15] if len(parts) > 1 else "?"
        short_names.append(f"{crop[:6]}-{condition}")

    # Full confusion matrix
    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️  Confusion matrix saved to {path}")

    # Also save a smaller version showing only errors
    plot_error_matrix(cm, class_names, short_names, save_dir)


def plot_error_matrix(
    cm: np.ndarray,
    class_names: list,
    short_names: list,
    save_dir: str,
):
    """Plot the most confused class pairs."""
    # Find top misclassifications
    errors = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                errors.append((class_names[i], class_names[j], cm[i][j]))

    errors.sort(key=lambda x: x[2], reverse=True)
    top_errors = errors[:15]

    if not top_errors:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    labels_err = [f"{e[0].split('___')[1][:20]}\n→ {e[1].split('___')[1][:20]}" for e in top_errors]
    counts = [e[2] for e in top_errors]

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))
    ax.barh(range(len(counts)), counts, color=colors)
    ax.set_yticks(range(len(labels_err)))
    ax.set_yticklabels(labels_err, fontsize=8)
    ax.set_xlabel("Number of Misclassifications", fontsize=12)
    ax.set_title("Top 15 Misclassification Pairs", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(save_dir, "top_misclassifications.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️  Misclassification chart saved to {path}")


def plot_top_bottom_classes(report: dict, class_names: list, save_dir: str):
    """Plot the best and worst performing classes by F1-score."""
    class_f1 = []
    for name in class_names:
        if name in report:
            class_f1.append((name, report[name]["f1-score"]))

    class_f1.sort(key=lambda x: x[1])

    # Bottom 10 and Top 10
    bottom_10 = class_f1[:10]
    top_10 = class_f1[-10:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bottom 10
    names_b = [c[0].split("___")[1].replace("_", " ")[:25] for c in bottom_10]
    scores_b = [c[1] for c in bottom_10]
    colors_b = plt.cm.Reds(np.linspace(0.3, 0.8, len(scores_b)))
    ax1.barh(names_b, scores_b, color=colors_b)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("F1-Score")
    ax1.set_title("⚠️ 10 Worst Performing Classes", fontweight="bold")

    # Top 10
    names_t = [c[0].split("___")[1].replace("_", " ")[:25] for c in top_10]
    scores_t = [c[1] for c in top_10]
    colors_t = plt.cm.Greens(np.linspace(0.3, 0.8, len(scores_t)))
    ax2.barh(names_t, scores_t, color=colors_t)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("F1-Score")
    ax2.set_title("✅ 10 Best Performing Classes", fontweight="bold")

    plt.suptitle("Per-Class F1-Score Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "class_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️  Class performance chart saved to {path}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: str = None,
    stage1_epochs: int = config.STAGE1_EPOCHS,
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with keys: train_loss, train_acc, val_loss, val_acc
        save_dir: Directory to save plots.
        stage1_epochs: Number of epochs in stage 1 (for stage boundary marker).
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.axvline(x=stage1_epochs, color="gray", linestyle="--", alpha=0.5, label="Stage Boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax.axvline(x=stage1_epochs, color="gray", linestyle="--", alpha=0.5, label="Stage Boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    ax = axes[2]
    ax.plot(epochs, history["lr"], "g-", linewidth=2)
    ax.axvline(x=stage1_epochs, color="gray", linestyle="--", alpha=0.5, label="Stage Boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Training curves saved to {path}")


if __name__ == "__main__":
    # Load and plot saved training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history)
        print("Plots generated!")
    else:
        print("No training history found. Train the model first.")
