"""
Grad-CAM Explainability Module
===============================
Generates Gradient-weighted Class Activation Maps to visualize
which regions of the leaf image the model focuses on for its prediction.
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import get_inference_transforms


class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.base_model.layer4[-1]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out.detach()
        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
            )
        ) / 255.0
        return cam_resized, target_class, confidence


def visualize_gradcam(image_path, model, device, class_names=None, save_path=None, target_class=None):
    """Generate and display Grad-CAM visualization for a single image."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    original_image = Image.open(image_path).convert("RGB")
    original_resized = original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    transform = get_inference_transforms()
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model)
    heatmap, pred_class, confidence = grad_cam.generate(input_tensor, target_class)
    pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
    display_name = config.get_display_name(pred_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_resized); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")

    original_np = np.array(original_resized) / 255.0
    overlay = np.clip(original_np * 0.6 + plt.cm.jet(heatmap)[:, :, :3] * 0.4, 0, 1)
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

    color = "darkgreen" if "healthy" in pred_name.lower() else "darkred"
    plt.suptitle(f"Prediction: {display_name} ({confidence:.1%})", fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return heatmap, pred_name, confidence


def generate_gradcam_grid(image_paths, model, device, class_names=None, save_path=None, cols=4):
    """Generate a grid of Grad-CAM visualizations for multiple images."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    n = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    transform = get_inference_transforms()
    grad_cam = GradCAM(model)

    for idx, img_path in enumerate(image_paths):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        try:
            orig = Image.open(img_path).convert("RGB")
            orig_r = orig.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
            inp = transform(orig).unsqueeze(0).to(device)
            hm, pc, conf = grad_cam.generate(inp)
            orig_np = np.array(orig_r) / 255.0
            overlay = np.clip(orig_np * 0.6 + plt.cm.jet(hm)[:, :, :3] * 0.4, 0, 1)
            ax.imshow(overlay)
            name = class_names[pc].split("___")[1].replace("_", " ")[:20]
            ax.set_title(f"{name}\n{conf:.1%}", fontsize=9,
                        color="green" if "healthy" in class_names[pc].lower() else "red")
        except Exception as e:
            ax.text(0.5, 0.5, str(e)[:30], ha="center", va="center", fontsize=8)
        ax.axis("off")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    plt.suptitle("Grad-CAM Explainability Grid", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
