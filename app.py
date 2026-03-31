"""
🌿 Plant Leaf Disease Detection — Streamlit Web Application
=============================================================
Upload a leaf image → Get instant disease diagnosis with
AI-powered explanations using Grad-CAM.

Run: streamlit run app.py
"""

import os
import sys
import io

import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.model import PlantDiseaseNet, load_model
from src.data_loader import get_inference_transforms
from src.gradcam import GradCAM


# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-healthy {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #28a745;
    }
    .result-diseased {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stApp { max-width: 1200px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    """Load the trained model (cached to avoid re-loading)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.FINAL_MODEL_PATH

    if not os.path.exists(model_path):
        return None, device

    model = load_model(model_path, device)
    return model, device


def predict(image: Image.Image, model, device):
    """Run inference on a single image."""
    transform = get_inference_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)

    top5_probs, top5_indices = torch.topk(probs, 5)
    results = []
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        class_name = config.CLASS_NAMES[idx.item()]
        results.append({
            "class_name": class_name,
            "display_name": config.get_display_name(class_name),
            "probability": prob.item(),
            "info": config.DISEASE_INFO.get(class_name, {}),
        })
    return results, input_tensor


def generate_gradcam_overlay(image, model, device):
    """Generate Grad-CAM overlay for the image."""
    transform = get_inference_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model)
    heatmap, pred_class, confidence = grad_cam.generate(input_tensor)

    original_resized = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    original_np = np.array(original_resized) / 255.0
    jet_heatmap = plt.cm.jet(heatmap)[:, :, :3]
    overlay = np.clip(original_np * 0.6 + jet_heatmap * 0.4, 0, 1)
    return heatmap, overlay, confidence


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("🌿 Plant Leaf Disease Detection")
    st.markdown("*AI-powered plant disease diagnosis using Deep Learning & Computer Vision*")
    st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **ResNet50** deep learning model
        trained on the **PlantVillage** dataset to identify
        **38 different plant diseases** across **14 crop species**.

        **How to use:**
        1. Upload a clear photo of a plant leaf
        2. The AI will analyze the image
        3. Get disease diagnosis + treatment suggestions
        4. View Grad-CAM to see what the AI focuses on
        """)

        st.divider()
        st.header("📊 Model Info")
        st.metric("Classes", "38")
        st.metric("Crops", "14")
        st.metric("Architecture", "ResNet50")

        st.divider()
        st.header("🌱 Supported Crops")
        crops = sorted(set(n.split("___")[0].replace("_", " ") for n in config.CLASS_NAMES))
        for crop in crops:
            st.markdown(f"  • {crop}")

    # Load model
    model, device = load_trained_model()

    if model is None:
        st.warning("⚠️ No trained model found. Please train the model first.")
        st.code("python -m src.train", language="bash")
        st.info("Once training is complete, refresh this page.")
        return

    # File upload
    st.header("📸 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a plant leaf...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear, well-lit photo of a single plant leaf.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

        # Run prediction
        with st.spinner("🔍 Analyzing leaf..."):
            results, input_tensor = predict(image, model, device)

        top_result = results[0]
        is_healthy = "healthy" in top_result["class_name"].lower()

        with col2:
            st.subheader("🔬 Diagnosis Result")

            if is_healthy:
                st.success(f"✅ **{top_result['display_name']}**")
                st.markdown(f"**Confidence:** {top_result['probability']:.1%}")
            else:
                st.error(f"⚠️ **{top_result['display_name']}**")
                st.markdown(f"**Confidence:** {top_result['probability']:.1%}")

            # Disease details
            info = top_result.get("info", {})
            if info:
                st.markdown("---")
                st.markdown(f"**🦠 Cause:** {info.get('cause', 'N/A')}")
                st.markdown(f"**📋 Symptoms:** {info.get('symptoms', 'N/A')}")
                st.markdown(f"**💊 Treatment:** {info.get('treatment', 'N/A')}")

        # Top 5 predictions
        st.subheader("📊 Top 5 Predictions")
        for i, result in enumerate(results):
            bar_color = "green" if "healthy" in result["class_name"].lower() else "red"
            st.markdown(f"**{i+1}. {result['display_name']}** — {result['probability']:.1%}")
            st.progress(result["probability"])

        # Grad-CAM visualization
        st.subheader("🔥 Grad-CAM Explainability")
        st.markdown("*Where does the AI look? Warm colors = high attention regions.*")

        with st.spinner("Generating Grad-CAM..."):
            heatmap, overlay, confidence = generate_gradcam_overlay(image, model, device)

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.image(image.resize((224, 224)), caption="Original", use_container_width=True)
        with gc2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Grad-CAM Heatmap")
        with gc3:
            st.image((overlay * 255).astype(np.uint8), caption="Overlay", use_container_width=True)
    else:
        st.info("👆 Upload a leaf image above to get started!")

        st.markdown("---")
        st.subheader("🌾 Example Disease Classes")
        example_classes = [
            ("🍎 Apple Scab", "Dark olive-green lesions on apple leaves"),
            ("🥔 Potato Late Blight", "Water-soaked brown/black lesions"),
            ("🍅 Tomato Yellow Leaf Curl", "Upward curling and yellowing"),
            ("🌽 Corn Common Rust", "Brown pustules on both leaf surfaces"),
            ("🍇 Grape Black Rot", "Brown circular lesions, mummified fruit"),
            ("🫑 Pepper Bacterial Spot", "Water-soaked spots on leaves"),
        ]
        cols = st.columns(3)
        for i, (name, desc) in enumerate(example_classes):
            with cols[i % 3]:
                st.markdown(f"**{name}**")
                st.caption(desc)


if __name__ == "__main__":
    main()
