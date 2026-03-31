# 🌿 Plant Leaf Disease Detection using Deep Learning

> AI-powered plant disease diagnosis system that identifies **38 diseases across 14 crop species** from leaf photographs using Computer Vision and Transfer Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Web Application](#-web-application)
- [Technologies Used](#-technologies-used)
- [References](#-references)

---

## 🔍 Problem Statement

**India loses approximately 15–25% of its crop yield annually** due to plant diseases that go undetected until visible damage appears. Smallholder farmers, who constitute over 80% of Indian farming households, often lack timely access to agricultural experts for disease diagnosis.

This project addresses this challenge by building an **intelligent plant disease detection system** that can:
- Accept a simple photograph of a plant leaf
- Identify the disease (or confirm the plant is healthy)
- Provide treatment recommendations
- Explain its reasoning through visual attention maps (Grad-CAM)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning Classification** | ResNet50 with transfer learning for 38-class disease detection |
| 🔬 **Two-Stage Training** | Stage 1: Head training → Stage 2: Fine-tuning for optimal accuracy |
| 🎨 **Data Augmentation** | Random flips, rotation, color jitter, and affine transforms |
| 📊 **Comprehensive Evaluation** | Accuracy, Precision, Recall, F1, confusion matrix |
| 🔥 **Grad-CAM Explainability** | Visual attention maps showing what the model focuses on |
| 🌐 **Streamlit Web App** | User-friendly interface for real-time disease diagnosis |
| 💊 **Treatment Suggestions** | Disease-specific treatment and prevention recommendations |

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────────┐
│   ResNet50 Base        │ ← Pretrained on ImageNet (1.2M images)
│   (Feature Extractor)  │   Frozen in Stage 1, partially unfrozen in Stage 2
│   Conv → BN → ReLU    │
│   ×50 layers           │
└──────────┬────────────┘
           │ 2048-dim features
           ▼
┌───────────────────────┐
│   Custom Classifier    │
│   Dropout(0.5)         │
│   FC(2048 → 256)       │
│   ReLU + BatchNorm     │
│   Dropout(0.3)         │
│   FC(256 → 38)         │ ← 38 disease/healthy classes
└──────────┬────────────┘
           │
           ▼
     Softmax Output
   (38 class probabilities)
```

### Training Strategy

| Stage | What's Trained | Learning Rate | Epochs | Purpose |
|-------|---------------|--------------|--------|---------|
| **Stage 1** | Classification head only | 1e-3 | 10 | Learn disease-specific features |
| **Stage 2** | Head + top 20 base layers | 1e-4 | 15 | Fine-tune for domain adaptation |

---

## 📦 Dataset

**PlantVillage Dataset** — the largest openly available plant disease image dataset.

| Property | Details |
|----------|---------|
| Total Images | ~54,300 |
| Classes | 38 (diseases + healthy states) |
| Crop Species | 14 |
| Source | [HuggingFace](https://huggingface.co/datasets/mohanty/PlantVillage) |

### Supported Crops
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for faster training

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/CV_23BAI10139.git
cd CV_23BAI10139

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (automatic)
python main.py --download-only
```

---

## 💻 Usage

### Full Pipeline (Download → Train → Evaluate → Visualize)

```bash
python main.py
```

### Individual Steps

```bash
# Download dataset only
python main.py --download-only

# Exploratory Data Analysis only
python main.py --eda-only

# Evaluate a trained model
python main.py --evaluate-only

# Train without Grad-CAM generation
python main.py --no-gradcam
```

### Launch Web Application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Using on Google Colab

If you don't have a GPU locally, you can train on Google Colab:

```python
# In a Colab notebook:
!git clone https://github.com/yourusername/CV_23BAI10139.git
%cd CV_23BAI10139
!pip install -r requirements.txt
!python main.py
```

---

## 📁 Project Structure

```
CV_23BAI10139/
├── main.py                 # 🚀 Main entry point — runs full pipeline
├── app.py                  # 🌐 Streamlit web application
├── config.py               # ⚙️  All hyperparameters & configurations
├── requirements.txt        # 📦 Python dependencies
├── .gitignore              # 🚫 Git ignore rules
│
├── src/                    # 📂 Source code modules
│   ├── __init__.py
│   ├── data_loader.py      # 📥 Dataset download, augmentation, DataLoaders
│   ├── model.py            # 🧠 ResNet50 transfer learning architecture
│   ├── train.py            # 🏋️ Two-stage training loop
│   ├── evaluate.py         # 📊 Metrics, confusion matrix, plots
│   ├── gradcam.py          # 🔥 Grad-CAM explainability
│   └── utils.py            # 🛠️  Helper functions
│
├── data/                   # 📁 Dataset (auto-downloaded, gitignored)
│   └── PlantVillage/       #    54K+ leaf images in 38 class folders
│
├── models/                 # 💾 Saved model weights (gitignored)
│   ├── best_model.pth      #    Best validation accuracy checkpoint
│   └── final_model.pth     #    Final training checkpoint
│
├── results/                # 📈 Generated outputs
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── gradcam_grid.png
│   └── test_metrics.json
│
├── README.md               # 📖 This file
└── REPORT.md               # 📝 Detailed project report
```

---

## 📊 Results

After training, the following outputs are generated in the `results/` directory:

| Output | Description |
|--------|-------------|
| `training_curves.png` | Loss and accuracy curves (both stages) |
| `confusion_matrix.png` | 38×38 confusion matrix heatmap |
| `class_performance.png` | Best and worst performing classes |
| `top_misclassifications.png` | Most common misclassification pairs |
| `classification_report.txt` | Per-class precision, recall, F1-score |
| `test_metrics.json` | Machine-readable evaluation metrics |

### Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Overall Accuracy | >93% |
| Macro F1-Score | >0.90 |
| Top-1 Accuracy | >93% |

---

## 🔥 Grad-CAM Explainability

**Grad-CAM** (Gradient-weighted Class Activation Mapping) provides visual explanations by highlighting the leaf regions that most influenced the model's prediction.

This is critical for:
- **Trust**: Farmers and agronomists can verify the AI is looking at the right features
- **Debugging**: Identify when the model relies on background artifacts instead of disease symptoms
- **Education**: Understand which visual features distinguish diseases

The Grad-CAM module targets the last convolutional layer (`layer4`) of ResNet50 to produce high-resolution attention maps.

---

## 🌐 Web Application

The Streamlit web application provides:

1. **Image Upload** — Drag and drop or browse for a leaf photo
2. **Instant Diagnosis** — Disease name, confidence score, and top 5 predictions
3. **Treatment Info** — Cause, symptoms, and recommended treatment
4. **Grad-CAM Overlay** — Visual explanation of the AI's decision
5. **Crop Support Info** — List of all 14 supported crop species

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework |
| **torchvision** | Pretrained ResNet50, image transforms |
| **Streamlit** | Web application framework |
| **scikit-learn** | Evaluation metrics |
| **matplotlib / seaborn** | Visualization |
| **Pillow / OpenCV** | Image processing |
| **HuggingFace Datasets** | Dataset downloading |

---

## 📚 References

1. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
3. Mohanty, S.P., et al. (2016). *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science.
4. Hughes, D.P. & Salathe, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.

---

## 👤 Author

**Ekansh Sukla** — 23BAI10139
Computer Vision Course, VIT

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
