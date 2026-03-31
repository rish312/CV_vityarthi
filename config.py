"""
Configuration file for Plant Leaf Disease Detection System.
Contains all hyperparameters, paths, and class mappings.
"""

import os

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "PlantVillage")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for d in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset Configuration ────────────────────────────────────────────────────
IMAGE_SIZE = 224          # ResNet50 input size
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ─── Model Configuration ─────────────────────────────────────────────────────
NUM_CLASSES = 38
MODEL_NAME = "resnet50"
PRETRAINED = True
DROPOUT_RATE = 0.5

# ─── Training Configuration ──────────────────────────────────────────────────
# Stage 1: Train only the classifier head (base frozen)
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

# Stage 2: Fine-tune top layers of base + head
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-4
STAGE2_UNFREEZE_LAYERS = 20   # Number of layers to unfreeze from the end

# General training
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 5

# ─── Saved Model Paths ───────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pth")

# ─── Class Names (PlantVillage 38 classes) ────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ─── Disease Information (for web app display) ───────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "crop": "Apple",
        "cause": "Fungus Venturia inaequalis",
        "symptoms": "Dark, olive-green to black lesions on leaves and fruit.",
        "treatment": "Apply fungicides (captan, myclobutanil). Remove fallen infected leaves. Plant resistant varieties.",
    },
    "Apple___Black_rot": {
        "disease": "Black Rot",
        "crop": "Apple",
        "cause": "Fungus Botryosphaeria obtusa",
        "symptoms": "Brown rotting areas on fruit, 'frog-eye' leaf spots.",
        "treatment": "Prune dead wood, remove mummified fruits. Apply captan or thiophanate-methyl fungicides.",
    },
    "Apple___Cedar_apple_rust": {
        "disease": "Cedar Apple Rust",
        "crop": "Apple",
        "cause": "Fungus Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright orange-yellow spots on leaves.",
        "treatment": "Remove nearby cedar trees. Apply myclobutanil fungicide in spring.",
    },
    "Apple___healthy": {
        "disease": "Healthy",
        "crop": "Apple",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Blueberry___healthy": {
        "disease": "Healthy",
        "crop": "Blueberry",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Powdery Mildew",
        "crop": "Cherry",
        "cause": "Fungus Podosphaera clandestina",
        "symptoms": "White powdery coating on leaves, stunted growth.",
        "treatment": "Apply sulfur or potassium bicarbonate. Improve air circulation by pruning.",
    },
    "Cherry_(including_sour)___healthy": {
        "disease": "Healthy",
        "crop": "Cherry",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Gray Leaf Spot",
        "crop": "Corn",
        "cause": "Fungus Cercospora zeae-maydis",
        "symptoms": "Rectangular gray to tan lesions on leaves.",
        "treatment": "Rotate crops, use resistant hybrids. Apply foliar fungicides if severe.",
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Common Rust",
        "crop": "Corn",
        "cause": "Fungus Puccinia sorghi",
        "symptoms": "Small, round to elongate brown pustules on both leaf surfaces.",
        "treatment": "Plant resistant varieties. Apply fungicides (azoxystrobin) if detected early.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Northern Leaf Blight",
        "crop": "Corn",
        "cause": "Fungus Exserohilum turcicum",
        "symptoms": "Long, elliptical gray-green lesions on leaves.",
        "treatment": "Use resistant hybrids. Apply foliar fungicides. Rotate crops.",
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy",
        "crop": "Corn",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Grape___Black_rot": {
        "disease": "Black Rot",
        "crop": "Grape",
        "cause": "Fungus Guignardia bidwellii",
        "symptoms": "Brown circular lesions on leaves, black shriveled fruit (mummies).",
        "treatment": "Remove mummified berries. Apply mancozeb or myclobutanil before bloom.",
    },
    "Grape___Esca_(Black_Measles)": {
        "disease": "Esca (Black Measles)",
        "crop": "Grape",
        "cause": "Complex of fungi (Phaeomoniella, Phaeoacremonium)",
        "symptoms": "Tiger-stripe pattern on leaves, dark spots on berries.",
        "treatment": "No cure; remove infected vines. Protect pruning wounds with paste.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Leaf Blight",
        "crop": "Grape",
        "cause": "Fungus Pseudocercospora vitis",
        "symptoms": "Dark brown spots with yellow halos on leaves.",
        "treatment": "Apply copper-based fungicides. Remove infected debris.",
    },
    "Grape___healthy": {
        "disease": "Healthy",
        "crop": "Grape",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Greening (Huanglongbing)",
        "crop": "Orange",
        "cause": "Bacterium Candidatus Liberibacter",
        "symptoms": "Yellow shoots, lopsided bitter fruit, mottled leaves.",
        "treatment": "No cure. Remove infected trees. Control psyllid vector with insecticides.",
    },
    "Peach___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "crop": "Peach",
        "cause": "Bacterium Xanthomonas arboricola",
        "symptoms": "Small dark spots on leaves, pitting on fruit.",
        "treatment": "Apply copper sprays. Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Peach___healthy": {
        "disease": "Healthy",
        "crop": "Peach",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "crop": "Bell Pepper",
        "cause": "Bacterium Xanthomonas campestris",
        "symptoms": "Water-soaked spots on leaves, raised bumps on fruit.",
        "treatment": "Use disease-free seeds. Apply copper-based bactericides. Rotate crops.",
    },
    "Pepper,_bell___healthy": {
        "disease": "Healthy",
        "crop": "Bell Pepper",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Potato___Early_blight": {
        "disease": "Early Blight",
        "crop": "Potato",
        "cause": "Fungus Alternaria solani",
        "symptoms": "Dark concentric rings (target spots) on older leaves.",
        "treatment": "Apply chlorothalonil or mancozeb fungicides. Rotate crops. Remove debris.",
    },
    "Potato___Late_blight": {
        "disease": "Late Blight",
        "crop": "Potato",
        "cause": "Oomycete Phytophthora infestans",
        "symptoms": "Water-soaked lesions turning brown/black, white mold on underside.",
        "treatment": "Apply metalaxyl-based fungicides immediately. Destroy infected plants. Use resistant varieties.",
    },
    "Potato___healthy": {
        "disease": "Healthy",
        "crop": "Potato",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Raspberry___healthy": {
        "disease": "Healthy",
        "crop": "Raspberry",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Soybean___healthy": {
        "disease": "Healthy",
        "crop": "Soybean",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Squash___Powdery_mildew": {
        "disease": "Powdery Mildew",
        "crop": "Squash",
        "cause": "Fungus Podosphaera xanthii",
        "symptoms": "White powdery spots on leaves and stems.",
        "treatment": "Apply neem oil or sulfur fungicide. Ensure good air circulation.",
    },
    "Strawberry___Leaf_scorch": {
        "disease": "Leaf Scorch",
        "crop": "Strawberry",
        "cause": "Fungus Diplocarpon earlianum",
        "symptoms": "Purple to brown spots merging into scorched appearance.",
        "treatment": "Remove infected leaves. Apply captan fungicide. Use drip irrigation.",
    },
    "Strawberry___healthy": {
        "disease": "Healthy",
        "crop": "Strawberry",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
    "Tomato___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "crop": "Tomato",
        "cause": "Bacterium Xanthomonas vesicatoria",
        "symptoms": "Small, dark, water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper sprays. Remove infected plants. Use certified disease-free seeds.",
    },
    "Tomato___Early_blight": {
        "disease": "Early Blight",
        "crop": "Tomato",
        "cause": "Fungus Alternaria solani",
        "symptoms": "Dark concentric rings (target-like) on lower leaves first.",
        "treatment": "Apply chlorothalonil fungicide. Mulch around plants. Remove lower infected leaves.",
    },
    "Tomato___Late_blight": {
        "disease": "Late Blight",
        "crop": "Tomato",
        "cause": "Oomycete Phytophthora infestans",
        "symptoms": "Large, irregularly shaped brown-green water-soaked lesions.",
        "treatment": "Remove and destroy infected plants immediately. Apply mancozeb or chlorothalonil.",
    },
    "Tomato___Leaf_Mold": {
        "disease": "Leaf Mold",
        "crop": "Tomato",
        "cause": "Fungus Passalora fulva",
        "symptoms": "Yellow spots on upper leaf surface, olive-green mold underneath.",
        "treatment": "Improve ventilation in greenhouses. Apply chlorothalonil. Use resistant varieties.",
    },
    "Tomato___Septoria_leaf_spot": {
        "disease": "Septoria Leaf Spot",
        "crop": "Tomato",
        "cause": "Fungus Septoria lycopersici",
        "symptoms": "Many small circular spots with dark borders and gray centers.",
        "treatment": "Remove infected leaves. Apply copper or chlorothalonil. Avoid overhead watering.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Spider Mites",
        "crop": "Tomato",
        "cause": "Tetranychus urticae (Two-spotted spider mite)",
        "symptoms": "Tiny yellow dots on leaves, fine webbing, leaf browning.",
        "treatment": "Spray with neem oil or insecticidal soap. Introduce predatory mites.",
    },
    "Tomato___Target_Spot": {
        "disease": "Target Spot",
        "crop": "Tomato",
        "cause": "Fungus Corynespora cassiicola",
        "symptoms": "Brown spots with concentric rings on leaves.",
        "treatment": "Apply chlorothalonil or azoxystrobin. Improve air circulation. Rotate crops.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV), spread by whiteflies",
        "symptoms": "Upward curling of leaves, yellowing, stunted growth.",
        "treatment": "Control whiteflies with insecticides. Use reflective mulch. Plant resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease": "Mosaic Virus",
        "crop": "Tomato",
        "cause": "Tomato Mosaic Virus (ToMV)",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves, distorted growth.",
        "treatment": "No cure. Remove infected plants. Disinfect tools. Use resistant varieties.",
    },
    "Tomato___healthy": {
        "disease": "Healthy",
        "crop": "Tomato",
        "cause": "N/A",
        "symptoms": "No disease symptoms detected.",
        "treatment": "Continue regular maintenance and monitoring.",
    },
}

# ─── Friendly Display Names ──────────────────────────────────────────────────
def get_display_name(class_name: str) -> str:
    """Convert class folder name to a human-readable display name."""
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return f"{crop} — {condition}"
