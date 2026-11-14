import os
import urllib.request
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from model_arch import MHViTModel

# ---------- 1. Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Download & Load Model ----------
MODEL_PATH = "best_mhvit.pth"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        # Yahan apna GitHub Release ka copy kiya hua link paste karein:
        MODEL_URL = "https://github.com/IMK-hub/MHViT-Detection-models/releases/download/detection-models/best_mhvit.pth"
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_model():
    model = MHViTModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

model = load_model()

# ---------- 3. Pre-processing ----------
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ---------- 4. Class Names ----------
CLASS_NAMES = ["Flood", "Wildfire", "Landslide", "Normal"]

# ---------- 5. Prediction Function ----------
def predict(img: Image.Image):
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return label, confidence, {
        name: f"{p*100:.1f}%" for name, p in zip(CLASS_NAMES, probs)
    }

# ---------- 6. Streamlit UI ----------
st.title("🌍 xView‑2 Disaster Classification (MHViT)")
st.write("Upload a satellite image (1024×1024) to classify: Flood, Wildfire, Landslide, or Normal.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, conf, probs = predict(img)

    st.success(f"**Prediction:** {label} ({conf:.1f}% confidence)")
    st.write("### Class Probabilities:")
    st.json(probs)
