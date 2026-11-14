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

# Helper: Kisi bhi output format me se Tensor nikalne ke liye
def find_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = find_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        if "logits" in obj and isinstance(obj["logits"], torch.Tensor):
            return obj["logits"]
        for v in obj.values():
            t = find_tensor(v)
            if t is not None:
                return t
    if hasattr(obj, "logits"):
        return find_tensor(obj.logits)
    return None

# ---------- 5. Prediction Function ----------
def predict(img: Image.Image):
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        logits = find_tensor(output)
        
        if logits is None:
            raise ValueError(f"Model output me Tensor nahi mila. Output type: {type(output)}")

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() > 2:
            logits = logits.squeeze()
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
    confidence = float(probs[pred_idx]) * 100

    class_probs = {
        (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"): f"{float(p)*100:.1f}%"
        for i, p in enumerate(probs)
    }

    return label, confidence, class_probs

# ---------- 6. Streamlit UI ----------
st.title("🌍 xView‑2 Disaster Classification (MHViT)")
st.write("Upload a satellite image to classify: Flood, Wildfire, Landslide, or Normal.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    try:
        with st.spinner("Analyzing..."):
            label, conf, probs = predict(img)

        st.success(f"**Prediction:** {label} ({conf:.1f}% confidence)")
        st.write("### Class Probabilities:")
        st.json(probs)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.exception(e)
