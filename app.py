import os
import urllib.request
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from model_arch import MHViTModel

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="DisasterVision AI | Satellite Disaster Detection",
    page_icon="🛰️",
    layout="wide"
)

# ---------- Blue + White Modern Styling ----------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #F0F4F8 0%, #FFFFFF 100%);
    }
    
    /* Top Header Container */
    .header-box {
        background: linear-gradient(135deg, #0B1E36 0%, #0052CC 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(0, 82, 204, 0.15);
    }
    .header-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .header-sub {
        font-size: 14px;
        color: #B3D4FF;
        margin-top: 4px;
    }

    /* Result Metric Card */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #D0DBE5;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(11, 30, 54, 0.05);
        margin-bottom: 16px;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
    }
    .badge-flood { background-color: #E0F2FE; color: #0284C7; }
    .badge-wildfire { background-color: #FEE2E2; color: #DC2626; }
    .badge-landslide { background-color: #FEF3C7; color: #D97706; }
    .badge-normal { background-color: #DCFCE7; color: #16A34A; }
</style>
""", unsafe_allow_html=True)

# ---------- 1. Device Setup ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Download & Load Weights ----------
MODEL_PATH = "best_mhvit.pth"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
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

# Class metadata
CLASS_METADATA = {
    "Flood": {"icon": "🌊", "color": "flood"},
    "Wildfire": {"icon": "🔥", "color": "wildfire"},
    "Landslide": {"icon": "⛰️", "color": "landslide"},
    "Normal": {"icon": "✅", "color": "normal"}
}
CLASS_NAMES = list(CLASS_METADATA.keys())

# Helper: Tensor Extractor
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

# ---------- 4. Prediction Function ----------
def predict(img: Image.Image):
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        logits = find_tensor(output)
        
        if logits is None:
            raise ValueError("Model output returned None.")

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() > 2:
            logits = logits.squeeze()
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

        # Softmax probabilities
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
    confidence = float(probs[pred_idx]) * 100

    return label, confidence, probs

# ---------- 5. Header & Banner ----------
if os.path.exists("Picture1.png"):
    st.image("Picture1.png", use_container_width=True)

st.markdown("""
<div class="header-box">
    <div class="header-title">🛰️ DisasterVision AI</div>
    <div class="header-sub">High-Resolution Satellite Imagery Disaster Assessment (MHViT Backbone)</div>
</div>
""", unsafe_allow_html=True)

# ---------- 6. Two-Column Dashboard ----------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Satellite Tile Upload")
    uploaded_file = st.file_uploader(
        "Upload a satellite image (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Target Observation Area", use_container_width=True)

with col2:
    st.markdown("### 📊 Detection Analysis")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing neural features..."):
            label, conf, probs = predict(img)

        info = CLASS_METADATA.get(label, {"icon": "📌", "color": "normal"})
        badge_style = f"badge-{info['color']}"

        # Summary Metric Box
        st.markdown(f"""
        <div class="result-card">
            <span style="font-size: 13px; color: #64748B; font-weight: 600;">DETECTED EVENT</span><br>
            <span class="badge {badge_style}" style="margin-top: 8px;">{info['icon']} {label}</span>
            <h2 style="margin: 10px 0 0 0; color: #0F172A; font-size: 26px;">{conf:.1f}% Confidence</h2>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Distribution with Clean Progress Bars
        st.markdown("#### Probability Breakdown")
        for i, name in enumerate(CLASS_NAMES):
            p = float(probs[i])
            m = CLASS_METADATA[name]
            
            lbl_col, val_col = st.columns([3, 1])
            lbl_col.write(f"{m['icon']} **{name}**")
            val_col.write(f"**{p * 100:.1f}%**")
            st.progress(p)

    else:
        st.info("👈 Please upload a satellite image on the left to view the AI analysis.")
