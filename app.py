import os
import urllib.request
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from model_arch import MHViTModel

# ---------- Page Config ----------
st.set_page_config(
    page_title="DisasterVision AI | Satellite Detection",
    page_icon="🛰️",
    layout="wide"
)

# ---------- Custom Blue + White Modern CSS ----------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #F0F4F8 0%, #FFFFFF 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #0A2540 0%, #0066CC 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 102, 204, 0.15);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #B0D4FF;
        margin-top: 6px;
    }

    /* Result Cards */
    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-flood { background-color: #E0F2FE; color: #0284C7; }
    .badge-wildfire { background-color: #FEE2E2; color: #DC2626; }
    .badge-landslide { background-color: #FEF3C7; color: #D97706; }
    .badge-normal { background-color: #DCFCE7; color: #16A34A; }
</style>
""", unsafe_allow_html=True)

# ---------- 1. Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Download & Load Model ----------
MODEL_PATH = "best_mhvit.pth"
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Initializing satellite detection model..."):
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

CLASS_INFO = {
    "Flood": {"icon": "🌊", "color": "flood"},
    "Wildfire": {"icon": "🔥", "color": "wildfire"},
    "Landslide": {"icon": "⛰️", "color": "landslide"},
    "Normal": {"icon": "✅", "color": "normal"}
}
CLASS_NAMES = list(CLASS_INFO.keys())

# Helper: Extract Tensor
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
            raise ValueError("Model output did not return any Tensor.")

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() > 2:
            logits = logits.squeeze()
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

        # Confidence percentage (Softmax)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
    confidence = float(probs[pred_idx]) * 100

    return label, confidence, probs

# ---------- 5. Header Section ----------
if os.path.exists("banner.jpg"):
    st.image("banner.jpg", use_container_width=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛰️ DisasterVision AI</div>
    <div class="hero-subtitle">High-Resolution Satellite Damage & Disaster Assessment System (MHViT Backbone)</div>
</div>
""", unsafe_allow_html=True)

# ---------- 6. Two-Column Layout ----------
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### 📥 Upload Satellite Imagery")
    uploaded_file = st.file_uploader(
        "Upload high-resolution satellite tile (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Target Region Preview", use_container_width=True)

with col2:
    st.markdown("### 📊 Assessment Report")
    
    if uploaded_file is not None:
        with st.spinner("🛰️ Scanning satellite imagery through MHViT neural layers..."):
            label, conf, probs = predict(img)

        info = CLASS_INFO.get(label, {"icon": "📌", "color": "normal"})
        badge_class = f"badge-{info['color']}"

        # Result Summary Card
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 13px; color: #64748B; margin-bottom: 8px;">DETECTED STATUS</p>
            <span class="badge {badge_class}">{info['icon']} {label}</span>
            <h2 style="margin: 12px 0 0 0; color: #0F172A; font-size: 28px;">{conf:.1f}% Confidence</h2>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Probabilities with Progress Bars
        st.markdown("#### Category Distribution")
        for i, name in enumerate(CLASS_NAMES):
            p = float(probs[i])
            c_info = CLASS_INFO[name]
            
            c_label, c_val = st.columns([3, 1])
            c_label.write(f"{c_info['icon']} **{name}**")
            c_val.write(f"**{p*100:.1f}%**")
            st.progress(p)

    else:
        st.info("👈 Please upload a satellite image on the left to generate the real-time AI report.")
