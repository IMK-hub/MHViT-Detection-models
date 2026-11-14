
import os
import urllib.request
import html

import torch
import numpy as np
import streamlit as st

from PIL import Image
from torchvision import transforms
from model_arch import MHViTModel


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MHViT Detection Model | DisasterVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* Hide Streamlit Header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Hide Streamlit Toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide Deploy Button */
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Hide Main Menu */
#MainMenu {
    visibility: hidden;
}

/* Hide Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #071936;
    --blue: #1559a9;
    --cyan: #22d3ee;
    --light: #edf6ff;
    --text: #123568;
}

.stApp {
    background: linear-gradient(135deg, #edf5ff, #ffffff, #eaf3ff);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 0.7rem;
    padding-bottom: 0.3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Header */

.hero-container {
    background: linear-gradient(110deg, #071936, #103b7a, #1559a9);
    padding: 15px 24px;
    border-radius: 13px;
    border: 1px solid #285c9d;
    box-shadow: 0 5px 18px rgba(15,65,130,0.15);
    margin-bottom: 13px;
}

.hero-title {
    font-size: 25px;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: 0.3px;
}

.hero-subtitle {
    color: #c3dcff;
    font-size: 11px;
    margin-top: 5px;
}

.header-status {
    background: rgba(34,211,238,0.12);
    border: 1px solid #22d3ee;
    color: #67e8f9;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
}

/* Section headings */

.section-heading {
    color: #103568;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #8496ad;
    font-size: 11px;
    margin-bottom: 12px;
}

/* Left panel */

.left-panel {
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 13px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(15,55,110,0.06);
    min-height: 470px;
}

/* Right panel */

.assessment-card {
    background: linear-gradient(145deg, #071c42, #10346e);
    border: 1px solid #2861a5;
    border-radius: 13px;
    padding: 17px;
    color: white;
    box-shadow: 0 5px 18px rgba(10,45,100,0.15);
}

.assessment-title {
    font-size: 17px;
    font-weight: 700;
    color: white;
    margin-bottom: 13px;
}

.threat-box {
    background: linear-gradient(120deg, #123b76, #174b8c);
    border: 1px solid #3976ba;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}

.threat-label {
    font-size: 22px;
    font-weight: 800;
    margin-top: 6px;
}

.confidence {
    font-size: 29px;
    font-weight: 800;
    color: white;
}

.confidence-label {
    color: #c3dcff;
    font-size: 11px;
}

/* Probability bars */

.prob-heading {
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin-bottom: 11px;
}

.prob-row {
    margin-bottom: 13px;
}

.prob-title {
    display: flex;
    justify-content: space-between;
    color: white;
    font-size: 11px;
    margin-bottom: 5px;
}

.prob-track {
    width: 100%;
    height: 7px;
    background: #294c7b;
    border-radius: 20px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 20px;
}

/* System information */

.info-strip {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 11px;
    padding: 11px 16px;
    margin-top: 12px;
    box-shadow: 0 3px 12px rgba(15,55,110,0.05);
}

.info-item {
    color: #7188a5;
    font-size: 10px;
}

.info-item b {
    color: #123c78;
    font-size: 12px;
    display: block;
    margin-top: 3px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background: #f5faff;
    border: 1px dashed #6da4dc;
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 8px;
}

[data-testid="stImage"] {
    margin-top: 5px;
}

[data-testid="stImage"] img {
    max-height: 390px;
    width: 100%;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid #d3e4f6;
    background: #f0f6fc;
}

/* Hide unnecessary spacing */

div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* Responsive */

@media (max-width: 768px) {

    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .hero-title {
        font-size: 20px;
    }

    .left-panel {
        min-height: auto;
    }

    [data-testid="stImage"] img {
        max-height: 320px;
    }

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DEVICE CONFIGURATION
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# 4. LOAD MHViT MODEL
# =====================================================

MODEL_PATH = "best_mhvit.pth"

if not os.path.exists(MODEL_PATH):

    MODEL_URL = (
        "https://github.com/IMK-hub/MHViT-Detection-models/"
        "releases/download/detection-models/best_mhvit.pth"
    )

    with st.spinner("Downloading MHViT Detection Model..."):

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )


@st.cache_resource
def load_model():

    model = MHViTModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()
    model.to(DEVICE)

    return model


model = load_model()


# =====================================================
# 5. IMAGE PREPROCESSING
# =====================================================

val_tf = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =====================================================
# 6. CLASS INFORMATION
# =====================================================

CLASS_INFO = {

    "Flood": {
        "icon": "🌊",
        "color": "#22d3ee"
    },

    "Wildfire": {
        "icon": "🔥",
        "color": "#60a5fa"
    },

    "Landslide": {
        "icon": "⛰️",
        "color": "#818cf8"
    },

    "Normal": {
        "icon": "✓",
        "color": "#34d399"
    }

}

CLASS_NAMES = list(CLASS_INFO.keys())


# =====================================================
# 7. FIND TENSOR
# =====================================================

def find_tensor(obj):

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (tuple, list)):

        for item in obj:

            t = find_tensor(item)

            if t is not None:
                return t

    if isinstance(obj, dict):

        if "logits" in obj and isinstance(
            obj["logits"], torch.Tensor
        ):
            return obj["logits"]

        for value in obj.values():

            t = find_tensor(value)

            if t is not None:
                return t

    if hasattr(obj, "logits"):

        return find_tensor(obj.logits)

    return None


# =====================================================
# 8. PREDICTION FUNCTION
# =====================================================

def predict(img: Image.Image):

    tensor = val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        logits = find_tensor(output)

        if logits is None:

            raise ValueError(
                "Model output did not return any Tensor."
            )

        if logits.dim() == 1:

            logits = logits.unsqueeze(0)

        elif logits.dim() > 2:

            logits = logits.squeeze()

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

        probs = torch.softmax(
            logits.float(),
            dim=-1
        ).cpu().numpy()[0]

    if len(probs) != len(CLASS_NAMES):

        raise ValueError(
            "Model output classes do not match CLASS_NAMES."
        )

    pred_idx = int(np.argmax(probs))

    label = CLASS_NAMES[pred_idx]

    confidence = float(probs[pred_idx]) * 100

    return label, confidence, probs


# =====================================================
# 9. HEADER
# =====================================================

st.html("""
<div class="hero-container">

    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        flex-wrap:wrap;
        gap:10px;">

        <div>

            <div class="hero-title">
                🛰️ MHViT Detection Model
            </div>

            <div class="hero-subtitle">
                DisasterVision AI | Satellite Image-Based
                Natural Disaster Classification System
            </div>

        </div>

        <div class="header-status">
            ● MODEL ONLINE
        </div>

    </div>

</div>
""")


# =====================================================
# 10. MAIN DASHBOARD
# =====================================================

col1, col2 = st.columns(
    [1.15, 0.85],
    gap="medium"
)


# =====================================================
# 11. LEFT PANEL
# =====================================================

with col1:

    st.html("""
    <div style="margin-bottom:9px;">

        <div class="section-heading">
            🛰️ Satellite Telemetry Viewport
        </div>

        <div class="section-subtitle">
            Satellite image input and visualization
        </div>

    </div>
    """)

    uploaded_file = st.file_uploader(
        "Upload Satellite Image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible"
    )

    if uploaded_file is not None:

        img = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            img,
            caption="Satellite Image - Target Region",
            use_container_width=True
        )

    else:

        st.html("""
        <div style="
            background:white;
            border:1px dashed #78a9dc;
            border-radius:12px;
            padding:70px 15px;
            text-align:center;
            margin-top:5px;">

            <div style="font-size:48px;">🛰️</div>

            <h3 style="color:#174477;font-size:18px;">
                Satellite Image Viewport
            </h3>

            <p style="color:#8496ad;font-size:12px;">
                Upload a satellite image to start
                disaster classification.
            </p>

            <p style="color:#9aabc0;font-size:10px;">
                Supported: PNG | JPG | JPEG
            </p>

        </div>
        """)


# =====================================================
# 12. RIGHT PANEL
# =====================================================

with col2:

    st.html("""
    <div class="section-heading" style="margin-bottom:12px;">
        ⚡ AI Threat Assessment
    </div>
    """)

    if uploaded_file is not None:

        try:

            with st.spinner("MHViT analyzing satellite image..."):

                label, conf, probs = predict(img)

            info = CLASS_INFO[label]

            # Create all probability bars in one compact card

            probability_html = ""

            for i, name in enumerate(CLASS_NAMES):

                p = float(probs[i]) * 100

                color = CLASS_INFO[name]["color"]

                icon = CLASS_INFO[name]["icon"]

                probability_html += f"""
                <div class="prob-row">

                    <div class="prob-title">
                        <span>{icon} {html.escape(name)}</span>
                        <b>{p:.1f}%</b>
                    </div>

                    <div class="prob-track">

                        <div class="prob-fill"
                             style="width:{p:.2f}%;background:{color};">
                        </div>

                    </div>

                </div>
                """

            st.html(f"""
            <div class="assessment-card">

                <div class="assessment-title">
                    AI Classification Result
                </div>

                <div class="threat-box">

                    <div style="
                        color:#b9d9ff;
                        font-size:11px;
                        font-weight:600;">

                        DETECTED DISASTER CATEGORY

                    </div>

                    <div class="threat-label"
                         style="color:{info['color']};">

                        {info['icon']} {html.escape(label.upper())}

                    </div>

                    <div style="margin-top:10px;">

                        <span class="confidence">
                            {conf:.1f}%
                        </span>

                        <span class="confidence-label">
                            Confidence Score
                        </span>

                    </div>

                </div>

                <div class="prob-heading">
                    Category Distribution
                </div>

                {probability_html}

            </div>
            """)

        except Exception as e:

            st.error(f"Prediction Error: {str(e)}")

    else:

        st.html("""
        <div class="assessment-card">

            <div class="assessment-title">
                AI Classification Result
            </div>

            <div class="threat-box">

                <div style="
                    color:#b9d9ff;
                    font-size:11px;">

                    DETECTION STATUS

                </div>

                <div style="
                    font-size:21px;
                    font-weight:800;
                    color:#7dd3fc;
                    margin-top:10px;">

                    AWAITING IMAGE

                </div>

                <div style="
                    color:#b9d9ff;
                    font-size:11px;
                    margin-top:8px;">

                    Upload satellite imagery
                    to initiate MHViT analysis.

                </div>

            </div>

            <div class="prob-heading">
                Supported Disaster Categories
            </div>

            <div style="color:#7dd3fc;font-size:12px;margin:12px 0;">
                🌊 Flood Detection
            </div>

            <div style="color:#93c5fd;font-size:12px;margin:12px 0;">
                🔥 Wildfire Detection
            </div>

            <div style="color:#a5b4fc;font-size:12px;margin:12px 0;">
                ⛰️ Landslide Detection
            </div>

            <div style="color:#6ee7b7;font-size:12px;margin:12px 0;">
                ✓ Normal Region
            </div>

        </div>
        """)


# =====================================================
# 13. BOTTOM SYSTEM INFORMATION
# =====================================================

st.html(f"""
<div class="info-strip">

    <div class="info-item">
        MODEL ARCHITECTURE
        <b>MHViT</b>
    </div>

    <div class="info-item">
        INPUT RESOLUTION
        <b>224 × 224</b>
    </div>

    <div class="info-item">
        IMAGE SENSOR
        <b>RGB</b>
    </div>

    <div class="info-item">
        CLASSIFICATION
        <b>4 Classes</b>
    </div>

    <div class="info-item">
        COMPUTE DEVICE
        <b>{str(DEVICE).upper()}</b>
    </div>

</div>
""")
