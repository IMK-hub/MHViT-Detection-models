
import os
import urllib.request
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from model_arch import MHViTModel

# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="DisasterVision AI | Satellite Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #edf5ff, #ffffff, #eaf3ff);
    font-family: 'Inter', sans-serif;
    color: #102c59;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero-container {
    background: linear-gradient(110deg, #071936, #103b7a, #1559a9);
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(15,65,130,0.18);
    border: 1px solid #285c9d;
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero-subtitle {
    color: #b9d9ff;
    font-size: 13px;
    margin-top: 7px;
}

.header-status {
    background: rgba(34,211,238,0.12);
    border: 1px solid #22d3ee;
    color: #67e8f9;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
}

.dashboard-card {
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 5px 20px rgba(15,55,110,0.07);
}

.card-heading {
    color: #102e5d;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}

.card-subtitle {
    color: #8293aa;
    font-size: 12px;
}

.assessment-card {
    background: linear-gradient(145deg, #071c42, #10346e);
    border: 1px solid #2861a5;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 18px;
    color: white;
    box-shadow: 0 6px 22px rgba(10,45,100,0.16);
}

.assessment-title {
    font-size: 19px;
    font-weight: 700;
    color: white;
    margin-bottom: 18px;
}

.threat-box {
    background: linear-gradient(120deg, #123b76, #174b8c);
    border: 1px solid #3976ba;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
}

.threat-label {
    font-size: 23px;
    font-weight: 800;
    color: #7dd3fc;
    margin-top: 8px;
}

.confidence {
    font-size: 31px;
    font-weight: 800;
    color: white;
}

.confidence-label {
    color: #b9d9ff;
    font-size: 12px;
}

.prob-row {
    margin: 17px 0;
}

.prob-title {
    display: flex;
    justify-content: space-between;
    color: white;
    font-size: 13px;
    margin-bottom: 8px;
}

.prob-track {
    width: 100%;
    height: 8px;
    background: #244b7e;
    border-radius: 20px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 20px;
}

.protocol-card {
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 5px 18px rgba(15,55,110,0.06);
}

.protocol-title {
    color: #103568;
    font-size: 17px;
    font-weight: 700;
}

.metric-box {
    background: #f0f7ff;
    border: 1px solid #d4e6fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.metric-label {
    color: #7188a5;
    font-size: 11px;
}

.metric-value {
    color: #123c78;
    font-size: 18px;
    font-weight: 800;
    margin-top: 6px;
}

[data-testid="stFileUploader"] {
    background: #f5faff;
    border: 1px dashed #6da4dc;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #c6dcf3;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22d3ee, #2563eb);
}

.stButton button {
    background: linear-gradient(90deg, #1559bd, #2479df);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #8da2bb;
    font-size: 11px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# 4. MODEL DOWNLOAD
# =====================================================

MODEL_PATH = "best_mhvit.pth"

if not os.path.exists(MODEL_PATH):

    MODEL_URL = (
        "https://github.com/IMK-hub/MHViT-Detection-models/"
        "releases/download/detection-models/best_mhvit.pth"
    )

    with st.spinner("Downloading MHViT model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


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
        "color": "#38bdf8"
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
        gap:15px;">

        <div>
            <div class="hero-title">
                🛰️ DisasterVision AI
            </div>

            <div class="hero-subtitle">
                Satellite Reconnaissance & Disaster
                Assessment System | MHViT Deep Learning
            </div>
        </div>

        <div class="header-status">
            ● SYSTEM ONLINE
        </div>

    </div>

</div>
""")


# =====================================================
# 10. MAIN DASHBOARD
# =====================================================

col1, col2 = st.columns(
    [1.15, 0.85],
    gap="large"
)


# =====================================================
# 11. LEFT - SATELLITE VIEWPORT
# =====================================================

with col1:

    st.html("""
    <div class="dashboard-card">

        <div class="card-heading">
            🛰️ Satellite Telemetry Viewport
        </div>

        <div class="card-subtitle">
            High-resolution satellite image monitoring
        </div>

    </div>
    """)

    st.markdown("### 📡 Upload Satellite Imagery")

    uploaded_file = st.file_uploader(
        "Upload satellite image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        st.image(
            img,
            caption="Target Region - Satellite Image",
            use_container_width=True
        )

        st.html("""
        <div class="dashboard-card">
            <div style="color:#174477;font-weight:700;">
                ✓ IMAGE UPLOADED SUCCESSFULLY
            </div>
            <div class="card-subtitle" style="margin-top:8px;">
                Sensor: RGB | Input: MHViT | Resolution: 224 × 224
            </div>
        </div>
        """)

    else:

        st.html("""
        <div style="
            background:white;
            border:1px dashed #78a9dc;
            border-radius:14px;
            padding:65px 20px;
            text-align:center;
            margin-top:12px;">

            <div style="font-size:55px;">🛰️</div>

            <h3 style="color:#174477;">
                Satellite Viewport
            </h3>

            <p style="color:#7d91aa;font-size:13px;">
                Upload satellite imagery to start
                AI-powered disaster detection.
            </p>

            <p style="color:#9aabc0;font-size:11px;">
                Supported formats: PNG | JPG | JPEG
            </p>

        </div>
        """)


# =====================================================
# 12. RIGHT - AI THREAT ASSESSMENT
# =====================================================

with col2:

    if uploaded_file is not None:

        try:

            with st.spinner(
                "Analyzing satellite imagery through MHViT..."
            ):

                label, conf, probs = predict(img)

            info = CLASS_INFO[label]

            st.html(f"""
            <div class="assessment-card">

                <div class="assessment-title">
                    ⚡ AI Threat Assessment
                </div>

                <div class="threat-box">

                    <div style="
                        color:#b9d9ff;
                        font-size:12px;">
                        DETECTED STATUS
                    </div>

                    <div class="threat-label">
                        {info['icon']} {label.upper()}
                    </div>

                    <div style="margin-top:15px;">

                        <span class="confidence">
                            {conf:.1f}%
                        </span>

                        <span class="confidence-label">
                            Confidence Score
                        </span>

                    </div>

                </div>

                <div style="
                    font-size:15px;
                    font-weight:700;
                    color:white;
                    margin-bottom:15px;">

                    Category Distribution

                </div>

            </div>
            """)

            # Probability distribution
            for i, name in enumerate(CLASS_NAMES):

                p = float(probs[i]) * 100

                color = CLASS_INFO[name]["color"]
                icon = CLASS_INFO[name]["icon"]

                st.html(f"""
                <div class="assessment-card"
                     style="padding:12px 18px;margin-bottom:8px;">

                    <div class="prob-row">

                        <div class="prob-title">
                            <span>{icon} {name}</span>
                            <b>{p:.1f}%</b>
                        </div>

                        <div class="prob-track">
                            <div class="prob-fill"
                                 style="width:{p}%;background:{color};">
                            </div>
                        </div>

                    </div>

                </div>
                """)

        except Exception as e:

            st.error(f"Prediction Error: {str(e)}")

    else:

        st.html("""
        <div class="assessment-card">

            <div class="assessment-title">
                ⚡ AI Threat Assessment
            </div>

            <div class="threat-box">

                <div style="color:#b9d9ff;font-size:12px;">
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
                    font-size:12px;
                    color:#b9d9ff;
                    margin-top:8px;">
                    Upload satellite imagery to initiate AI analysis.
                </div>

            </div>

            <div style="
                color:white;
                font-size:15px;
                font-weight:700;
                margin-bottom:15px;">
                Supported Detection Categories
            </div>

            <div style="color:#7dd3fc;margin:12px 0;">
                🌊 Flood Detection
            </div>

            <div style="color:#93c5fd;margin:12px 0;">
                🔥 Wildfire Detection
            </div>

            <div style="color:#a5b4fc;margin:12px 0;">
                ⛰️ Landslide Detection
            </div>

            <div style="color:#6ee7b7;margin:12px 0;">
                ✓ Normal Region
            </div>

        </div>
        """)


# =====================================================
# 13. MODEL INFORMATION
# =====================================================

st.html("""
<div class="protocol-card">

    <div class="protocol-title">
        ⚙️ Model Information & System Details
    </div>

</div>
""")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.html("""
    <div class="metric-box">
        <div class="metric-label">AI ARCHITECTURE</div>
        <div class="metric-value">MHViT</div>
    </div>
    """)

with m2:
    st.html("""
    <div class="metric-box">
        <div class="metric-label">INPUT RESOLUTION</div>
        <div class="metric-value">224 × 224</div>
    </div>
    """)

with m3:
    st.html("""
    <div class="metric-box">
        <div class="metric-label">IMAGE SENSOR</div>
        <div class="metric-value">RGB</div>
    </div>
    """)

with m4:
    st.html(f"""
    <div class="metric-box">
        <div class="metric-label">COMPUTE DEVICE</div>
        <div class="metric-value">{str(DEVICE).upper()}</div>
    </div>
    """)


# =====================================================
# 14. FOOTER
# =====================================================

st.html("""
<div class="footer">

    DISASTERVISION AI
    &nbsp; | &nbsp;
    SATELLITE DISASTER CLASSIFICATION SYSTEM
    &nbsp; | &nbsp;
    POWERED BY MHViT DEEP LEARNING

</div>
""")
