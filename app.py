import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FruitScan — Quality Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e8e8e8;
        font-family: 'Courier New', monospace;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="block-container"] { padding-top: 2rem; max-width: 680px; }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 1.1rem;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        font-size: 2rem;
        font-weight: 800;
        color: #f0f0f0;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-desc {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* Upload zone */
    .upload-label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 0.4rem;
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed #2e2e2e !important;
        border-radius: 4px;
        background: #13151c;
        padding: 1rem;
    }

    /* Image card */
    .img-card {
        background: #13151c;
        border: 1px solid #1e1e1e;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .img-label {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #444;
        margin-bottom: 0.6rem;
    }

    /* Verdict banner */
    .verdict-banner {
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-left: 4px solid;
    }
    .verdict-fresh   { background: #0a1f0f; border-color: #22c55e; }
    .verdict-rotten  { background: #1f1200; border-color: #f59e0b; }
    .verdict-formalin { background: #1f0a0a; border-color: #ef4444; }

    .verdict-label {
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .verdict-value {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }
    .verdict-fresh   .verdict-value { color: #22c55e; }
    .verdict-rotten  .verdict-value { color: #f59e0b; }
    .verdict-formalin .verdict-value { color: #ef4444; }

    .verdict-conf {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        color: #333;
    }
    .verdict-fresh   .verdict-conf { color: #16653a; }
    .verdict-rotten  .verdict-conf { color: #7a5200; }
    .verdict-formalin .verdict-conf { color: #7a1a1a; }

    /* Probability rows */
    .prob-section-label {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #444;
        margin-bottom: 0.8rem;
    }
    .prob-row {
        margin-bottom: 0.9rem;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }
    .prob-name {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
    }
    .prob-pct {
        font-size: 0.75rem;
        font-family: 'Courier New', monospace;
        color: #555;
    }
    .prob-track {
        height: 3px;
        background: #1e1e1e;
        border-radius: 2px;
        overflow: hidden;
    }
    .prob-fill-fresh    { background: #22c55e; height: 3px; border-radius: 2px; }
    .prob-fill-rotten   { background: #f59e0b; height: 3px; border-radius: 2px; }
    .prob-fill-formalin { background: #ef4444; height: 3px; border-radius: 2px; }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        color: #333;
        border-top: 1px solid #1a1a1a;
        margin-top: 2rem;
    }

    /* Hide streamlit default elements */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["formalin_mixed", "fresh", "rotten"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "mobilenetv3_transfer.keras"

VERDICT_CONFIG = {
    "formalin_mixed": {
        "css_class": "verdict-formalin",
        "display":   "FORMALIN DETECTED",
        "fill":      "prob-fill-formalin",
    },
    "fresh": {
        "css_class": "verdict-fresh",
        "display":   "FRESH",
        "fill":      "prob-fill-fresh",
    },
    "rotten": {
        "css_class": "verdict-rotten",
        "display":   "ROTTEN",
        "fill":      "prob-fill-rotten",
    },
}

# ─── Load Model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="app-title">GET 324 — Group C10</div>
    <div class="app-subtitle">FruitScan</div>
    <div class="app-desc">formalin detection via deep learning — MobileNetV3</div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────

st.markdown('<div class="upload-label">Upload fruit image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ─── Inference ────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.markdown('<div class="img-card"><div class="img-label">Input Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img_array = np.array(image.resize(IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    # Predict
    with st.spinner("Running inference..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = predictions[0][predicted_index] * 100
    cfg             = VERDICT_CONFIG[predicted_class]

    # Verdict banner
    st.markdown(f"""
    <div class="verdict-banner {cfg['css_class']}">
        <div>
            <div class="verdict-label">Classification Result</div>
            <div class="verdict-value">{cfg['display']}</div>
        </div>
        <div class="verdict-conf">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown('<div class="prob-section-label">Confidence breakdown</div>', unsafe_allow_html=True)

    for name, prob in zip(CLASS_NAMES, predictions[0]):
        pct      = prob * 100
        fill_cls = VERDICT_CONFIG[name]["fill"]
        display  = VERDICT_CONFIG[name]["display"]
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-header">
                <span class="prob-name">{display}</span>
                <span class="prob-pct">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="{fill_cls}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #2e2e2e; font-size: 0.8rem; letter-spacing: 0.1em;">
        NO IMAGE LOADED — UPLOAD A FRUIT IMAGE ABOVE
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    FRUITSCAN &nbsp;|&nbsp; MOBILENETV3 TRANSFER LEARNING &nbsp;|&nbsp; TENSORFLOW &amp; STREAMLIT
</div>
""", unsafe_allow_html=True)
