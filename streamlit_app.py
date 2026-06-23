"""
streamlit_app.py
----------------
Modern Streamlit application for the AI Image Detection system.

Run:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Path setup ──
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

# ────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRNU Forensic Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark-forensics palette */
    :root {
        --bg: #0d1117;
        --card: #161b22;
        --accent-real: #3fb950;
        --accent-ai: #f78166;
        --accent-blue: #58a6ff;
        --text: #c9d1d9;
    }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: var(--card);
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid var(--accent-blue);
    }
    .verdict-real {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-real);
    }
    .verdict-ai {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-ai);
    }
    .feature-pill {
        display: inline-block;
        background: #21262d;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.78rem;
        color: #8b949e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 PRNU Forensic Detector")
    st.markdown(
        "**Sensor-noise forensic analysis** using DB8 wavelet denoising "
        "and multi-group residual features to distinguish real camera images "
        "from AI-generated ones."
    )
    st.markdown("---")
    st.markdown("**Pipeline**")
    steps = [
        "1. DB8 Wavelet Denoising",
        "2. Noise Residual Extraction",
        "3. Multi-Group Feature Extraction",
        "4. ML Classifier",
        "5. Real / AI Prediction",
    ]
    for s in steps:
        st.markdown(f"&nbsp;&nbsp;→ {s}", unsafe_allow_html=True)

    st.markdown("---")
    models_dir = st.text_input("Models directory", value="outputs/models")

# ────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────
st.title("AI Image Detection via PRNU Forensic Analysis")
st.markdown(
    "Upload any image to analyse its sensor-noise residual and determine "
    "whether it was captured by a real camera or synthesised by an AI model."
)

# ────────────────────────────────────────────────────────────────────
# Upload
# ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("👆 Upload an image above to begin analysis.")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# Run pipeline
# ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes, filename: str, models_dir: str):
    """Cache pipeline results so re-renders don't re-process."""
    import tempfile, os

    with tempfile.NamedTemporaryFile(
        suffix=Path(filename).suffix, delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        from predict import predict as _predict
        result = _predict(tmp_path, models_dir)
    finally:
        os.unlink(tmp_path)

    return result


with st.spinner("🔬 Running forensic analysis…"):
    try:
        result = run_pipeline(
            uploaded.read(), uploaded.name, models_dir
        )
    except FileNotFoundError as e:
        st.error(
            f"**Model not found.** {e}\n\n"
            "Train the model first with:\n```\npython src/train.py\n```"
        )
        st.stop()
    except Exception as e:
        st.error(f"**Analysis failed:** {e}")
        st.stop()

# ────────────────────────────────────────────────────────────────────
# Layout: 3 image columns
# ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

def arr_to_pil(arr: np.ndarray) -> Image.Image:
    vis = np.clip(arr, 0, 1) * 255
    return Image.fromarray(vis.astype(np.uint8))

def residual_to_pil(arr: np.ndarray) -> Image.Image:
    r = arr.copy()
    r -= r.min()
    mx = r.max()
    if mx > 1e-10:
        r /= mx
    return Image.fromarray((r * 255).astype(np.uint8))

with col1:
    st.subheader("Original Image")
    st.image(arr_to_pil(result["arrays"]["original"]), use_container_width=True)

with col2:
    st.subheader("Denoised (DB8)")
    st.image(arr_to_pil(result["arrays"]["denoised"]), use_container_width=True)

with col3:
    st.subheader("Noise Residual")
    st.image(residual_to_pil(result["arrays"]["residual"]), use_container_width=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────
# Verdict
# ────────────────────────────────────────────────────────────────────
label = result["prediction"]
conf = result["confidence"]
is_ai = result["label_int"] == 1

vcol1, vcol2 = st.columns([2, 3])
with vcol1:
    verdict_class = "verdict-ai" if is_ai else "verdict-real"
    icon = "🤖" if is_ai else "📷"
    st.markdown(
        f'<div class="metric-card">'
        f'<p style="margin:0;color:#8b949e;font-size:0.85rem">CLASSIFICATION</p>'
        f'<p class="{verdict_class}">{icon}&nbsp;{label}</p>'
        f'<p style="margin:0;color:#8b949e;font-size:0.9rem">Confidence: '
        f'<strong style="color:#c9d1d9">{conf:.1f}%</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

with vcol2:
    # Confidence gauge
    bar_color = "#f78166" if is_ai else "#3fb950"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.35},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#21262d"},
                {"range": [50, 100], "color": "#21262d"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Confidence", "font": {"size": 14}},
    ))
    fig_gauge.update_layout(
        height=200, margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────
# Feature values
# ────────────────────────────────────────────────────────────────────
st.subheader("🧮 Extracted Forensic Features")

features = result["features"]
feat_df = pd.DataFrame(
    [{"Feature": k, "Value": v, "Group": k.split("_")[0].upper()}
     for k, v in features.items()]
)

tab_bar, tab_table = st.tabs(["📊 Bar Chart", "📋 Full Table"])

with tab_bar:
    # Show top 20 by absolute value
    top = feat_df.reindex(
        feat_df["Value"].abs().nlargest(20).index
    ).copy()
    top["Colour"] = top["Value"].apply(lambda v: "#58a6ff" if v >= 0 else "#f78166")

    fig_bar = px.bar(
        top, x="Value", y="Feature", orientation="h",
        color="Group",
        title="Top 20 Features by Magnitude",
        height=520,
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_table:
    st.dataframe(
        feat_df.style.format({"Value": "{:.6f}"}),
        use_container_width=True,
        height=400,
    )

st.markdown("---")

# ────────────────────────────────────────────────────────────────────
# Feature group radar
# ────────────────────────────────────────────────────────────────────
st.subheader("🕸️ Feature Group Energy Radar")

group_energy = (
    feat_df.assign(Energy=feat_df["Value"] ** 2)
    .groupby("Group")["Energy"]
    .sum()
    .reset_index()
)
fig_radar = go.Figure(go.Scatterpolar(
    r=group_energy["Energy"].values,
    theta=group_energy["Group"].values,
    fill="toself",
    fillcolor="rgba(88,166,255,0.2)",
    line_color="#58a6ff",
))
fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, color="#8b949e"),
        angularaxis=dict(color="#8b949e"),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    height=380,
    title="Feature Group Energy Distribution",
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────
# Forensic explanation
# ────────────────────────────────────────────────────────────────────
with st.expander("📖 What does each feature group tell us?"):
    st.markdown(
        """
| Group | Key features | Forensic meaning |
|---|---|---|
| **A – Statistics** | skewness, kurtosis, entropy | Real sensor noise is signal-dependent; AI residuals are more Gaussian or exhibit sharper peaks |
| **B – Energy** | total_energy, avg_energy | PRNU injects consistent energy; AI generators produce atypical energy levels |
| **C – FFT** | spectral_entropy, high/low freq ratios | GANs leave grid-frequency artifacts; diffusion models alter low-freq energy from the U-Net |
| **D – Autocorrelation** | max, mean, variance | Sensor noise decays rapidly; AI residuals have long-range spatial structure from convolutions |
| **E – Wavelet** | HH/HL/LH/LL energy & entropy | Real images concentrate noise in HH; AI residuals often leak into LL from coarse-to-fine synthesis |
| **F – PRNU** | noise_variance, local_consistency | True PRNU is spatially consistent; AI patterns vary locally in characteristic ways |
        """
    )

# ────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="color:#484f58;font-size:0.8rem;text-align:center;">'
    "PRNU-Inspired AI Image Forensic Detector · DB8 Wavelet · "
    "RandomForest / XGBoost / SVM / LightGBM / MLP"
    "</p>",
    unsafe_allow_html=True,
)
