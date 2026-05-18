from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import dl_utils as dl

MODEL_PATH = Path(__file__).parent / "brain_tumor_model.h5"

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #1e3a5f, #3d7ab8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #5a6a7a; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef4fb 100%);
        border: 1px solid #d8e4f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading neural network…")
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


def render_sidebar():
    st.sidebar.title("⚙️ Analysis settings")
    threshold = st.sidebar.slider(
        "Classification threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability above this value is classified as tumor.",
    )
    use_tta = st.sidebar.checkbox(
        "Test-time augmentation (TTA)",
        value=True,
        help="Average predictions over flips/rotations for more stable scores.",
    )
    show_heatmap = st.sidebar.checkbox("Show attention heatmap", value=True)
    heatmap_method = st.sidebar.selectbox(
        "Heatmap method",
        ["Grad-CAM (auto)", "Occlusion sensitivity"],
        disabled=not show_heatmap,
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Research / educational tool only. Not for clinical diagnosis."
    )
    return threshold, use_tta, show_heatmap, heatmap_method


def render_prediction_card(label: str, confidence: float, risk: str, prob: float):
    tumor = label == dl.CLASS_LABELS[1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", label)
    with col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with col3:
        st.metric("Risk level", risk)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Tumor probability")
        st.progress(min(max(prob, 0.0), 1.0))
    with c2:
        st.caption("No-tumor probability")
        st.progress(min(max(1.0 - prob, 0.0), 1.0))

    if tumor:
        st.error(f"Tumor signal detected — model score {prob * 100:.2f}%")
    else:
        st.success(f"No tumor signal — healthy confidence {(1 - prob) * 100:.2f}%")


def analyze_single(
    model,
    file,
    threshold: float,
    use_tta: bool,
    show_heatmap: bool,
    heatmap_method: str,
):
    display, batch = dl.preprocess(file)
    prob, tta_scores = dl.predict_with_tta(model, display, use_tta=use_tta)
    label, risk, confidence = dl.classify(prob, threshold)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("MRI scan")
        st.image(display, use_container_width=True, channels="RGB")

    with right:
        st.subheader("Deep learning result")
        render_prediction_card(label, confidence, risk, prob)

        if use_tta and len(tta_scores) > 1:
            with st.expander("TTA variant scores", expanded=False):
                st.bar_chart(pd.Series(tta_scores))

    if show_heatmap:
        st.subheader("Model attention (where the network looks)")
        heatmap = None
        if heatmap_method.startswith("Grad-CAM"):
            heatmap = dl.make_gradcam_heatmap(model, batch, display)
        if heatmap is None:
            with st.spinner("Computing occlusion sensitivity map…"):
                heatmap = dl.occlusion_sensitivity_map(model, display)
            method_note = "Occlusion sensitivity"
        else:
            method_note = "Grad-CAM"

        overlay = dl.overlay_heatmap(display, heatmap)
        h1, h2, h3 = st.columns(3)
        h1.image(display, caption="Input", use_container_width=True)
        h2.image((heatmap * 255).astype(np.uint8), caption="Heatmap", use_container_width=True)
        h3.image(overlay, caption=f"Overlay ({method_note})", use_container_width=True)

    return {
        "filename": getattr(file, "name", "upload"),
        "tumor_probability": prob,
        "prediction": label,
        "confidence": confidence,
        "risk": risk,
    }


def tab_detect(model, threshold, use_tta, show_heatmap, heatmap_method):
    st.markdown('<p class="sub-header">Upload a single MRI image for CNN-based tumor screening.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "MRI image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="single_upload",
    )
    if uploaded:
        analyze_single(model, uploaded, threshold, use_tta, show_heatmap, heatmap_method)


def tab_batch(model, threshold, use_tta):
    st.markdown('<p class="sub-header">Run inference on multiple scans and export results.</p>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Multiple MRI images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not files:
        st.info("Upload one or more images to run batch inference.")
        return

    if st.button("Run batch analysis", type="primary"):
        rows = []
        progress = st.progress(0, text="Processing…")
        for i, f in enumerate(files):
            display, _ = dl.preprocess(f)
            prob, _ = dl.predict_with_tta(model, display, use_tta=use_tta)
            label, risk, confidence = dl.classify(prob, threshold)
            rows.append(
                {
                    "File": f.name,
                    "Tumor probability": round(prob, 4),
                    "Prediction": label,
                    "Confidence %": round(confidence * 100, 2),
                    "Risk": risk,
                }
            )
            progress.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("File")["Tumor probability"])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results (CSV)",
            data=csv,
            file_name="brain_tumor_batch_results.csv",
            mime="text/csv",
        )


def tab_preprocess_lab(model):
    st.markdown('<p class="sub-header">Explore how preprocessing and augmentation affect the model.</p>', unsafe_allow_html=True)
    file = st.file_uploader("MRI for preprocessing lab", type=["jpg", "jpeg", "png"], key="lab_upload")
    if not file:
        return

    augment = st.selectbox(
        "Augmentation",
        ["none", "flip_h", "flip_v", "rotate_90", "brighten", "contrast"],
        format_func=lambda x: x.replace("_", " ").title() if x != "none" else "None",
    )
    display, batch = dl.preprocess(file, augment=None if augment == "none" else augment)
    norm = dl.normalize(display)
    prob = dl.predict_proba(model, batch)

    c1, c2, c3, c4 = st.columns(4)
    c1.image(display, caption="Resized 224×224", use_container_width=True)
    c2.image((norm * 255).astype(np.uint8), caption="Normalized input", use_container_width=True)
    c3.metric("Raw probability", f"{prob:.4f}")
    c4.metric("Augmentation", augment)

    edges = cv2.Canny(display, 80, 160)
    st.image(edges, caption="Edge map (OpenCV Canny)", use_container_width=True)


def tab_model_insights(model):
    st.markdown('<p class="sub-header">Neural network architecture and parameter statistics.</p>', unsafe_allow_html=True)
    info = dl.model_summary_dict(model)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total layers", info["layers"])
    m2.metric("Conv layers", info["conv_layers"])
    m3.metric("Trainable params", f"{info['trainable_params']:,}")
    m4.metric("Total parameters", f"{info['total_params']:,}")

    st.write(f"**Input shape:** `{info['input_shape']}`")
    st.write(f"**Output shape:** `{info['output_shape']}`")

    st.subheader("Layer stack (top 25)")
    st.dataframe(pd.DataFrame(dl.layer_table(model)), use_container_width=True, hide_index=True)

    st.subheader("Feature map preview")
    st.caption("First convolutional block activation on a sample (if available).")
    sample = st.file_uploader("Optional sample MRI", type=["jpg", "jpeg", "png"], key="feat_upload")
    if sample:
        display, batch = dl.preprocess(sample)
        try:
            conv = dl.find_last_conv_layer(model)
            if conv is not None:
                feat_model = tf.keras.Model(model.input, conv.output)
                features = feat_model.predict(batch, verbose=0)[0]
                # show mean across channels
                fmap = np.mean(features, axis=-1)
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                fmap = cv2.resize(fmap, (display.shape[1], display.shape[0]))
                st.image(fmap, caption=f"Mean activation — {conv.name}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not extract feature maps: {exc}")


def main():
    st.markdown('<p class="main-header">🧠 Brain Tumor Detection AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">CNN-powered MRI analysis with Grad-CAM, TTA, and batch inference.</p>',
        unsafe_allow_html=True,
    )

    threshold, use_tta, show_heatmap, heatmap_method = render_sidebar()

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Could not load model at `{MODEL_PATH}`.")
        st.code(str(exc))
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔬 Detect", "📁 Batch", "🧪 Preprocessing", "🤖 Model"]
    )
    with tab1:
        tab_detect(model, threshold, use_tta, show_heatmap, heatmap_method)
    with tab2:
        tab_batch(model, threshold, use_tta)
    with tab3:
        tab_preprocess_lab(model)
    with tab4:
        tab_model_insights(model)


if __name__ == "__main__":
    main()
