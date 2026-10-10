"""
Streamlit web app: upload a leaf photo -> disease prediction, top-5 chart and Grad-CAM heatmap.

Run from the project folder:
    streamlit run app/app.py

Design choices:
- Reuses load_checkpoint / preprocess / predict / gradcam_overlay from src/gradcam.py,
  so the app uses EXACTLY the same preprocessing as training and evaluation.
- Low-confidence warning: below CONFIDENCE_THRESHOLD the app says the result is
  uncertain instead of presenting a guess as fact (a simple risk control).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gradcam import gradcam_overlay, load_checkpoint, predict, preprocess, short_name  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
TEST_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_MODEL = "resnet50_rgb"
CONFIDENCE_THRESHOLD = 0.60
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_LABELS = {
    "resnet50_rgb": "ResNet50 (RGB) - best on clean images [default]",
    "resnet50_rgb_noise": "ResNet50 (RGB, noise-trained) - most robust to grainy photos",
    "densenet121_rgb": "DenseNet121 (RGB)",
    "resnet50_hsv": "ResNet50 (HSV) - experiment",
    "densenet121_hsv": "DenseNet121 (HSV) - experiment",
    "resnet50_hsv_noise": "ResNet50 (HSV, noise) - experiment",
    "densenet121_hsv_noise": "DenseNet121 (HSV, noise) - experiment",
}


@st.cache_resource(show_spinner="Loading model...")
def get_model(run: str):
    """Load once per model and reuse across interactions (Streamlit reruns the script on every click)."""
    return load_checkpoint(MODELS_DIR / f"{run}.pt")


def available_runs() -> list[str]:
    runs = sorted(p.stem for p in MODELS_DIR.glob("*.pt") if not p.stem.endswith("_quick"))
    # Default model first, then the rest in a sensible order.
    return sorted(runs, key=lambda r: (r != DEFAULT_MODEL, list(MODEL_LABELS).index(r) if r in MODEL_LABELS else 99))


def sample_images() -> dict[str, Path]:
    samples = {}
    if TEST_DIR.exists():
        for folder in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
            first = next((p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS), None)
            if first:
                samples[short_name(folder.name)] = first
    return samples


def top5_chart(probs: np.ndarray, class_names: list[str]):
    top = np.argsort(probs)[::-1][:5]
    labels = [short_name(class_names[i]) for i in top][::-1]
    values = [probs[i] * 100 for i in top][::-1]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(labels, values, color=["#9db4d6"] * 4 + ["#2f5d9e"])
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="wide")
    st.title("Plant Leaf Disease Detection")
    st.caption("Transfer learning (ResNet50 / DenseNet121) on 15 PlantVillage classes: tomato, potato and bell pepper. "
               "Grad-CAM shows which part of the leaf drove the prediction.")

    runs = available_runs()
    if not runs:
        st.error(f"No trained models found in {MODELS_DIR}. Run src/train.py first.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        run = st.selectbox("Model", runs, format_func=lambda r: MODEL_LABELS.get(r, r))
        show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
        st.divider()
        st.markdown(
            "**Limitations:** trained on lab photos (plain background, even lighting). "
            "Accuracy on field photos is likely lower. This is a decision-support tool, "
            "not a replacement for an expert diagnosis."
        )

    source = st.radio("Image source", ["Upload a photo", "Use a sample test image"], horizontal=True)
    image = None
    if source == "Upload a photo":
        uploaded = st.file_uploader("Leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
    else:
        samples = sample_images()
        if samples:
            choice = st.selectbox("Sample (true class shown)", list(samples))
            image = Image.open(samples[choice])
        else:
            st.info("No sample images found. Run src/prepare_data.py first, or upload a photo.")

    if image is None:
        st.stop()

    model, ckpt = get_model(run)
    class_names = ckpt["class_names"]
    tensor, rgb = preprocess(image, ckpt["color_mode"])
    probs = predict(model, tensor)
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    label = short_name(class_names[pred_idx])

    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"**Prediction: {label}** ({confidence:.1%} confidence)")
    else:
        st.warning(f"**Uncertain: most likely {label}** ({confidence:.1%} confidence). "
                   f"Confidence is below {CONFIDENCE_THRESHOLD:.0%}, so check the top-5 list "
                   "and confirm with an expert.")

    col_img, col_cam = st.columns(2)
    col_img.image(rgb, caption="Input (resized to 224x224)", width="stretch")
    if show_cam:
        with st.spinner("Computing Grad-CAM..."):
            overlay = gradcam_overlay(model, ckpt["model_name"], tensor, rgb, pred_idx)
        col_cam.image(overlay, caption=f"Grad-CAM for '{label}' (red = most influential)",
                      width="stretch")

    st.subheader("Top-5 predictions")
    st.pyplot(top5_chart(probs, class_names))


if __name__ == "__main__":
    main()
