# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import io
from pathlib import Path
import requests
from inference_utils import load_model, get_preprocessing, postprocess_mask

st.set_page_config(page_title="Road Segmentation", layout="wide")

st.title("Road Extraction - Road Segmentation Inference")

# Sidebar options
st.sidebar.header("Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    options=["SegNet", "DeepLabV3Plus"],
    index=0,
    help="Choose which model architecture to use for inference"
)

# Paths and download configuration
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_CACHE_DIR = APP_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "SegNet": {
        "default_path": PROJECT_ROOT / "best_model_Segnet.pth",
        "cache_name": "segnet_weights.pth",
        "encoder": "vgg16",
        "model_type": "segnet",
        "secret_key": "SEGNET_WEIGHTS_URL",
    },
    "DeepLabV3Plus": {
        "default_path": PROJECT_ROOT / "best_model_DeeplabV3Plus.pth",
        "cache_name": "deeplabv3plus_weights.pth",
        "encoder": "resnet50",
        "model_type": "deeplabv3plus",
        "secret_key": "DEEPLAB_WEIGHTS_URL",
    },
}


def download_weights(url: str, target_path: Path):
    """Download model weights from a URL into the cache folder."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024
    downloaded = 0
    progress_bar = st.sidebar.progress(0.0, f"Downloading {target_path.name}")
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress_bar.progress(min(downloaded / total, 1.0), f"Downloading {target_path.name}")
    progress_bar.empty()


def secret_lookup(secret_key: str):
    """Look for a secret key either at top level or under [weights]."""
    if "weights" in st.secrets and secret_key in st.secrets["weights"]:
        return st.secrets["weights"][secret_key]
    return st.secrets.get(secret_key)


@st.cache_resource
def resolve_weights(model_name: str):
    """Ensure weights are available locally, downloading if necessary."""
    cfg = MODEL_CONFIG[model_name]
    default_path = cfg["default_path"]
    if default_path.exists():
        return str(default_path), cfg["model_type"], cfg["encoder"]

    cached_path = MODEL_CACHE_DIR / cfg["cache_name"]
    if cached_path.exists():
        return str(cached_path), cfg["model_type"], cfg["encoder"]

    secret_url = secret_lookup(cfg["secret_key"])
    if not secret_url:
        raise FileNotFoundError(
            f"{default_path} not found and no secret '{cfg['secret_key']}' supplied. "
            "Provide a download URL via Streamlit secrets."
        )

    with st.spinner(f"Downloading {model_name} weights..."):
        download_weights(secret_url, cached_path)
    return str(cached_path), cfg["model_type"], cfg["encoder"]


try:
    weights_path, model_type, encoder_name = resolve_weights(model_option)
except Exception as path_err:
    st.sidebar.error(f"Could not prepare weights: {path_err}")
    st.stop()

# Display the weights path (read-only)
st.sidebar.text_input("Model weights path", value=weights_path, disabled=True)

device_option = st.sidebar.selectbox("Device", options=["cpu"], index=0)
threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05)

# lazy load model with caching
@st.cache_resource
def load_inference_model(weights, model_type, encoder_name, device):
    model = load_model(weights, model_type=model_type, device=device)
    preprocess_fn = get_preprocessing(encoder_name, 'imagenet')
    return model, preprocess_fn

try:
    model, preprocess_fn = load_inference_model(weights_path, model_type, encoder_name, device_option)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.success(f"{model_option} model loaded")

# Upload or sample
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload RGB satellite image (jpg/png)", type=["jpg", "jpeg", "png"])
    use_sample = st.checkbox("Use sample image from dataset (if available)", value=False)

with col2:
    st.markdown("**Controls**")
    run_button = st.button("Run Inference")

# helper to show images side-by-side
def show_images(original, pred_mask, overlay):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(original, caption="Original", use_container_width=True)
    with c2:
        st.image(pred_mask, caption="Predicted mask", use_container_width=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True)

# Load image
if uploaded_file is None and not use_sample:
    st.info("Upload an image or enable 'Use sample'.")
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        orig = np.array(image)
    else:
        # if user picks sample, try to load from local sample_images folder
        try:
            sample_path = "sample_images/sample1.jpg"
            image = Image.open(sample_path).convert("RGB")
            orig = np.array(image)
        except Exception as e:
            st.error("No sample images found. Upload an image instead.")
            st.stop()

    st.write("Image size:", orig.shape[:2])

    if run_button:
        # preprocess
        preprocess = preprocess_fn
        img_pre, scale = preprocess(orig.copy(), target_size=512)
        input_tensor = torch.from_numpy(img_pre).unsqueeze(0)  # 1,C,H,W
        input_tensor = input_tensor.to(torch.device(device_option))

        with torch.no_grad():
            preds = model(input_tensor)
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()  # 1,1,H,W or 1,H,W depending
            # normalize shape to (1,H,W)
            if preds.ndim == 4:
                preds = preds[:, 0, :, :]
            else:
                preds = preds

        # threshold and postprocess
        preds_bin = (preds >= threshold).astype("uint8")
        mask = postprocess_mask(preds_bin, orig.shape[:2], scale)

        # create colored visuals
        mask_rgb = np.stack([mask*255]*3, axis=-1).astype("uint8")
        overlay = ((0.6 * orig.astype("float32") + 0.4 * mask_rgb.astype("float32"))).astype("uint8")

        # display
        show_images(orig, mask_rgb, overlay)

        # download mask
        buf = io.BytesIO()
        Image.fromarray(mask_rgb).save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download predicted mask", data=byte_im, file_name="pred_mask.png", mime="image/png")
