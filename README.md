# Road Extraction using CNN-based Architectures

Dual-model road segmentation pipeline that compares **SegNet** and **DeepLabV3+** on the DeepGlobe dataset. The repo contains the original training notebooks plus a Streamlit inference app that lets you switch between both `.pth` checkpoints at runtime.

## Highlights
- ⚙️ **Two architectures**: VGG16-based SegNet and ResNet50 DeepLabV3+.
- 🧠 **Shared preprocessing** via `segmentation_models_pytorch`.
- 🖥️ **Streamlit UI** with model selector, sample image toggle, side-by-side visualization, and mask download.
- 📓 **Full training notebooks** for reproducibility/tweaks.

## Repository layout
```
├── Deployment/
│   ├── streamlit_app.py          # Streamlit UI (model switch, inference utilities)
│   ├── inference_utils.py        # SegNet + DeepLabV3+ loaders & preprocessing
│   └── requirements.txt          # Runtime dependencies
├── Road_Extraction_Segnet_Segmentation.ipynb
├── Road_Extraction_DeepalbV3_Segmentation.ipynb
├── best_model_Segnet.pth         # expected location for SegNet weights (not committed)
└── best_model_DeeplabV3Plus.pth  # expected location for DeepLabV3+ weights (not committed)
```

> **Note:** Keep the `.pth` checkpoints at the project root so the Streamlit app can auto-discover them. You can change the paths in `streamlit_app.py` if needed.

## Requirements
Python 3.10+ (tested on 3.13) with:
- `streamlit==1.40.2`
- `torch==2.7.1`, `torchvision==0.22.1`
- `segmentation-models-pytorch==0.5.0`
- `opencv-python`, `numpy`, `pillow`, `albumentations`
- `requests`

Install everything from the Deployment folder:
```bash
cd Deployment
pip install -r requirements.txt
```

## Running the Streamlit app locally
```bash
cd Deployment
streamlit run streamlit_app.py
```

The sidebar lets you:
1. Select **SegNet** or **DeepLabV3+** (auto-loads the corresponding checkpoint).
2. Choose CPU/GPU (if CUDA is available) and threshold.
3. Upload an image or enable “Use sample image”.

Outputs:
- Predicted mask
- Overlay (mask on RGB)
- Download button (`pred_mask.png`)

## Training notebooks
- `Road_Extraction_Segnet_Segmentation.ipynb`
- `Road_Extraction_DeepalbV3_Segmentation.ipynb`

Both notebooks include:
- Data ingestion from the DeepGlobe dataset
- Augmentations (`albumentations`)
- Loss/metric setup (Dice, IoU)
- Training + validation loops
- Checkpoint saving

## Deploying to Streamlit Community Cloud
1. Fork/clone this repo.
2. Upload/commit `Deployment/streamlit_app.py`, `Deployment/inference_utils.py`, `Deployment/requirements.txt`, and place checkpoints in a storage bucket (S3, GDrive, HuggingFace, etc.).
3. Create a Streamlit app at [share.streamlit.io](https://share.streamlit.io), targeting `Deployment/streamlit_app.py`.
4. Add download URLs to `.streamlit/secrets.toml` so the app can fetch weights automatically:
   ```toml
   [weights]
   SEGNET_WEIGHTS_URL = "https://<your-storage>/best_model_Segnet.pth"
   DEEPLAB_WEIGHTS_URL = "https://<your-storage>/best_model_DeeplabV3Plus.pth"
   ```
   Secrets take priority when the expected local files are missing and will be downloaded into `Deployment/model_cache/`.

## GitHub deployment
1. Initialize git (if needed) and add this repo as remote:
   ```bash
   git init
   git add .
   git commit -m "Initial commit with Streamlit dual-model inference"
   git branch -M main
   git remote add origin https://github.com/Suryanshambekar/Road-Extraction-using-CNN-based-Architectures.git
   git push -u origin main
   ```
2. Add Git LFS or release assets if you plan to ship the `.pth` files via GitHub.

## Next steps
- Evaluate newer encoders (e.g., EfficientNet) via `segmentation_models_pytorch`.
- Add metrics panel to the Streamlit app for quick comparisons.
- Automate download of checkpoints from cloud storage to keep repo lightweight.

Happy mapping! 🌍


