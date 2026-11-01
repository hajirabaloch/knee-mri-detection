import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

MODEL_REPO = "hajirabaloch488/knee-mri-yolo11n"
MODEL_FILE = "knee_mri_yolo11n.pt"
IMG_SIZE = 320

st.set_page_config(page_title="Knee MRI Detection", page_icon="🦴", layout="wide")


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return YOLO(path)


st.title("🦴 Knee MRI Structure Detection")
st.warning(
    "**Research prototype - not for medical use.** This model was fine-tuned on a very small "
    "academic dataset and is NOT clinically validated. Do not use it for diagnosis or any "
    "medical decision."
)
st.write(
    "Upload a 2D knee MRI slice (PNG/JPG). The model (YOLO11n fine-tuned from "
    "[Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11)) draws boxes around "
    "four structures: **ACL, PCL, Meniscus, Cartilage**."
)

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05,
                     help="Lower = more boxes (more false alarms). Higher = fewer boxes.")
    st.markdown("---")
    st.markdown(
        "**Model:** [hajirabaloch488/knee-mri-yolo11n](https://huggingface.co/hajirabaloch488/knee-mri-yolo11n)"
    )
    st.markdown("**Test mAP50:** 0.212 (small dataset, noisy)")

with st.expander("About the model, data and limitations"):
    st.markdown(
        """
- **Data:** fastMRI knee single-coil MRI with fastMRI+ bounding-box annotations
  (59 MRI files, 534 annotated slices, 953 boxes). The data is not included in this app.
- **Split:** by MRI file - 41 train / 9 validation / 9 test.
- **Test results (9 unseen MRI files):** mAP50 0.212, mAP50-95 0.064.
  Best class: PCL (mAP50 0.393). Weakest: Cartilage (0.085).
- **Works best on:** coronal knee slices similar to fastMRI images (320x320, grayscale).
  It may fail on other scanners, orientations or image types.
- The model can miss structures and can draw boxes where there is nothing.
        """
    )

uploaded = st.file_uploader("Upload a knee MRI slice", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image to start.")
else:
    image = Image.open(uploaded).convert("RGB")
    model = load_model()
    results = model.predict(np.array(image), imgsz=IMG_SIZE, conf=conf, verbose=False)
    r = results[0]
    annotated = r.plot()[:, :, ::-1]  # BGR -> RGB

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, width="stretch")
    with col2:
        st.subheader("Detections")
        st.image(annotated, width="stretch")

    if len(r.boxes) == 0:
        st.info("No detections at this confidence threshold. Try lowering it in the sidebar.")
    else:
        rows = []
        for box in r.boxes:
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            rows.append({
                "Structure": r.names[int(box.cls[0])],
                "Confidence": round(float(box.conf[0]), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        df = pd.DataFrame(rows).sort_values("Confidence", ascending=False).reset_index(drop=True)
        st.subheader(f"{len(df)} detection(s)")
        st.dataframe(df, width="stretch")
        st.caption("Box coordinates are in pixels of the image as seen by the model.")
