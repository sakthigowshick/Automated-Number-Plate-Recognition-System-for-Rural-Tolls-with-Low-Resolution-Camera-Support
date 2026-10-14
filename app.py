
from pathlib import Path

import cv2
import torch
import easyocr
import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = str(
    Path(__file__).resolve().parent
    / "models"
    / "best.pt"
)

st.set_page_config(
    page_title="Rural Toll ANPR",
    page_icon="🚘",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )
    return model, reader


# ---------------- UI ----------------
st.title("🚘 Smart Rural Toll – ANPR System")

st.caption(
    "AI-based Number Plate Detection and Recognition "
    "for low-resolution toll cameras"
)

gpu_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU mode"
)

st.success(f"Inference device: {gpu_name}")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# ---------------- IMAGE PROCESSING ----------------
if uploaded_file is not None:

    pil_image = Image.open(uploaded_file).convert("RGB")

    image = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    st.subheader("Input Image")

    # Smaller input image
    st.image(
        pil_image,
        caption="Original vehicle image",
        width=300
    )

    if st.button("🔍 Detect Number Plate", type="primary"):

        if not Path(MODEL_PATH).exists():
            st.error(f"Model not found: {MODEL_PATH}")
            st.stop()

        with st.spinner("Detecting plate and reading text..."):

            model, reader = load_models()

            results = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                device=(
                    0 if torch.cuda.is_available()
                    else "cpu"
                ),
                verbose=False
            )

            annotated = image.copy()
            records = []

            h, w = image.shape[:2]

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().tolist()
                )

                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Improve plate crop readability
                crop = cv2.resize(
                    crop,
                    None,
                    fx=3,
                    fy=3,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2GRAY
                )

                gray = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8)
                ).apply(gray)

                # OCR text recognition
                ocr = reader.readtext(gray, detail=1)

                plate_text = " ".join(
                    item[1]
                    for item in ocr
                    if item[2] >= 0.20
                ).strip()

                if not plate_text:
                    plate_text = "Not recognized"

                det_conf = float(box.conf[0].item())

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display recognized text on image
                cv2.putText(
                    annotated,
                    plate_text[:30],
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                records.append({
                    "Plate Text": plate_text,
                    "Detection Confidence": round(
                        det_conf, 3
                    ),
                    "Review Status": (
                        "Review required"
                        if plate_text == "Not recognized"
                        else "OCR result - verify"
                    )
                })

        # ---------------- OUTPUT ----------------
        st.divider()
        st.subheader("Detection Result")

        # Smaller output image
        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            ),
            caption="Detected number plates",
            width=400
        )

        if records:

            st.subheader("Recognized Plate Details")

            df = pd.DataFrame(records)

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download results as CSV",
                data=df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="anpr_results.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "No plate detected. Try another image "
                "or adjust the confidence threshold."
            )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Smart Rural Toll ANPR | YOLOv8n + EasyOCR + OpenCV"
)
