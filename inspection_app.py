"""
Part Inspection App - Simple Version (No Login, No Cloud)
--------------------------------------------------------------
Select Site -> Select Case/Block -> Upload/Capture Photo ->
Model Inspects -> User Confirms (Good / Wrong) -> Photo saved
to a local folder on your computer.

===========================================================================
SETUP
===========================================================================

1. Install required packages (run once in terminal):
     pip install streamlit ultralytics pillow numpy

2. Place your trained model file "best.pt" in this same folder once ready.
   Until then, the app runs fine but skips the detection step.

3. Edit SITE_CODES and CASE_TYPES below to match your real setup.

4. Run the app:
     streamlit run inspection_app.py
"""

import os
import datetime
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# CONFIG - EDIT THESE TO MATCH YOUR REAL SETUP
# ---------------------------------------------------------------------

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.30

EXPECTED_COUNTS = {
    "bolt": 17,
    "nut": 4,
    "breather": 1,
}

SITE_CODES = ["HLC", "Diff-Case", "Cell-2 MainLine"]
CASE_TYPES = ["Top-View", "Front-View", "Right-View", "Left-View"]

# Local folders where photos get saved (created automatically if missing)
APPROVED_FOLDER = "saved_photos/approved"
FLAGGED_FOLDER = "saved_photos/flagged"

os.makedirs(APPROVED_FOLDER, exist_ok=True)
os.makedirs(FLAGGED_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

st.set_page_config(page_title="Part Inspector", layout="centered")

model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
    except Exception as e:
        model_error = str(e)
else:
    model_error = "No trained model found yet (best.pt missing). Detection step will be skipped."


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("Tafe Quality Check")

site = st.selectbox("Select Assembly", SITE_CODES, index=0)
case = st.selectbox("Select View", CASE_TYPES, index=0)

st.divider()
st.subheader(f"{site} — {case}")

capture_mode = st.radio("Photo source", ["Upload from file", "Use camera"], horizontal=True)

if capture_mode == "Upload from file":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a photo")


# ---------------------------------------------------------------------
# MAIN INSPECTION FLOW
# ---------------------------------------------------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Captured image", use_container_width=True)

    if model is None:
        st.warning(model_error)
    else:
        with st.spinner("Analyzing image..."):
            results = model.predict(source=np.array(image), conf=CONF_THRESHOLD, verbose=False)
            result = results[0]
            names = result.names

            counts = {}
            for box in result.boxes:
                raw_name = names[int(box.cls[0])]
                upper_name = raw_name.upper()

                if "IGNORE" in upper_name:
                    continue  # skip irrelevant detections entirely

                # Determine present vs missing regardless of exact naming style
                # (handles BOLT_Missing, NUT_Missing, BREATHER_PRESENT, BOLT_P, NUT_P, etc.)
                if "MISSING" in upper_name or upper_name.endswith("_M"):
                    state = "missing"
                elif "PRESENT" in upper_name or upper_name.endswith("_P"):
                    state = "present"
                else:
                    continue  # unrecognized class naming, skip safely

                # Extract the part name (everything before the first underscore)
                part = raw_name.split("_")[0].lower()

                counts.setdefault(part, {"present": 0, "missing": 0})
                counts[part][state] += 1

        st.subheader("Inspection Report")
        all_ok = True
        for part, expected_total in EXPECTED_COUNTS.items():
            present = counts.get(part, {}).get("present", 0)
            missing = counts.get(part, {}).get("missing", 0)
            icon = "✅" if missing == 0 else "⚠️"
            if missing > 0:
                all_ok = False
            st.write(f"{icon} **{part.capitalize()}**: {present} / {expected_total} present"
                     + (f" — {missing} missing" if missing > 0 else ""))

        annotated = result.plot()
        st.image(annotated, caption="Detected parts", use_container_width=True)

    st.divider()
    st.write("**Is this result correct?**")

    col1, col2 = st.columns(2)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{site}_{case}_{timestamp}.jpg".replace(" ", "_")

    with col1:
        if st.button("✅ Looks correct - save as good data"):
            save_path = os.path.join(APPROVED_FOLDER, base_filename)
            image.save(save_path)
            st.success(f"Saved to {save_path}")

    with col2:
        if st.button("❌ Something's wrong - flag for retraining"):
            save_path = os.path.join(FLAGGED_FOLDER, base_filename)
            image.save(save_path)
            st.success(f"Saved to {save_path}")

else:
    st.info("Upload or capture a photo to begin inspection.")
