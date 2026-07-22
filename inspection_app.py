"""
Part Inspection App - Simple Version (No Login, No Cloud)
--------------------------------------------------------------
Select Site -> Select Case/Block -> Upload/Capture Photo ->
Model Inspects -> Report shown -> User Confirms (Good / Wrong) ->
Photo saved to a local folder on your computer.

===========================================================================
SETUP
===========================================================================

1. Install required packages (run once in terminal):
     pip install streamlit ultralytics pillow numpy

2. Place your trained model file "best.pt" in this same folder.

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

st.set_page_config(page_title="Tafe Quality Check", layout="centered")

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

            # Count raw detections per exact class name
            raw_counts = {}
            for box in result.boxes:
                raw_name = names[int(box.cls[0])]
                raw_counts[raw_name] = raw_counts.get(raw_name, 0) + 1

            # -----------------------------------------------------
            # BOLT logic - BOLT_Missing is the real, trusted missing
            # count straight from detection (note: model's actual class
            # name is "BOLT_Missing", not "BOLT_MISSING"). Present is
            # derived from expected - missing.
            # -----------------------------------------------------
            bolt_missing = raw_counts.get("BOLT_Missing", 0)
            bolt_present = max(0, EXPECTED_COUNTS["bolt"] - bolt_missing)

            # -----------------------------------------------------
            # NUT logic - no "missing" class trained yet, so missing
            # is inferred by comparing present count to expected total.
            # -----------------------------------------------------
            nut_present = raw_counts.get("NUT_P", 0)
            nut_missing = max(0, EXPECTED_COUNTS["nut"] - nut_present)

            # -----------------------------------------------------
            # BREATHER logic - same inference approach as nut.
            # -----------------------------------------------------
            breather_present = raw_counts.get("BREATHER_PRESENT", 0)
            breather_missing = max(0, EXPECTED_COUNTS["breather"] - breather_present)

        st.subheader("Inspection Report")

        bolt_expected = EXPECTED_COUNTS["bolt"]
        icon = "✅" if bolt_missing == 0 else "⚠️"
        st.write(f"{icon} **Bolt**: {bolt_present} / {bolt_expected} present"
                 + (f" — {bolt_missing} missing" if bolt_missing > 0 else ""))

        nut_expected = EXPECTED_COUNTS["nut"]
        icon = "✅" if nut_missing == 0 else "⚠️"
        st.write(f"{icon} **Nut**: {nut_present} / {nut_expected} present"
                 + (f" — {nut_missing} missing" if nut_missing > 0 else ""))

        breather_expected = EXPECTED_COUNTS["breather"]
        icon = "✅" if breather_missing == 0 else "⚠️"
        st.write(f"{icon} **Breather**: {breather_present} / {breather_expected} present"
                 + (f" — {breather_missing} missing" if breather_missing > 0 else ""))

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
