import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- UI COLORS (ONLY ADDITION) ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(20deg, #000000, #0a2540, #1e88e5);
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: white;
}

p, label, span {
    color: #e3f2fd;
}

.stButton > button {
    background-color: #1e88e5;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1565c0;
    color: white;
}

div[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 15px;
    border-radius: 10px;
}

div.stAlert-success {
    background-color: rgba(30, 136, 229, 0.25);
    color: white;
}

div.stAlert-info {
    background-color: rgba(255, 255, 255, 0.15);
    color: white;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------


st.title("🐟 Multiclass Fish Image Classification")
st.write("YOLOv8 Classification Model")

MODEL_PATH = "runs/classify/train/weights/best.pt"

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        results = model(image)

        probs = results[0].probs
        class_id = probs.top1
        confidence = probs.top1conf
        class_name = results[0].names[class_id]

        st.success(f"🐠 Predicted Species: **{class_name}**")
        st.info(f"Confidence: **{confidence:.2%}**")
