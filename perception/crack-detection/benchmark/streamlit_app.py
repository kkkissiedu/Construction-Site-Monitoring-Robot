import os
import streamlit as st
from ultralytics import solutions

# 1. Setup layout (Must be first)
st.set_page_config(page_title="Crack Detection App", layout="wide")

# 2. Sidebar: Device Selection (CPU vs GPU)
st.sidebar.title("Hardware Options")
device_choice = st.sidebar.radio(
    "Select Inference Device:",
    options=["GPU (CUDA)", "CPU"],
    index=0  # Default to GPU
)
# Convert UI text to Ultralytics device parameters ('0' for CUDA GPU, 'cpu' for CPU)
selected_device = "0" if device_choice == "GPU (CUDA)" else "cpu"

# 3. Sidebar: Model File Selection
st.sidebar.title("Model Settings")
MODELS_DIR = "models"

if os.path.exists(MODELS_DIR):
    available_models = [
        f for f in os.listdir(MODELS_DIR) 
        if f.endswith(('.pt', '.onnx', '.engine'))
    ]
else:
    available_models = []

if not available_models:
    st.sidebar.error(f"No models found in '{MODELS_DIR}/' folder.")
    selected_model_path = "models/yolo26s-seg.pt"
else:
    selected_model_name = st.sidebar.selectbox("Choose Model:", available_models)
    selected_model_path = os.path.join(MODELS_DIR, selected_model_name)

st.sidebar.markdown("---")

# 4. Initialize Inference with Selected Model & Hardware Device
# Notice we DO NOT pass a static source path. This forces the UI to provide a video upload button.
inf = solutions.Inference(
    model=selected_model_path,
    device=selected_device,
    conf=0.25,
    iou=0.45
)

# 5. Render the UI
inf.inference()

# streamlit run streamlit_app.py