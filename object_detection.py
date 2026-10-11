import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="AI Object Detection - Task 3", page_icon="👁️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4a5568;'>👁️ AI Object Detection Suite</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096;'>CodeAlpha AI Internship | Task 3: Powered by YOLOv8 & Streamlit</p>", unsafe_allow_html=True)
st.markdown("---")

# Load YOLOv8 model (using the lightweight nano model for instant loading)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

with st.spinner("Loading AI Computer Vision Model... 🤖"):
    model = load_model()

# File uploader for images
uploaded_file = st.file_uploader("Upload an image for object detection...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Original Image")
        st.image(image, use_container_width=True)
        
    if st.button("🚀 Run Object Detection", type="primary"):
        with st.spinner("Analyzing image frame..."):
            # Convert PIL image to OpenCV array format
            img_array = np.array(image)
            
            # Run YOLO prediction
            results = model(img_array)
            res_plotted = results[0].plot() # Draws bounding boxes
            
            with col2:
                st.markdown("### 🎯 Detected Objects")
                st.image(res_plotted, use_container_width=True)
                
            st.success("Object detection completed successfully! 🎉")
            st.balloons()
else:
    st.info("💡 Tip: Upload any street, room, or everyday object photo to see YOLOv8 detect and label items in real-time!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #a0aec0;'>Built with Python, Streamlit & Ultralytics YOLOv8</p>", unsafe_allow_html=True)