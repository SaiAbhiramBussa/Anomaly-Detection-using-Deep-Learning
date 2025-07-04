import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import os
import re
import csv
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Anomaly Detection System")
st.markdown("""
This application provides advanced anomaly detection capabilities for handwritten digit images using multiple deep learning models.
Features include real-time analysis, batch processing, and detailed visualization tools.
""")

# Sidebar controls
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Denoising Autoencoder", "Variational Autoencoder", "GAN-based"]
)

# Advanced threshold controls
st.sidebar.subheader("Threshold Settings")
threshold = st.sidebar.slider(
    "Base Anomaly Detection Threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.1,
    step=0.01
)

# Additional threshold parameters
use_adaptive_threshold = st.sidebar.checkbox("Use Adaptive Thresholding", value=False)
if use_adaptive_threshold:
    sensitivity = st.sidebar.slider("Adaptive Sensitivity", 0.1, 2.0, 1.0, 0.1)
    window_size = st.sidebar.slider("Adaptive Window Size", 5, 50, 20, 5)

# Visualization options
st.sidebar.subheader("Visualization Options")
show_heatmap = st.sidebar.checkbox("Show Anomaly Heatmap", value=True)
show_3d = st.sidebar.checkbox("Show 3D Reconstruction", value=False)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Load the pre-trained models
@st.cache_resource
def load_models():
    models = {}
    try:
        models['dae'] = keras.models.load_model('mnist_dae_model.h5')
        # Add other models here when available
        return models
    except:
        st.error("Model files not found. Please ensure model files are in the project directory.")
        return None

# Enhanced preprocessing function
def preprocess_image(image, target_size=(28, 28)):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model input
    image = image.reshape(1, *target_size, 1)
    return image

# Enhanced anomaly detection function
def detect_anomaly(model, image, threshold, use_adaptive=False, sensitivity=1.0, window_size=20):
    # Get reconstruction
    reconstruction = model.predict(image)
    
    # Calculate reconstruction error
    error = np.mean(np.square(image - reconstruction))
    
    # Calculate pixel-wise error map
    error_map = np.square(image - reconstruction)
    
    # Adaptive thresholding
    if use_adaptive:
        local_errors = np.convolve(error_map.flatten(), 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        adaptive_threshold = np.mean(local_errors) * sensitivity
        is_anomaly = error > adaptive_threshold
    else:
        is_anomaly = error > threshold
    
    # Calculate confidence score
    confidence = 1 - (error / (threshold * 2))  # Normalized confidence score
    confidence = np.clip(confidence, 0, 1)
    
    return error, reconstruction, is_anomaly, error_map, confidence

# Create main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    # File uploader with batch processing support
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process each uploaded image
        for uploaded_file in uploaded_files:
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Load models and detect anomaly
            models = load_models()
            if models is not None:
                error, reconstruction, is_anomaly, error_map, confidence = detect_anomaly(
                    models['dae'], 
                    processed_image,
                    threshold,
                    use_adaptive_threshold,
                    sensitivity if use_adaptive_threshold else 1.0,
                    window_size if use_adaptive_threshold else 20
                )
                
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Display reconstruction
                    st.image(reconstruction[0, :, :, 0], caption="Reconstructed Image", use_container_width=True)
                    
                    # Display metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Reconstruction Error", f"{error:.4f}")
                    with metrics_col2:
                        st.metric("Anomaly Status", "Anomaly Detected" if is_anomaly else "Normal Image")
                    with metrics_col3:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Create advanced visualizations
                    if show_heatmap:
                        # Error heatmap
                        fig_heatmap = px.imshow(error_map[0, :, :, 0],
                                              title="Anomaly Heatmap",
                                              color_continuous_scale='Viridis')
                        st.plotly_chart(fig_heatmap)
                    
                    if show_3d:
                        # 3D surface plot
                        fig_3d = go.Figure(data=[go.Surface(z=reconstruction[0, :, :, 0])])
                        fig_3d.update_layout(title='3D Reconstruction',
                                           scene=dict(
                                               xaxis_title='X',
                                               yaxis_title='Y',
                                               zaxis_title='Intensity'
                                           ))
                        st.plotly_chart(fig_3d)
                    
                    # Error analysis plots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Error distribution histogram with KDE
                    sns.histplot([error], bins=20, ax=ax1, kde=True)
                    ax1.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
                    ax1.set_title("Error Distribution with KDE")
                    ax1.set_xlabel("Reconstruction Error")
                    ax1.set_ylabel("Frequency")
                    ax1.legend()
                    
                    # Error threshold visualization with confidence interval
                    ax2.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                    ax2.scatter([0], [error], color='blue', s=100)
                    if use_adaptive_threshold:
                        ax2.axhline(y=threshold * sensitivity, color='g', linestyle=':', 
                                  label='Adaptive Threshold')
                    ax2.set_title("Error vs Threshold")
                    ax2.set_xlabel("Sample")
                    ax2.set_ylabel("Error")
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                    # Additional statistics
                    st.subheader("Detailed Statistics")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.write("Error Statistics:")
                        error_stats = {
                            "Mean": np.mean(error_map),
                            "Std Dev": np.std(error_map),
                            "Max": np.max(error_map),
                            "Min": np.min(error_map)
                        }
                        st.write(pd.DataFrame.from_dict(error_stats, orient='index', 
                                                     columns=['Value']))
                    
                    with stats_col2:
                        st.write("Reconstruction Quality Metrics:")
                        quality_metrics = {
                            "PSNR": 20 * np.log10(1.0 / np.sqrt(error)),
                            "SSIM": np.mean(stats.pearsonr(
                                image.flatten(), 
                                reconstruction.flatten()
                            )[0])
                        }
                        st.write(pd.DataFrame.from_dict(quality_metrics, orient='index', 
                                                     columns=['Value']))

# Add footer with timestamp
st.markdown("---")
st.markdown(f"Built with Streamlit • Powered by TensorFlow • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 