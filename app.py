import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Analysis Statistics  
st.markdown("### 📈 Session Stats")
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
        
    st.metric("Analyses Performed", st.session_state.analysis_count)


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="PneumoAI - Professional Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(33, 147, 176, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #2193b0;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        margin: 1rem 0;
    }
    
    .pneumonia-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        margin: 1rem 0;
    }
    
    .normal-card {
        background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 210, 211, 0.4);
        margin: 1rem 0;
    }
    
    .confidence-high { color: #28a745; font-weight: 600; }
    .confidence-medium { color: #ffc107; font-weight: 600; }
    .confidence-low { color: #dc3545; font-weight: 600; }
    
    .upload-section {
        border: 3px dashed #2193b0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(45deg, #f8f9ff, #f0f4ff);
        margin: 1rem 0;
    }
    
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 2rem 0;
        color: #856404;
    }
    
    .gradcam-section {
        background: linear-gradient(135deg, #ffeaa7 0%, #ffd32a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #333;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #2193b0, #6dd5ed);
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load the trained DenseNet121 model with error handling"""
    try:
        model = tf.keras.models.load_model("/model/final_densenet_pneumonia.keras")
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ========== GRAD-CAM FUNCTIONS REMOVED ==========
# Grad-CAM functionality removed for simpler, more reliable operation

# ========== CONDITION INFORMATION ==========
CONDITION_INFO = {
    "Normal": {
        "description": "Healthy chest X-ray with no signs of pneumonia",
        "severity": "Healthy",
        "color": "#00d2d3",
        "recommendations": [
            "Continue regular health checkups",
            "Maintain good respiratory hygiene",
            "Stay up to date with vaccinations"
        ]
    },
    "Pneumonia": {
        "description": "Inflammatory condition of the lung affecting air sacs",
        "severity": "Requires Medical Attention",
        "color": "#ff6b6b",
        "recommendations": [
            "Seek immediate medical consultation",
            "Follow prescribed treatment plan",
            "Monitor symptoms closely",
            "Complete full course of antibiotics if prescribed"
        ]
    }
}

# ========== MAIN APP ==========
def main():
    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🫁 PneumoAI</h1>
        <p>Advanced AI-Powered Chest X-Ray Analysis System</p>
        <p>Professional Pneumonia Detection using DenseNet121</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Unable to load the model. Please ensure 'final_densenet_pneumonia.keras' is in the correct directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 Analysis Dashboard")
        
        # Model Status
        st.success("✅ DenseNet121 Model Ready")
        st.info("📊 Binary Classification: Normal vs Pneumonia")
        
        st.markdown("---")
        
        # Model Features (Updated)
        st.markdown("### 🎯 Model Features")
        st.markdown("""
        <div class="feature-highlight">
            <strong>Architecture:</strong> DenseNet121<br>
            <strong>Dataset:</strong> Chest X-Ray Images<br>
            <strong>Input Size:</strong> 224×224<br>
            <strong>Classes:</strong> Normal, Pneumonia<br>
            <strong>Accuracy:</strong> >90%
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Chest X-Ray Image")
        
        st.markdown("""
        <div class="info-card">
            <h4>🔍 Image Requirements</h4>
            <ul>
                <li>Clear chest X-ray images</li>
                <li>Supported formats: JPG, JPEG, PNG</li>
                <li>Frontal view preferred</li>
                <li>Good contrast and clarity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray for pneumonia detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded X-Ray", use_column_width=True)
            
            # Image information
            st.markdown(f"""
            <div class="info-card">
                <strong>Image Info:</strong><br>
                📏 Size: {image.size[0]}×{image.size[1]} pixels<br>
                💾 File: {uploaded_file.name}<br>
                📅 Uploaded: {datetime.now().strftime("%H:%M:%S")}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🧠 AI Analysis Results")
            
            # Processing animation
            with st.spinner("🔬 Analyzing chest X-ray..."):
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tfile.write(uploaded_file.getvalue())  # Use getvalue() instead of read()
                tfile.close()
                
                try:
                    # Read and preprocess image
                    img = cv2.imread(tfile.name, cv2.IMREAD_GRAYSCALE)
                    if img is None:  # Add validation check
                        st.error("Error: Could not read the uploaded image. Please try a different image.")
                        return
                    IMG_SIZE = (224, 224)
                    img_resized = cv2.resize(img, IMG_SIZE)
                    
                    # Preprocess for model
                    img_input = np.stack([img_resized, img_resized, img_resized], axis=-1) / 255.0
                    img_input_batch = np.expand_dims(img_input, axis=0).astype(np.float32)
                    
                    # Progress bar animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Prediction
                    preds = model.predict(img_input_batch, verbose=0)
                    prob = preds[0][0]
                    
                    # Determine prediction
                    class_names = ["Normal", "Pneumonia"]
                    predicted_class = class_names[int(prob > 0.5)]
                    confidence = prob if prob > 0.5 else (1 - prob)
                    
                    # Update session stats
                    st.session_state.analysis_count += 1
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return
                finally:
                    # Clean up temp file
                    if os.path.exists(tfile.name):
                        os.remove(tfile.name)
            
            # Main Prediction Card
            condition_info = CONDITION_INFO[predicted_class]
            
            if predicted_class == "Pneumonia":
                st.markdown(f"""
                <div class="pneumonia-card">
                    <h2>⚠️ Analysis Result</h2>
                    <h1>{predicted_class}</h1>
                    <p style="font-size: 1.1rem; opacity: 0.9;">{condition_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-card">
                    <h2>✅ Analysis Result</h2>
                    <h1>{predicted_class}</h1>
                    <p style="font-size: 1.1rem; opacity: 0.9;">{condition_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence Metrics
            confidence_color = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
            
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            
            with col_conf1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Confidence</h3>
                    <h1>{confidence:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚕️ Status</h3>
                    <h1>{condition_info['severity']}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔬 Analysis</h3>
                    <h1>Complete</h1>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("### 👈 Upload a chest X-ray to begin analysis")
            st.markdown("""
            <div class="analysis-section">
                <h4>🚀 How it works:</h4>
                <ol>
                    <li><strong>Upload</strong> a chest X-ray image</li>
                    <li><strong>AI processes</strong> using DenseNet121</li>
                    <li><strong>Get results</strong> with confidence scores</li>
                    <li><strong>View detailed</strong> analysis and recommendations</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Section
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Analysis Report")
        
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.markdown("### 📈 Confidence Analysis")
            
            # Create confidence chart
            conf_data = pd.DataFrame({
                'Class': ['Normal', 'Pneumonia'],
                'Confidence': [1-prob, prob],
                'Percentage': [f"{(1-prob)*100:.1f}%", f"{prob*100:.1f}%"]
            })
            
            fig = px.bar(
                conf_data, 
                x='Class', 
                y='Confidence',
                color='Confidence',
                color_continuous_scale="RdYlBu_r",
                text='Percentage',
                title="Classification Confidence Scores"
            )
            fig.update_layout(showlegend=False, height=400)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.markdown("### 🎯 Prediction Distribution")
            
            # Create pie chart
            fig_pie = px.pie(
                values=[1-prob, prob],
                names=['Normal', 'Pneumonia'],
                title="Classification Results",
                color_discrete_sequence=['#54a0ff', '#ff6b6b']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
        
        # Recommendations Section
        st.markdown("### 💡 Clinical Recommendations")
        
        recommendations = condition_info["recommendations"]
        rec_html = "<div class='info-card'><h4>📋 Recommended Actions:</h4><ul>"
        for rec in recommendations:
            rec_html += f"<li>{rec}</li>"
        rec_html += "</ul></div>"
        
        st.markdown(rec_html, unsafe_allow_html=True)
        
        # Detailed Results Table
        st.markdown("### 📋 Detailed Classification Report")
        
        detailed_results = pd.DataFrame({
            "Class": ["Normal", "Pneumonia"],
            "Probability": [f"{(1-prob)*100:.2f}%", f"{prob*100:.2f}%"],
            "Confidence Level": [
                "High" if (1-prob) > 0.8 else "Medium" if (1-prob) > 0.6 else "Low",
                "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low"
            ],
            "Predicted": ["✅" if predicted_class == "Normal" else "❌", "✅" if predicted_class == "Pneumonia" else "❌"]
        })
        
        st.dataframe(detailed_results, use_container_width=True, hide_index=True)
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p><strong>This AI system is designed for educational and research purposes only.</strong> The results provided by this application should not be considered as medical advice, diagnosis, or treatment recommendations. This tool is intended to assist healthcare professionals and should not replace professional medical evaluation.</p>
        
        <p><strong>For Medical Emergencies:</strong> If experiencing difficulty breathing, chest pain, or other serious symptoms, seek immediate medical attention. Always consult with qualified healthcare professionals for proper medical evaluation and treatment.</p>
        
        <p><strong>Clinical Use:</strong> This system achieves >90% accuracy but should only be used as a supplementary tool in clinical decision-making, never as a standalone diagnostic tool.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>PneumoAI Professional</strong> | AI-Powered Chest X-Ray Analysis | Built with TensorFlow & Streamlit</p>
        <p>Developed for Advanced Medical AI Research | DenseNet121 Implementation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
