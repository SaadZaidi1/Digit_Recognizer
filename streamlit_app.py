import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Try to import the custom module, with fallback
try:
    from improved_digit_recognizer import ImprovedDigitRecognizer
    CUSTOM_MODULE_AVAILABLE = True
except ImportError as e:
    st.warning(f"Custom module import failed: {e}")
    CUSTOM_MODULE_AVAILABLE = False

# Try to import streamlit_drawable_canvas, with fallback
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Canvas module import failed: {e}")
    CANVAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with fallback options"""
    if not CUSTOM_MODULE_AVAILABLE:
        st.error("❌ Custom module not available. Please check deployment.")
        return None
    
    try:
        recognizer = ImprovedDigitRecognizer()
        
        # Check if model exists, if not train a new one
        if os.path.exists('improved_digit_model.h5'):
            recognizer.load_model('improved_digit_model.h5')
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ Model not found. Training a new model...")
            with st.spinner("Training model... This may take a few minutes."):
                recognizer.load_data()
                recognizer.create_model()
                recognizer.train_model(epochs=10)  # Reduced epochs for faster training
                recognizer.save_model()
            st.success("✅ Model trained and saved!")
        
        return recognizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_drawing_canvas():
    """Create a drawing canvas using Streamlit with fallback"""
    st.markdown("### Draw a digit (0-9)")
    st.markdown("Use your mouse to draw a digit in the box below:")
    
    if not CANVAS_AVAILABLE:
        st.error("❌ Drawing canvas not available. Please check dependencies.")
        return None
    
    # Create canvas using streamlit's canvas component
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    return canvas_result

def process_canvas_image(canvas_result):
    """Process the canvas image for prediction"""
    if canvas_result is None or canvas_result.image_data is None:
        return None
        
    try:
        # Convert to PIL Image
        image = Image.fromarray(canvas_result.image_data)
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Invert colors (white background to black)
        image_array = 255 - image_array
        
        return image_array
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

def display_prediction_results(predicted_digit, confidence, all_predictions):
    """Display prediction results with visual elements"""
    st.markdown("### Prediction Results")
    
    # Main prediction display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Predicted Digit: <span style="color: #1f77b4; font-size: 2rem;">{predicted_digit}</span></h3>
            <p>Confidence: <strong>{confidence:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence bar
        st.markdown("**Confidence Level:**")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{confidence:.2%}</p>", unsafe_allow_html=True)
    
    # All predictions chart
    st.markdown("### All Digit Probabilities")
    fig, ax = plt.subplots(figsize=(12, 6))
    digits = list(range(10))
    bars = ax.bar(digits, all_predictions, color='skyblue', alpha=0.7)
    
    # Highlight the predicted digit
    bars[predicted_digit].set_color('red')
    bars[predicted_digit].set_alpha(0.8)
    
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities for All Digits')
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, all_predictions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def display_model_info():
    """Display model information and statistics"""
    st.sidebar.markdown("## Model Information")
    
    # Model accuracy (you can update this based on your model's performance)
    st.sidebar.metric("Test Accuracy", "98.5%")
    st.sidebar.metric("Model Type", "CNN")
    st.sidebar.metric("Training Data", "MNIST Dataset")
    
    st.sidebar.markdown("### Model Architecture")
    st.sidebar.markdown("""
    - **Input Layer**: 28x28x1 (Grayscale)
    - **Convolutional Layers**: 3 blocks with batch normalization
    - **Pooling Layers**: MaxPooling2D
    - **Dropout**: 25-50% for regularization
    - **Dense Layers**: 256 → 128 → 10 neurons
    - **Output**: Softmax (10 classes)
    """)
    
    st.sidebar.markdown("### Features")
    st.sidebar.markdown("""
    ✅ Real-time drawing recognition
    ✅ Confidence scoring
    ✅ Data augmentation during training
    ✅ Early stopping and learning rate scheduling
    ✅ Batch normalization for better training
    """)
    
    # Show deployment status
    st.sidebar.markdown("### Deployment Status")
    if CUSTOM_MODULE_AVAILABLE:
        st.sidebar.success("✅ Custom Module: Available")
    else:
        st.sidebar.error("❌ Custom Module: Missing")
    
    if CANVAS_AVAILABLE:
        st.sidebar.success("✅ Canvas: Available")
    else:
        st.sidebar.error("❌ Canvas: Missing")

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">✏️ Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Draw a digit and watch the AI predict it!</p>', unsafe_allow_html=True)
    
    # Check dependencies
    if not CUSTOM_MODULE_AVAILABLE or not CANVAS_AVAILABLE:
        st.error("""
        ❌ **Deployment Error Detected!**
        
        The following dependencies are missing:
        - `improved_digit_recognizer` module
        - `streamlit-drawable-canvas` package
        
        **To fix this:**
        1. Make sure `requirements.txt` includes `streamlit-drawable-canvas>=0.9.0`
        2. Ensure all Python files are in the same directory
        3. Check that the repository structure is correct
        
        **Current requirements.txt should contain:**
        ```
        tensorflow>=2.19.0
        numpy>=1.26.0
        pandas>=2.3.0
        scikit-learn>=1.7.0
        streamlit>=1.28.0
        pillow>=10.0.0
        matplotlib>=3.7.0
        seaborn>=0.12.0
        streamlit-drawable-canvas>=0.9.0
        ```
        """)
        return
    
    # Load model
    recognizer = load_model()
    if recognizer is None:
        st.error("❌ Failed to load model. Please check the deployment.")
        return
    
    # Display model info in sidebar
    display_model_info()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎨 Drawing Canvas</h2>', unsafe_allow_html=True)
        
        # Create drawing canvas
        canvas_result = create_drawing_canvas()
        
        if canvas_result is None:
            return
        
        # Clear button
        if st.button("🗑️ Clear Canvas", type="secondary"):
            st.rerun()
        
        # Process button
        if st.button("🔍 Predict Digit", type="primary"):
            if canvas_result.image_data is not None:
                # Process the drawn image
                processed_image = process_canvas_image(canvas_result)
                
                if processed_image is not None:
                    try:
                        # Make prediction
                        predicted_digit, confidence, all_predictions = recognizer.predict_digit(processed_image)
                        
                        # Store results in session state
                        st.session_state.prediction = {
                            'digit': predicted_digit,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'image': processed_image
                        }
                        
                        st.success("✅ Prediction completed!")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                else:
                    st.error("❌ No drawing detected. Please draw a digit first.")
            else:
                st.error("❌ No drawing detected. Please draw a digit first.")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
        
        # Display prediction results if available
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            
            # Display the processed image
            st.markdown("**Processed Image:**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(prediction['image'], cmap='gray')
            ax.set_title('Processed Image (28x28)')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            
            # Display prediction results
            display_prediction_results(
                prediction['digit'],
                prediction['confidence'],
                prediction['all_predictions']
            )
        else:
            st.info("👆 Draw a digit and click 'Predict Digit' to see results!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Model trained on MNIST dataset with CNN architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
