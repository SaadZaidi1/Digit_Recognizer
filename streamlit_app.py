import streamlit as st
import numpy as np
import os

# Try to import TensorFlow, with fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. Running in demo mode.")

# Try to import PIL, with fallback
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.warning("⚠️ PIL not available. Image processing disabled.")

# Try to import matplotlib, with fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ Matplotlib not available. Plotting disabled.")

# Try to import the custom module, with fallback
try:
    from improved_digit_recognizer import ImprovedDigitRecognizer
    CUSTOM_MODULE_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODULE_AVAILABLE = False
    st.warning(f"⚠️ Custom module import failed: {e}")

# Try to import streamlit_drawable_canvas, with fallback
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError as e:
    CANVAS_AVAILABLE = False
    st.warning(f"⚠️ Canvas module import failed: {e}")

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
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available. Cannot load model.")
        return None
    
    if not CUSTOM_MODULE_AVAILABLE:
        st.error("❌ Custom module not available. Cannot load model.")
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
        st.warning("⚠️ Drawing canvas not available. Using file upload instead.")
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

def create_file_upload():
    """Create a file upload interface as fallback"""
    if not PIL_AVAILABLE:
        st.error("❌ PIL not available. File upload disabled.")
        return None
        
    st.markdown("### Upload a digit image")
    st.markdown("Upload an image of a handwritten digit (0-9):")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a handwritten digit"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the uploaded image
            processed_image = process_uploaded_image(image)
            return processed_image
        except Exception as e:
            st.error(f"❌ Error processing uploaded image: {e}")
            return None
    
    return None

def process_uploaded_image(image):
    """Process uploaded image for prediction"""
    try:
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Invert colors (white background to black)
        image_array = 255 - image_array
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
        return image_array
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

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
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
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
    if MATPLOTLIB_AVAILABLE:
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
    else:
        st.warning("⚠️ Matplotlib not available. Cannot display prediction chart.")

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
    if TENSORFLOW_AVAILABLE:
        st.sidebar.success("✅ TensorFlow: Available")
    else:
        st.sidebar.error("❌ TensorFlow: Missing")
    
    if CUSTOM_MODULE_AVAILABLE:
        st.sidebar.success("✅ Custom Module: Available")
    else:
        st.sidebar.error("❌ Custom Module: Missing")
    
    if CANVAS_AVAILABLE:
        st.sidebar.success("✅ Canvas: Available")
    else:
        st.sidebar.warning("⚠️ Canvas: Using File Upload")
    
    if PIL_AVAILABLE:
        st.sidebar.success("✅ PIL: Available")
    else:
        st.sidebar.error("❌ PIL: Missing")
    
    if MATPLOTLIB_AVAILABLE:
        st.sidebar.success("✅ Matplotlib: Available")
    else:
        st.sidebar.warning("⚠️ Matplotlib: Missing")

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">✏️ Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Draw a digit and watch the AI predict it!</p>', unsafe_allow_html=True)
    
    # Check if we can run the full app
    if not TENSORFLOW_AVAILABLE:
        st.error("""
        ❌ **TensorFlow Not Available!**
        
        The app cannot run without TensorFlow. This is likely a deployment issue.
        
        **Current Status:**
        - TensorFlow: ❌ Missing
        - PIL: {'✅ Available' if PIL_AVAILABLE else '❌ Missing'}
        - Matplotlib: {'✅ Available' if MATPLOTLIB_AVAILABLE else '❌ Missing'}
        
        **To fix this:**
        1. Check that `requirements.txt` includes `tensorflow`
        2. Ensure all dependencies are properly installed
        3. Try deploying with the simple test app first
        """)
        
        # Show demo mode
        st.info("🔄 **Demo Mode Active**")
        st.write("This is a demonstration of the app interface. Full functionality requires TensorFlow.")
        
        # Display model info in sidebar
        display_model_info()
        
        # Show a sample prediction
        st.markdown("### Sample Prediction (Demo)")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Demo Image:**")
            # Create a simple demo image
            demo_array = np.random.rand(28, 28) * 0.3
            demo_array[10:18, 10:18] = 0.8  # Create a "digit-like" pattern
            
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(demo_array, cmap='gray')
                ax.set_title('Demo Image (28x28)')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.write("Demo image (Matplotlib not available)")
        
        with col2:
            st.markdown("**Demo Results:**")
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Digit: <span style="color: #1f77b4; font-size: 2rem;">7</span></h3>
                <p>Confidence: <strong>87.5%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 This is a demo prediction. Real predictions require TensorFlow and a trained model.")
        
        return
    
    # Check other dependencies
    if not CUSTOM_MODULE_AVAILABLE:
        st.error("""
        ❌ **Custom Module Missing!**
        
        The `improved_digit_recognizer` module is not available.
        
        **To fix this:**
        1. Ensure all Python files are in the same directory
        2. Check that the repository structure is correct
        3. Make sure `improved_digit_recognizer.py` is present
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
        if CANVAS_AVAILABLE:
            st.markdown('<h2 class="sub-header">🎨 Drawing Canvas</h2>', unsafe_allow_html=True)
            
            # Create drawing canvas
            canvas_result = create_drawing_canvas()
            
            if canvas_result is not None:
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
        else:
            st.markdown('<h2 class="sub-header">📁 File Upload</h2>', unsafe_allow_html=True)
            
            # Create file upload interface
            processed_image = create_file_upload()
            
            if processed_image is not None:
                # Process button
                if st.button("🔍 Predict Digit", type="primary"):
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
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
        
        # Display prediction results if available
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            
            # Display the processed image
            st.markdown("**Processed Image:**")
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(prediction['image'], cmap='gray')
                ax.set_title('Processed Image (28x28)')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.write("Processed image (Matplotlib not available)")
            
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
