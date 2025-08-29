import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Custom CSS for dark theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app background */
    .main .block-container {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fafafa !important;
    }
    
    /* Success/Info/Error boxes dark theme */
    .stAlert > div {
        background-color: #1e2124;
        border: 1px solid #40444b;
    }
    
    .stSuccess > div {
        background-color: #2d5a27;
        border: 1px solid #4caf50;
        color: #ffffff;
    }
    
    .stInfo > div {
        background-color: #1a365d;
        border: 1px solid #2196f3;
        color: #ffffff;
    }
    
    .stError > div {
        background-color: #5d1a1a;
        border: 1px solid #f44336;
        color: #ffffff;
    }
    
    /* File uploader dark theme */
    .stFileUploader > div > div {
        background-color: #262730;
        border: 2px dashed #40444b;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2d5a87;
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    .stButton > button:hover {
        background-color: #1e3a5f;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #40444b;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #fafafa;
    }
    
    /* Sidebar content */
    .css-17eq0hr {
        color: #fafafa;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* Custom dark styling for prediction results */
    .prediction-card {
        background-color: #1e2124;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #40444b;
        margin: 10px 0;
    }
    
    .top-prediction {
        background: linear-gradient(135deg, #2d5a87, #1e3a5f);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #4caf50;
    }
    
    .prediction-item {
        background-color: #262730;
        padding: 8px 15px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 3px solid #40444b;
    }
    
    .medal-gold { border-left-color: #ffd700 !important; }
    .medal-silver { border-left-color: #c0c0c0 !important; }
    .medal-bronze { border-left-color: #cd7f32 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained CIFAR-10 model"""
    try:
        # Try to load the best model first
        if os.path.exists('models/best_cifar10_model.keras'):
            model = tf.keras.models.load_model('models/best_cifar10_model.keras')
            model_name = "Best CIFAR-10 Model"
        elif os.path.exists('models/cifar10_cnn_final_model.keras'):
            model = tf.keras.models.load_model('models/cifar10_cnn_final_model.keras')
            model_name = "Final CIFAR-10 Model (Keras)"
        elif os.path.exists('models/cifar10_cnn_final_model.h5'):
            model = tf.keras.models.load_model('models/cifar10_cnn_final_model.h5')
            model_name = "Final CIFAR-10 Model (H5)"
        else:
            # Load from architecture and weights
            with open('models/cifar10_cnn_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('models/cifar10_cnn_weights.weights.h5')
            model_name = "CIFAR-10 Model (Architecture + Weights)"
        
        return model, model_name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the uploaded image for CIFAR-10 prediction"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize to 32x32 (CIFAR-10 input size)
    image_resized = cv2.resize(image, (32, 32))
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def predict_image(model, image):
    """Make prediction on the preprocessed image"""
    try:
        predictions = model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def display_prediction_results(predicted_class, confidence, all_predictions):
    """Display prediction results with dark theme styling"""
    
    # Main prediction with custom styling
    st.markdown(f"""
    <div class="top-prediction">
        <h3>🎯 Predicted Class: {CIFAR10_CLASSES[predicted_class]}</h3>
        <h4>Confidence: {confidence:.2%}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence bar
    st.progress(float(confidence))
    
    # Display all class probabilities
    st.markdown("### 📊 All Class Probabilities")
    
    # Create a dataframe for better visualization
    prob_data = []
    for i, prob in enumerate(all_predictions):
        prob_data.append({
            'Class': CIFAR10_CLASSES[i],
            'Probability': prob,
            'Percentage': f"{prob:.2%}"
        })
    
    # Sort by probability (descending)
    prob_data.sort(key=lambda x: x['Probability'], reverse=True)
    
    # Display with custom styling
    for i, data in enumerate(prob_data):
        class_name = "medal-gold" if i == 0 else "medal-silver" if i == 1 else "medal-bronze" if i == 2 else ""
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        
        st.markdown(f"""
        <div class="prediction-item {class_name}">
            {emoji} <strong>{data['Class']}</strong>: {data['Percentage']}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="CIFAR-10 Image Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS for dark theme
    load_custom_css()
    
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.markdown("Upload an image to classify it into one of the 10 CIFAR-10 categories!")
    
    # Load model
    with st.spinner("Loading model..."):
        model, model_name = load_model()
    
    if model is None:
        st.error("Could not load the model. Please ensure the model files are in the same directory as this script.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully: {model_name}")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📋 CIFAR-10 Classes")
        
        # Display classes with dark theme styling
        for i, class_name in enumerate(CIFAR10_CLASSES):
            st.markdown(f"""
            <div class="prediction-item">
                <strong>{i}</strong>: {class_name}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div class="prediction-card">
        This app uses a CNN model trained on the CIFAR-10 dataset to classify images into 10 categories.
        
        <strong>Image requirements:</strong>
        • Any format (JPG, PNG, etc.)
        • Will be resized to 32x32 pixels
        • Best results with clear, centered objects
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Add predict button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(model, processed_image)
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions
                        }
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        if hasattr(st.session_state, 'prediction_results'):
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Display prediction results with dark theme
            display_prediction_results(predicted_class, confidence, all_predictions)
            
        else:
            st.markdown("""
            <div class="prediction-card">
                Upload an image and click 'Classify Image' to see prediction results.
            </div>
            """, unsafe_allow_html=True)
    
    # Additional information
    st.markdown("---")
    st.markdown("### 🔧 Model Information")
    if model is not None:
        try:
            total_params = model.count_params()
            st.markdown(f"""
            <div class="prediction-card">
                <strong>Total Parameters:</strong> {total_params:,}<br>
                <strong>Input Shape:</strong> {model.input_shape}<br>
                <strong>Output Shape:</strong> {model.output_shape}
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="prediction-card">
                Model information not available
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()