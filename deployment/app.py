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

@st.cache_resource
def load_model():
    """Load the trained CIFAR-10 model"""
    try:
        # Try to load the best model first
        if os.path.exists('best_cifar10_model.keras'):
            model = tf.keras.models.load_model('best_cifar10_model.keras')
            model_name = "Best CIFAR-10 Model"
        elif os.path.exists('cifar10_cnn_final_model.keras'):
            model = tf.keras.models.load_model('cifar10_cnn_final_model.keras')
            model_name = "Final CIFAR-10 Model (Keras)"
        elif os.path.exists('cifar10_cnn_final_model.h5'):
            model = tf.keras.models.load_model('cifar10_cnn_final_model.h5')
            model_name = "Final CIFAR-10 Model (H5)"
        else:
            # Load from architecture and weights
            with open('cifar10_cnn_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('cifar10_cnn_weights.weights.h5')
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

def main():
    st.set_page_config(
        page_title="CIFAR-10 Image Classifier",
        page_icon="🖼️",
        layout="wide"
    )
    
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
        st.header("📋 CIFAR-10 Classes")
        for i, class_name in enumerate(CIFAR10_CLASSES):
            st.write(f"{i}: {class_name}")
        
        st.header("ℹ️ About")
        st.write("This app uses a CNN model trained on the CIFAR-10 dataset to classify images into 10 categories.")
        st.write("**Image requirements:**")
        st.write("- Any format (JPG, PNG, etc.)")
        st.write("- Will be resized to 32x32 pixels")
        st.write("- Best results with clear, centered objects")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
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
        st.header("🎯 Prediction Results")
        
        if hasattr(st.session_state, 'prediction_results'):
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Display main prediction
            st.success(f"**Predicted Class:** {CIFAR10_CLASSES[predicted_class]}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            # Display confidence bar
            st.progress(float(confidence))
            
            # Display all class probabilities
            st.subheader("📊 All Class Probabilities")
            
            # Create a dataframe for better visualization
            prob_data = []
            for i, prob in enumerate(all_predictions):
                prob_data.append({
                    'Class': CIFAR10_CLASSES[i],
                    'Probability': f"{prob:.4f}",
                    'Percentage': f"{prob:.2%}"
                })
            
            # Sort by probability (descending)
            prob_data.sort(key=lambda x: float(x['Probability']), reverse=True)
            
            # Display as a table
            for i, data in enumerate(prob_data):
                if i == 0:  # Highlight top prediction
                    st.markdown(f"🥇 **{data['Class']}**: {data['Percentage']}")
                elif i == 1:
                    st.markdown(f"🥈 {data['Class']}: {data['Percentage']}")
                elif i == 2:
                    st.markdown(f"🥉 {data['Class']}: {data['Percentage']}")
                else:
                    st.write(f"{data['Class']}: {data['Percentage']}")
        else:
            st.info("Upload an image and click 'Classify Image' to see prediction results.")
    
    # Additional information
    st.markdown("---")
    st.markdown("### 🔧 Model Information")
    if model is not None:
        try:
            total_params = model.count_params()
            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
        except:
            st.write("Model information not available")

if __name__ == "__main__":
    main()