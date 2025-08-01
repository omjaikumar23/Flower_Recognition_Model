import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Flower Classifier")
st.header('🌼 Flower Classification CNN Model')

# Flower classes (ensure these match your model's output)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@st.cache_resource
def load_flower_model():
    try:
        model = load_model('Flower_Recog_Model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_flower_model()
img_size = 180  # Should match your model's expected input size

def classify_image(image):
    try:
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocess image
        input_image = image.resize((img_size, img_size))
        input_image_array = np.array(input_image) / 255.0
        
        # Add batch dimension
        if len(input_image_array.shape) == 3:
            input_image_exp_dim = np.expand_dims(input_image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        
        # Get top prediction
        confidence = np.max(result) * 100
        predicted_class = flower_names[np.argmax(result)]
        
        return f"The image belongs to **{predicted_class}** with {confidence:.2f}% confidence"
    
    except Exception as e:
        return f"Error during classification: {str(e)}"

# File uploader
uploaded_file = st.file_uploader('Upload a flower image', 
                                type=['jpg', 'jpeg', 'png'],
                                help="Supported formats: JPG, JPEG, PNG")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width=200)
        with col2:
            with st.spinner('Classifying...'):
                result = classify_image(image)
                st.success(result)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

if model is None:
    st.warning("Model not loaded - classification unavailable")
