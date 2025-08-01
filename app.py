import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

st.header('Flower Classification CNN Model')

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.h5')
img_size = 180  # Change to your model's expected input size

def classify_images(image):
    input_image = image.resize((img_size, img_size))
    input_image_array = np.array(input_image) / 255.0
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = (
        f'The Image belongs to {flower_names[np.argmax(result)]} '
        f'with a score of {np.max(result)*100:.2f}%'
    )
    return outcome

uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=180)
    st.markdown(classify_images(image))
