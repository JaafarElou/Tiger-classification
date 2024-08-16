import joblib
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

naive = joblib.load("another_one.pkl")

def preprocess_image(image):
    image = resize(image,(15, 15)) 
    image = image.flatten()
    image = np.asarray(image)
    #image = np.array(image) / 255.0 
    #image = np.expand_dims(image, axis=0) 
    return image.reshape(1, -1)

def predict(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

st.title("Image Classification App")

st.write("Upload an image and let the model classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Load the image
    image = imread(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Make prediction
    predictions = predict(image, naive)
    

    # Display the predictions
    st.title("Predictions:")
    if predictions == 1:
        st.header("The image has a tiger.")
    else:
        st.header("The image has not a tiger.")