import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('model.h5')


def preprocess_image(image):

    image = tf.keras.preprocessing.image.img_to_array(image)
  
    image = tf.expand_dims(image, axis=0)  # Ad
    return image


def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction


st.title("Skin Disease Classification")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


    prediction = predict(image)
    class_names = ['Acne and Rosacea Photos','Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions','Atopic Dermatitis Photos','Bullous Disease Photos','Cellulitis Impetigo and other Bacterial Infections','Eczema Photos','Exanthems and Drug Eruptions','Hair Loss Photos Alopecia and other Hair Diseases','Herpes HPV and other STDs Photos','Light Diseases and Disorders of Pigmentation','Lupus and other Connective Tissue diseases','Melanoma Skin Cancer Nevi and Moles','Nail Fungus and other Nail Disease','Poison Ivy Photos and other Contact Dermatitis','Psoriasis pictures Lichen Planus and related diseases','Scabies Lyme Disease and other Infestations and Bites','Seborrheic Keratoses and other Benign Tumors','Systemic Disease','Tinea Ringworm Candidiasis and other Fungal Infections','Urticaria Hives','Vascular Tumors','Vasculitis Photos','Warts Molluscum and other Viral Infections']# Replace with actual class names

    
    st.write(f"Predicted class: {class_names[np.argmax(prediction)]}")
    st.write(f"Prediction confidence: {np.max(prediction)*100:.2f}%")

