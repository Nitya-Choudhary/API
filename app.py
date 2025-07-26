import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Step 1: Download model from Hugging Face
MODEL_URL = "https://huggingface.co/NityaIGDTUW28/Brain_Tumor_Detection/resolve/main/brain_tumor_model.h5"
MODEL_PATH = "brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("Model downloaded!")

# Step 2: Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Tumor classes
tumor_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Step 3: Prediction function
def predict_tumor(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    predicted_class = tumor_classes[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    recommendations = {
        "Glioma": "Consult a neuro-oncologist immediately.",
        "Meningioma": "Get a biopsy and follow up with a specialist.",
        "No Tumor": "No tumor detected, but consult a doctor if symptoms persist.",
        "Pituitary": "Schedule an MRI for further diagnosis and meet an endocrinologist."
    }

    return predicted_class, f"{confidence:.2f}%", r_
