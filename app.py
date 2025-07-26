import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define tumor classes (modify according to your model)
tumor_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Prediction function
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

    return {
        "Prediction": predicted_class,
        "Confidence": f"{confidence:.2f}%",
        "Recommendation": recommendations[predicted_class]
    }

# Gradio UI
interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Tumor Type"),
        gr.Text(label="Confidence"),
        gr.Text(label="Recommendation")
    ],
    title="Brain Tumor Detection",
    description="Upload an MRI image to detect brain tumor type"
)

interface.launch()
