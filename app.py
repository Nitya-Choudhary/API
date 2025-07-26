from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import io
import matplotlib.pyplot as plt
import base64

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download the model from Hugging Face (first-time only)
MODEL_URL = "https://huggingface.co/NityaIGDTUW28/Brain_Tumor_Detection/resolve/main/brain_tumor_model.h5"
MODEL_PATH = "brain_tumor_model.h5"

# Load model once
def load_model():
    if not tf.io.gfile.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Dummy labels — update with your actual class labels
labels = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))  # Resize to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_recommendation(pred_label):
    recs = {
        'No Tumor': 'Healthy brain. Continue regular checkups.',
        'Glioma': 'Consult an oncologist. Treatment may include surgery or chemotherapy.',
        'Meningioma': 'Consult a neurosurgeon. May require surgery or radiation.',
        'Pituitary': 'Endocrinologist and MRI follow-ups suggested.'
    }
    return recs.get(pred_label, 'No recommendation available.')

def generate_graph():
    plt.figure(figsize=(4, 3))
    tumor_types = ['Glioma', 'Meningioma', 'Pituitary']
    cases = [120, 80, 60]
    plt.bar(tumor_types, cases, color='skyblue')
    plt.title('Common Tumor Types')
    plt.xlabel('Tumor')
    plt.ylabel('Cases')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        pred_label = labels[class_index]
        confidence = float(np.max(prediction))

        recommendations = get_recommendation(pred_label)
        graph_base64 = generate_graph()

        return JSONResponse({
            "prediction": pred_label,
            "confidence": f"{confidence:.2%}",
            "recommendation": recommendations,
            "graph": graph_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
