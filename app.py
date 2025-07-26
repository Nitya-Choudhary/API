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

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update to specific domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face model link
MODEL_URL = "https://huggingface.co/NityaIGDTUW28/Brain_Tumor_Detection/resolve/main/brain_tumor_model.h5"
MODEL_PATH = "brain_tumor_model.h5"

# Download model if not already
def load_model():
    if not tf.io.gfile.exists(MODEL_PATH):
        print("📥 Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Dummy labels (replace with real ones if different)
labels = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

# Process image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Recommendations
def get_recommendation(label):
    return {
        "No Tumor": "You appear healthy. Continue regular checkups.",
        "Glioma": "Consult an oncologist immediately. Consider surgery or chemo.",
        "Meningioma": "Consult a neurosurgeon. MRI monitoring recommended.",
        "Pituitary": "Consult an endocrinologist. Hormonal evaluation may be needed."
    }.get(label, "No recommendation available.")

# Generate bar chart graph
def generate_graph():
    plt.figure(figsize=(4, 3))
    tumor_types = ['Glioma', 'Meningioma', 'Pituitary']
    case_counts = [120, 90, 70]
    plt.bar(tumor_types, case_counts, color='skyblue')
    plt.title('Tumor Type Distribution')
    plt.xlabel('Tumor')
    plt.ylabel('Cases')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor)
        class_idx = np.argmax(preds)
        label = labels[class_idx]
        confidence = float(np.max(preds))
        recommendation = get_recommendation(label)
        graph = generate_graph()

        return JSONResponse({
            "prediction": label,
            "confidence": f"{confidence:.2%}",
            "recommendation": recommendation,
            "graph": graph
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Root for health check
@app.get("/")
def read_root():
    return {"message": "Brain Tumor Detection API is live!"}
