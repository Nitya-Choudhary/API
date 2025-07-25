
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download the model from Hugging Face (once)
MODEL_URL = "https://huggingface.co/NityaIGDTUW28/Brain_Tumor_Detection/resolve/main/brain_tumor_model.h5"
MODEL_PATH = "brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocessing function
def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Detection API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img_array = preprocess_image(image_data)
    prediction = model.predict(img_array)[0][0]
    result = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    # Sample static response logic
    recommendation = "Visit neurologist immediately." if result == "Tumor Detected" else "Continue healthy habits."
    medicine = "Temozolomide, Bevacizumab" if result == "Tumor Detected" else "No medicine required."

    return {
        "result": result,
        "confidence": f"{confidence:.2%}",
        "recommendation": recommendation,
        "medicine": medicine,
        "graph_url": "https://www.cdc.gov/cancer/images/bts/data-trends/brain-graph.png"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
