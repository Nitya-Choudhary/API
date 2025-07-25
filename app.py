
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import io
from fpdf import FPDF
import base64

st.set_page_config(page_title="NeuroTrace – Brain Tumor Detection", layout="centered")

# Download model
@st.cache_resource
def load_model():
    url = "https://huggingface.co/NityaIGDTUW28/Brain_Tumor_Detection/resolve/main/brain_tumor_model.h5"
    r = requests.get(url)
    with open("brain_tumor_model.h5", "wb") as f:
        f.write(r.content)
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])

# Dark/Light Mode toggle
mode = st.sidebar.radio("Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown('<style>body { background-color: #0e1117; color: white; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body { background-color: white; color: black; }</style>', unsafe_allow_html=True)

if page == "Home":
    st.title("🧠 NeuroTrace – Brain Tumor Detection")
    st.write("Upload an MRI image to detect tumor type.")

    uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB").resize((150, 150))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("Diagnose"):
            prediction = model.predict(img_array)[0]
            pred_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")

            # PDF Report Download
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="🧠 NeuroTrace Brain Tumor Diagnosis Report", ln=True)
            pdf.cell(200, 10, txt=f"Diagnosis: {pred_class}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)

            pdf_path = "/mnt/data/diagnosis_report.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="diagnosis_report.pdf">📄 Download Report</a>', unsafe_allow_html=True)

elif page == "About":
    st.title("About NeuroTrace")
    st.write("NeuroTrace is a brain tumor detection app using deep learning.")
    st.write("It classifies brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor.")

elif page == "Contact":
    st.title("Contact Us")
    st.write("**Name:** Nitya Choudhary")
    st.write("**Email:** nityapunia141@gmail.com")
    st.image("https://res.cloudinary.com/dx9m0jvtj/image/upload/v1753360412/4e45b59b-fc96-4871-8110-3964dfbf0bfe_vkes4w.jpg", use_column_width=True)
