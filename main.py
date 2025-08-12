from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pymongo import MongoClient
from PIL import Image
import numpy as np
import io
import os
import uuid

from model import predict_image, make_gradcam_heatmap, overlay_heatmap_on_image
from utils.pdf_generator import generate_pdf
from utils import save_patient_record  # MongoDB insert function

app = FastAPI(title="Eye Disease Detection API")

MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["eyecaredb"]

TEMP_DIR = "tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Eye Disease Detection API is running"}

@app.post("/predict")
async def predict(
    file: UploadFile,
    patient_name: str = Form(...),
    patient_age: str = Form(...),
    phone: str = Form(""),
    email: str = Form(""),
    lang: str = Form("English"),
    include_gradcam: bool = Form(False)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    selected, confidence, info = predict_image(image)

    gradcam_image = None
    gradcam_temp_path = None
    if include_gradcam:
        img_resized = image.resize((224, 224))
        img_array = np.asarray(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        heatmap = make_gradcam_heatmap(img_array, last_conv_layer_name="Conv_1")
        gradcam_image = overlay_heatmap_on_image(image, heatmap)

        gradcam_temp_path = os.path.join(TEMP_DIR, f"gradcam_{uuid.uuid4().hex}.jpg")
        gradcam_image.save(gradcam_temp_path)

    predictions = [{"disease": selected, "confidence": float(confidence)}]
    pdf_path = generate_pdf(
        patient_name,
        patient_age,
        image,
        predictions,
        lang=lang,
        gradcam_image=gradcam_image
    )

    try:
        save_patient_record(patient_name, patient_age, phone, email, selected, float(confidence))
    except Exception as e:
        print("Warning: Could not save record to MongoDB:", e)

    if gradcam_temp_path and os.path.exists(gradcam_temp_path):
        os.remove(gradcam_temp_path)

    return FileResponse(pdf_path, filename="Eye_Report.pdf")
