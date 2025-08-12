from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
from predict import predict_image, model, feature_model, classifier_head, last_conv_layer_name
from gradcam import make_gradcam_heatmap, overlay_heatmap_on_image
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Get the Mongo URI from .env
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["eyecaredb"]

# Create FastAPI app
app = FastAPI()

# Test route
@app.get("/")
def home():
    return {"message": "MongoDB connection successful!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    class_name, confidence = predict_image(temp_path)

    # Prepare image for Grad-CAM
    image = Image.open(temp_path).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(
        img_array,
        feature_model=feature_model,
        classifier_head=classifier_head,
        last_conv_layer_name=last_conv_layer_name
    )

    gradcam_img = overlay_heatmap_on_image(image, heatmap)
    gradcam_path = temp_path.replace(".jpg", "_gradcam.jpg")
    gradcam_img.save(gradcam_path)

    # Remove original temp image
    os.remove(temp_path)

    return {
        "disease": class_name,
        "confidence": confidence,
        "gradcam_image": gradcam_path
    }
