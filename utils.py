import requests
from datetime import datetime
from pymongo import MongoClient
import os
from ml.model import predict_image, make_gradcam_heatmap, overlay_heatmap_on_image
from pdf_generator import generate_pdf
import numpy as np
from PIL import Image
import io
import uuid
from evaluation import plot_confusion_matrix

# ✅ Connect to MongoDB (get URI from env var for security)
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["retina_ai"]  # database name
records_collection = db["patient_records"]  # collection name

# Lists for confusion matrix
y_true = []
y_pred_classes = []

# Define your class names from your model (must match training order)
class_names = ["Healthy", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

def get_location():
    """Fetch approximate location based on IP address."""
    try:
        ip_info = requests.get("https://ipinfo.io").json()
        loc = ip_info['loc'].split(",")
        latitude, longitude = loc[0], loc[1]
        return {"latitude": latitude, "longitude": longitude}
    except Exception as e:
        return {"error": "Failed to get location", "details": str(e)}

def save_patient_record(name, age, phone, email, disease, confidence):
    """Save a patient record to MongoDB."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "Timestamp": timestamp,
        "Name": name,
        "Age": age,
        "Phone": phone,
        "Email": email,
        "Disease": disease,
        "Confidence": confidence
    }
    result = records_collection.insert_one(record)
    return str(result.inserted_id)

def process_batch(image_files, patient_details_list, include_gradcam=False):
    """
    Process a batch of images and return prediction results.
    """
    results = []

    for file_obj, patient in zip(image_files, patient_details_list):
        # Read image
        if hasattr(file_obj, "read"):
            contents = file_obj.read()
        else:
            with open(file_obj, "rb") as f:
                contents = f.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Prediction
        selected, confidence, _ = predict_image(image)
        pred_index = class_names.index(selected)

        # Store for confusion matrix
        if "true_label" in patient:
            y_true.append(class_names.index(patient["true_label"]))
        y_pred_classes.append(pred_index)

        gradcam_image = None
        if include_gradcam:
            img_resized = image.resize((224, 224))
            img_array = np.asarray(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            heatmap = make_gradcam_heatmap(img_array, last_conv_layer_name="Conv_1")
            gradcam_image = overlay_heatmap_on_image(image, heatmap)

        # Save PDF report
        pdf_path = generate_pdf(
            patient["name"],
            patient["age"],
            image,
            [{"disease": selected, "confidence": float(confidence)}],
            lang=patient.get("lang", "English"),
            gradcam_image=gradcam_image
        )

        # Save patient record to MongoDB
        save_patient_record(
            patient["name"],
            patient["age"],
            patient.get("phone", ""),
            patient.get("email", ""),
            selected,
            float(confidence)
        )

        results.append({
            "patient": patient,
            "prediction": selected,
            "confidence": confidence,
            "pdf_path": pdf_path
        })

    # ✅ Generate confusion matrix after all images are processed
    if y_true and y_pred_classes:
        cm_path = plot_confusion_matrix(y_true, y_pred_classes, class_names)
        # You could also pass cm_path into each PDF if you want it inside the report

    return results
