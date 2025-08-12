from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------
# 📦 Load Model and Labels
# ----------------------------
model = load_model("keras_model.h5")

# Extract MobileNetV2 feature extractor
feature_model = model.get_layer("sequential_1").get_layer("model1")

# Extract classifier head
classifier_head = model.get_layer("sequential_3")

# This is the last conv layer in MobileNetV2
last_conv_layer_name = "Conv_1"

# Load class names
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ----------------------------
# 🧠 Prediction Function
# ----------------------------
def predict_image(image_path: str):
    """Run model prediction on a retina image."""
    image = Image.open(image_path).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index])
    
    return class_name, confidence
