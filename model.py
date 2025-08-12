import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model at startup
MODEL_PATH = os.path.join(BASE_DIR, "..", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "labels.txt")

model = load_model(MODEL_PATH)
# feature extractor & classifier head (as in your exported model)
feature_model = model.get_layer("sequential_1").get_layer("model1")
classifier_head = model.get_layer("sequential_3")

# load class names robustly
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [ln.strip() for ln in f.readlines() if ln.strip()]

# (Optional) disease info - expand with your real descriptions
disease_info = {
    "Cataract": {"desc": "Clouding of the lens.", "treat": "Surgery"},
    "Glaucoma": {"desc": "Optic nerve damage.", "treat": "Eye drops, surgery"},
}

def predict_image(pil_image):
    """
    Accept a PIL.Image (RGB), return (clean_class, confidence (float), info dict).
    """
    try:
        img = pil_image.resize((224, 224)).convert("RGB")
        img_array = np.asarray(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        preds = model.predict(img_array)
        index = int(np.argmax(preds[0]))
        class_name = class_names[index]
        confidence = float(preds[0][index])

        # Clean label formatting (matches your earlier code)
        raw_class = class_name
        clean_class = ' '.join(raw_class.split(' ')[1:]).replace('_', ' ').strip().title()
        info = disease_info.get(clean_class, {"desc": "No info available", "treat": "N/A"})

        return clean_class, confidence, info

    except Exception as e:
        # Return fallback values on error
        return "Unknown", 0.0, {"desc": f"Error during prediction: {str(e)}", "treat": "N/A"}


def make_gradcam_heatmap(img_array, last_conv_layer_name="Conv_1", pred_index=None):
    """
    img_array: numpy array shaped (1, H, W, C) scaled [0,1]
    Returns: heatmap numpy array (H_small, W_small) normalized 0..1
    """
    # Build grad model from the globally loaded feature_model
    grad_model = tf.keras.models.Model(
        inputs=feature_model.input,
        outputs=[
            feature_model.get_layer(last_conv_layer_name).output,
            feature_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, conv_features = grad_model(img_array)
        tape.watch(conv_outputs)
        # Global average pooling on conv_outputs -> features for classifier head
        gap_features = tf.reduce_mean(conv_outputs, axis=[1, 2])  # (1, channels)
        predictions = classifier_head(gap_features)  # classifier head on pooled features

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)  # grads w.r.t conv outputs
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H', W', C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # matrix multiply
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.4):
    """
    img_pil: PIL.Image (original size)
    heatmap: small 2D numpy array (values 0..1)
    returns: PIL.Image blended
    """
    # resize heatmap to image size
    heatmap_uint8 = np.uint8(255 * heatmap)
    # create colored heatmap using matplotlib colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]  # (256,3) values 0..1
    # index heatmap values (0..255)
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    jet_pil = Image.fromarray(jet_heatmap).resize(img_pil.size)

    blended = Image.blend(img_pil.convert("RGB"), jet_pil.convert("RGB"), alpha=alpha)
    return blended
