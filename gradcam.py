import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, feature_model, classifier_head, last_conv_layer_name):
    """Generate a Grad-CAM heatmap."""
    last_conv_layer = feature_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [feature_model.inputs],
        [last_conv_layer.output, feature_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.4):
    """Overlay heatmap onto original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    img_with_heatmap = cv2.addWeighted(np.array(original_img), 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(img_with_heatmap)
