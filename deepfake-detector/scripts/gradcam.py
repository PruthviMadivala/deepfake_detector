import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

VIDEO_PATH = "../test/trump.mp4"
MODEL_PATH = "../models/final_deepfake_model.h5"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

# Last conv layer
LAST_CONV_LAYER = "Conv_1"

# Read first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Could not read video frame.")

original_frame = frame.copy()
h, w = frame.shape[:2]

# Resize for model (224x224)
img = cv2.resize(frame, (224,224))
img_norm = img.astype("float32") / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# Grad-CAM
grad_model = tf.keras.models.Model(
    [model.inputs], 
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(img_input)
    loss = preds[:, 0]  

grads = tape.gradient(loss, conv_out)[0]
weights = tf.reduce_mean(grads, axis=(0, 1))

cam = np.zeros(conv_out.shape[1:3], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * conv_out[0, :, :, i]

cam = cv2.resize(cam.numpy(), (w, h))
cam = np.maximum(cam, 0)
cam = cam / cam.max()

# Convert to heatmap
heatmap = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay
overlay = cv2.addWeighted(original_frame, 0.6, heatmap, 0.4, 0)

# Save images
cv2.imwrite("gradcam_heatmap_full.jpg", heatmap)
cv2.imwrite("gradcam_overlay_full.jpg", overlay)

print("Saved full-frame heatmap and overlay:")
print(" - gradcam_heatmap_full.jpg")
print(" - gradcam_overlay_full.jpg")
