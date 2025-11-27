# generate_gradcam.py
import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from keras.models import load_model

# Paths (adjust if needed)
MODEL_PATH = "../models/final_deepfake_model.h5"
VIDEO_PATH = "../test/trump.mp4"
OUT_HEATMAP = "gradcam_heatmap.jpg"
OUT_OVERLAY = "gradcam_overlay.jpg"

def find_last_conv_layer(model):
    # Find last layer with 4D output (HxWxC) and class name containing 'Conv' or 'DepthwiseConv2D'
    for layer in reversed(model.layers):
        # Some conv-like layer class names: Conv2D, DepthwiseConv2D, SeparableConv2D
        cls = layer.__class__.__name__.lower()
        if "conv" in cls or "depthwise" in cls or "separable" in cls:
            # Ensure layer output is at least 3D spatial (exclude Dense/Flatten)
            try:
                shape = layer.output.shape
            except Exception:
                shape = None
            if shape is not None and len(shape) >= 3:
                return layer.name
    raise ValueError("No convolutional layer found in model.")

def extract_first_face_from_video(video_path):
    print("Extracting first face from video:", video_path)
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    face_img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # mtcnn expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        if len(results) > 0:
            x, y, w, h = results[0]['box']
            x = max(0, x); y = max(0, y)
            x2 = min(x + w, rgb.shape[1]); y2 = min(y + h, rgb.shape[0])
            face = rgb[y:y2, x:x2]
            if face.size == 0:
                continue
            face_img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)  # back to BGR for saving/overlay
            break
    cap.release()
    if face_img is None:
        raise FileNotFoundError("No face found in video.")
    return face_img

def preprocess_face_for_model(face_bgr, target_size):
    # face_bgr: BGR image
    face = cv2.resize(face_bgr, target_size)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    arr = face_rgb.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def make_gradcam_heatmap(model, img_tensor, last_conv_layer_name, pred_index=None):
    """
    model: keras model
    img_tensor: (1, H, W, 3) preprocessed input
    last_conv_layer_name: string name of last conv layer
    pred_index: class index to inspect; if None uses model's argmax or single-output
    returns heatmap (H, W) normalized 0..1
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if isinstance(predictions, (list, tuple)):
            preds = predictions[0]
        else:
            preds = predictions
        if pred_index is None:
            # For binary single-output model, use predictions[0][0]
            # If multi-class, pick argmax
            if preds.shape[-1] == 1:
                pred_index = 0
            else:
                pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    # grads w.r.t. conv outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute gradients (got None).")

    # compute channel-wise mean of gradients over spatial axes (H, W)
    # conv_outputs shape: (1, H, W, C)
    # grads shape matches conv_outputs
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)

    conv_outputs = conv_outputs[0]  # (H, W, C)
    pooled_grads = pooled_grads.numpy()

    conv_outputs_np = conv_outputs.numpy()  # (H, W, C)
    # weight the channels by gradient
    for i in range(pooled_grads.shape[-1]):
        conv_outputs_np[:, :, i] *= pooled_grads[i]

    heatmap = np.sum(conv_outputs_np, axis=-1)

    # ReLU
    heatmap = np.maximum(heatmap, 0)

    # Normalize
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap = heatmap / np.max(heatmap)
    return heatmap

def save_heatmap_and_overlay(face_bgr, heatmap, out_heatmap_path, out_overlay_path):
    # face_bgr: original face image (BGR)
    h, w = face_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Save raw heatmap (color)
    cv2.imwrite(out_heatmap_path, heatmap_color)
    print("Saved heatmap to", out_heatmap_path)

    # Overlay: 0.4 heatmap + 0.6 original
    overlay = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(out_overlay_path, overlay)
    print("Saved overlay to", out_overlay_path)

def main():
    print("Loading model:", MODEL_PATH)
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    try:
        last_conv = find_last_conv_layer(model)
        print("Last conv layer detected:", last_conv)
    except Exception as e:
        print("Error finding conv layer:", e)
        return

    # extract face
    try:
        face_bgr = extract_first_face_from_video(VIDEO_PATH)
    except Exception as e:
        print("Error extracting face from video:", e)
        return

    # target size from model input (assume (None, H, W, C))
    try:
        input_shape = model.input_shape
        if input_shape is None:
            raise ValueError("Model input_shape is None")
        # Some models set (None, H, W, C)
        if len(input_shape) == 4:
            _, H, W, C = input_shape
            target_size = (int(W), int(H))
        else:
            # fallback
            target_size = (224, 224)
    except Exception:
        target_size = (224, 224)

    print("Using input size:", target_size)

    img_tensor = preprocess_face_for_model(face_bgr, target_size)

    # build gradcam
    try:
        heatmap = make_gradcam_heatmap(model, img_tensor, last_conv)
    except Exception as e:
        print("Grad-CAM generation failed:", e)
        return

    # save images
    save_heatmap_and_overlay(face_bgr, heatmap, OUT_HEATMAP, OUT_OVERLAY)
    print("Grad-CAM complete.")

if __name__ == "__main__":
    main()
