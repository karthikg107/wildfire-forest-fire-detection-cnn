import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "forest_fire_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "🔥 Fire", 1: "✅ No Fire"}
CLASS_COLORS = {0: "#FF4B4B", 1: "#21c354"}

# ── Grad-CAM helper ──────────────────────────────────────────────────────────
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_np, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    overlayed = heatmap_colored * alpha + img_np / 255.0 * (1 - alpha)
    return np.clip(overlayed, 0, 1)

# ── Main prediction function ─────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Upload an image to get started.", {}, None

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch, verbose=0)[0][0]

    # Class: sigmoid output — 0=Fire, 1=No Fire
    predicted_class = int(prediction > 0.5)
    confidence = prediction if predicted_class == 1 else 1 - prediction
    label = CLASS_NAMES[predicted_class]
    color = CLASS_COLORS[predicted_class]

    # FIX: gr.Label expects values between 0 and 1, NOT percentages
    fire_conf = float(1 - prediction)      # e.g. 0.655
    nofire_conf = float(prediction)        # e.g. 0.345

    label_output = f"**Prediction: {label}**\n\nConfidence: {confidence * 100:.1f}%"

    scores = {
        "🔥 Fire": round(fire_conf, 4),
        "✅ No Fire": round(nofire_conf, 4),
    }

    # Grad-CAM
    try:
        last_conv = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv)
        img_np = np.array(img.resize(IMG_SIZE))  # use resized for cleaner display
        overlayed = overlay_gradcam(img_np, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor('#0f0f1a')

        for ax in axes:
            ax.set_facecolor('#0f0f1a')

        # Original
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image", color='white', fontsize=13,
                          fontweight='bold', pad=12)
        axes[0].axis('off')

        # Heatmap only
        axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
        axes[1].set_title("Grad-CAM Heatmap", color='white', fontsize=13,
                          fontweight='bold', pad=12)
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title("Overlay", color='white', fontsize=13,
                          fontweight='bold', pad=12)
        axes[2].axis('off')

        # Prediction label at bottom
        result_text = f"Prediction: {label}   |   Confidence: {confidence*100:.1f}%"
        fig.suptitle(result_text, fontsize=14, fontweight='bold',
                     color=color, y=0.04)

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.subplots_adjust(wspace=0.05)
        gradcam_fig = fig
        plt.close(fig)
    except Exception as e:
        gradcam_fig = None

    return label_output, scores, gradcam_fig

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
    title="Forest Fire Detection",
    css="""
        .result-box { border-radius: 12px; padding: 16px; }
        .gradcam-section { margin-top: 8px; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
# 🌲🔥 Forest Fire Detection System
Upload a forest or wildland image to classify it as **Fire** or **No Fire**
using a **MobileNetV2** transfer-learning model trained on the Wildfire Dataset.

The app also generates a **Grad-CAM heatmap** showing which image regions the model focused on to make its prediction.

---
**Model:** MobileNetV2 (frozen backbone, feature extraction) &nbsp;|&nbsp;
**Dataset:** The Wildfire Dataset — 2,699 images, 2 classes &nbsp;|&nbsp;
**Test Accuracy:** 89.02% &nbsp;|&nbsp; **AUC:** 0.957 &nbsp;|&nbsp;
**Project:** Pattern Recognition Course — UE Potsdam (2026)
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Forest Image",
                height=300
            )
            submit_btn = gr.Button(
                "🔍 Analyse Image",
                variant="primary",
                size="lg"
            )
            gr.Markdown("*Upload any forest image — fire, smoke, or clear forest scenes.*")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown(
                value="*Upload an image to get started.*",
                label="Result"
            )
            confidence_output = gr.Label(
                label="Class Confidence Scores",
                num_top_classes=2
            )

    gr.Markdown("---")
    gr.Markdown("### 🔬 Grad-CAM Explainability — What did the model look at?")

    gradcam_output = gr.Plot(
        label="Grad-CAM Visualization",
        show_label=False
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[prediction_output, confidence_output, gradcam_output]
    )

    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[prediction_output, confidence_output, gradcam_output]
    )

    gr.Markdown("""
---
**Student:** Satya Sai Karthik Guttula &nbsp;|&nbsp; **Supervisor:** Raja Hashim Ali &nbsp;|&nbsp;
**University of Europe for Applied Sciences, Potsdam**
""")

if __name__ == "__main__":
    demo.launch()
