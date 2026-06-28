# 🌲🔥 Forest Fire Detection using CNN

A comparative study of three CNN architectures for binary forest fire image
classification, with explainability (Grad-CAM + SHAP) and robustness analysis.

**Course:** Pattern Recognition — M.Sc. Software Engineering  
**University:** University of Europe for Applied Sciences, Potsdam  
**Student:** Satya Sai Karthik Guttula  
**Supervisor:** Raja Hashim Ali  
**Year:** 2026

---

## 🔗 Live Demo

👉 **[Try the app on HuggingFace](https://huggingface.co/spaces/karthikg10/forest-fire-detection)**

Upload any forest image and get:
- Fire 🔥 or No Fire ✅ prediction with confidence score
- Grad-CAM heatmap showing what the model focused on

---

## 📊 Results Summary

| Model | Accuracy | AUC | Parameters | Size |
|---|---|---|---|---|
| Custom CNN | 80.73% | 0.898 | 11.2M | 127.87 MB |
| **MobileNetV2** | **89.02%** | **0.957** | **2.4M** | **11.05 MB** |
| ResNet50 | 62.93% | 0.725 | 23.9M | 93.61 MB |

**Winner: MobileNetV2** — best accuracy with smallest model size.

> **Note:** ResNet50 underperformed due to a preprocessing mismatch.
> Generic 1/255 rescaling was used instead of
> `tf.keras.applications.resnet50.preprocess_input()`.
> This is a key transfer learning lesson — always use the correct
> preprocessing for each pretrained model.

---

## 🗂️ Dataset

**The Wildfire Dataset** by Elmadafri et al. (Kaggle, 2023)

- 2,699 RGB images — Fire / No Fire
- Split: Train 1,887 / Val 402 / Test 410
- 📥 [Download from Kaggle](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)

---

## 🧠 Models

### 1. Custom CNN (from scratch)
- 3 Conv blocks: 32 → 64 → 128 filters
- BatchNorm + MaxPool + Dropout
- Dense 128 → Sigmoid output
- Trained for 10 epochs

### 2. MobileNetV2 (Transfer Learning)
- ImageNet pretrained backbone (frozen)
- Depthwise separable convolutions
- Custom classification head
- Trained for 5 epochs

### 3. ResNet50 (Transfer Learning)
- ImageNet pretrained backbone (frozen)
- 50-layer residual network
- Custom classification head
- Trained for 5 epochs

---

## 🔍 Explainability

### Grad-CAM
Heatmaps showing which image regions the model focused on:
- **MobileNetV2** → correctly focused on flame and smoke regions ✅
- **Custom CNN** → broader, less focused activations
- **ResNet50** → inconsistent maps due to preprocessing mismatch

### SHAP
Pixel-level contribution scores (positive = supports fire prediction):
- MobileNetV2 showed strong positive contributions from flame areas
- Validated that predictions are based on fire-relevant features

---

## 🧪 Robustness Testing (MobileNetV2)

| Perturbation | Severity | Accuracy |
|---|---|---|
| None (baseline) | — | 89.02% |
| Gaussian noise | σ=0.05 | 86.59% |
| Gaussian noise | σ=0.10 | 71.22% |
| Gaussian blur | kernel=5 | 78.54% |
| Gaussian blur | kernel=9 | 73.90% |
| Brightness reduction | 0.7 | 89.76% |
| Brightness reduction | 0.4 | 88.54% |

**Key finding:** Model is robust to brightness changes but sensitive to
high-severity Gaussian noise.

---

## 📁 Repository Structure

```
wildfire-forest-fire-detection-cnn/
│
├── cnn-based-forest-fire-detection.ipynb  ← Main Kaggle notebook
├── app.py                                  ← HuggingFace Gradio app
├── requirements.txt                        ← App dependencies
├── README.md                               ← This file
│
├── Figures/
│   ├── dataset_samples.pdf
│   ├── accuracy_loss_curve.pdf
│   ├── cnn_confusion_matrix.pdf
│   ├── mobilenet_confusion_matrix.pdf
│   ├── resnet_confusion_matrix.pdf
│   ├── gradcam_all.pdf          ← Custom CNN Grad-CAM
│   ├── gradcam_all_1.pdf        ← MobileNetV2 Grad-CAM
│   ├── gradcam_all_2.pdf        ← ResNet50 Grad-CAM
│   ├── shap_all.pdf             ← Custom CNN SHAP
│   ├── shap_all_1.pdf           ← MobileNetV2 SHAP
│   └── robustness_plot.pdf
│
└── Results/
    ├── model_comparison.csv
    └── robustness_results.csv
```

---

## ⚙️ How to Run

### Option 1 — Kaggle Notebook (recommended)
Open directly on Kaggle — no setup needed:  
👉 [kaggle.com/code/karthikguttula10/cnn-based-forest-fire-detection](https://www.kaggle.com/code/karthikguttula10/cnn-based-forest-fire-detection)

### Option 2 — Run locally
```bash
# Install dependencies
pip install tensorflow numpy opencv-python matplotlib scikit-learn shap

# Run notebook
jupyter notebook cnn-based-forest-fire-detection.ipynb
```

---

## 🛠️ Technologies

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- OpenCV
- Scikit-learn
- SHAP
- Gradio (frontend)

---

## 📄 Report

Full academic report written in Elsevier article format (LaTeX/Overleaf):  
👉 [View/Edit on Overleaf](https://www.overleaf.com/1747325196zfmfppjxmmvk#e87e9a)

---

## 📬 Contact

**Satya Sai Karthik Guttula**  
M.Sc. Software Engineering  
University of Europe for Applied Sciences, Potsdam
