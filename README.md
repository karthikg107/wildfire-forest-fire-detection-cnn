# CNN Based Forest Fire Detection

## Project Overview

This project uses a Convolutional Neural Network (CNN) to automatically detect the presence of forest fires from images.

The model is trained on wildfire image data and classifies images into two categories:

- Fire
- No Fire

The objective of this project is to assist in early forest fire detection using deep learning techniques.

---

## Features

- Image classification using CNN
- Binary classification (Fire / No Fire)
- Data augmentation for improved generalization
- Model evaluation using classification metrics
- Confusion matrix visualization
- Accuracy and loss curve analysis
- Trained model generation (.h5)

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Kaggle Notebook Environment

---

## Dataset

The project uses the Wildfire Dataset available on Kaggle.

Classes:

1. Fire
2. No Fire

The dataset is divided into:

- Training Set
- Validation Set
- Test Set

---

## Project Structure

```
CNN-Based-Forest-Fire-Detection
│
├── cnn-based-forest-fire-detection.ipynb
├── accuracy_loss_curve.pdf
├── confusion_matrix.pdf
├── README.md
```

---

## Data Preprocessing

The following preprocessing steps were applied:

- Image resizing
- Image normalization
- Data augmentation
  - Rotation
  - Zoom
  - Horizontal Flip
  - Width Shift
  - Height Shift

---

## CNN Architecture

The model consists of:

```
Input Layer

Conv2D (32 Filters, ReLU)
MaxPooling2D

Conv2D (64 Filters, ReLU)
MaxPooling2D

Conv2D (128 Filters, ReLU)
MaxPooling2D

Flatten Layer

Dense (128 Units, ReLU)

Dropout

Output Layer
Dense (1 Unit, Sigmoid)
```

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metrics | Accuracy |
| Batch Size | 32 |
| Epochs | 10 |
| Activation Function | ReLU, Sigmoid |

---

## Model Evaluation

The trained model was evaluated on the test dataset using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Classification Report

| Class | Precision | Recall | F1-Score |
|---------|-----------|---------|----------|
| Fire | 0.79 | 0.75 | 0.77 |
| No Fire | 0.85 | 0.87 | 0.86 |

### Overall Accuracy

**82%**

---

## Results

The model successfully learned to distinguish between fire and non-fire images.

### Generated Outputs

- Classification Report
- Confusion Matrix
- Accuracy Curve
- Loss Curve
- Trained CNN Model

---

## Output Files

### Confusion Matrix

File:

```
confusion_matrix.pdf
```

### Accuracy and Loss Curves

File:

```
accuracy_loss_curve.pdf
```

---

## Trained Model

The trained model was saved as:

```
forest_fire_cnn.h5
```

**Note:** The model file is not included in this repository because GitHub upload size limitations prevent uploading large model files.

---

## Future Improvements

- Use Transfer Learning (ResNet50, EfficientNet)
- Increase dataset size
- Real-time fire detection from video streams
- Deploy as a web application
- Improve model accuracy using hyperparameter tuning

---

## Author

**Karthik Guttula**

Forest Fire Detection using Deep Learning and Convolutional Neural Networks.
