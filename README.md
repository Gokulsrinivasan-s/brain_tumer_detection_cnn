# brain_tumer_detection_cnn

Brain Tumor MRI Classification using ResNet50
Project Overview

This project builds a Deep Learning model to classify brain MRI images into 4 classes:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

Healthy Brain

The model uses Transfer Learning with ResNet50 and achieves high accuracy on the dataset.

### 📁 Dataset

Structure:

Training/
    glioma/
    meningioma/
    pituitary/
    healthy/
Testing/
    glioma/
    meningioma/
    pituitary/
    healthy/

### Requirements

Python 3.9+

TensorFlow 2.x

KaggleHub (pip install kagglehub)

Numpy

OS and Shutil (standard Python libraries)

### Install packages:

pip install tensorflow kagglehub numpy

### Data Loading

Images are automatically loaded from the Training and Testing folders using:

tf.keras.utils.image_dataset_from_directory()


Images are resized to 224x224 pixels for ResNet50.

Labels are one-hot encoded for 4 classes.

### Model Architecture

Base Model: ResNet50 pretrained on ImageNet (include_top=False)

### Layers Added:

GlobalAveragePooling2D

Dropout(0.5)

Dense(256, ReLU)

Dropout(0.3)

Dense(4, Softmax)

Optimizer: Adam (learning_rate=0.0001)

Loss Function: Categorical Crossentropy

Metrics: Accuracy

### Training

Epochs: 20

EarlyStopping: patience=5, restores best weights

ReduceLROnPlateau: factor=0.5, patience=3

Batch size: 32

### Model Saving

The trained model is saved as:

brain_tumor_resnet50.keras


It can be loaded later for predictions using:

model = tf.keras.models.load_model("brain_tumor_resnet50.keras")

### Usage
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("brain_tumor_resnet50.keras")

# Predict a new image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

img = load_img("new_brain_mri.jpg", target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = np.argmax(pred)
print(f"Predicted class: {model.class_names[pred_class]}")
