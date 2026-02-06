# Multiclass-Fish-Image-Classification
This project focuses on building a multiclass image classification system to identify different species of fish using Deep Learning.
Both a custom CNN model and multiple pre-trained transfer learning models are implemented, evaluated, and compared.
The best-performing model is deployed using a Streamlit web application for real-time predictions.

# Problem Statement

Manual fish species identification is time-consuming and prone to errors.
This project aims to automate fish classification using image data, improving accuracy and efficiency through deep learning techniques.

# Solution Approach

The solution follows an end-to-end deep learning pipeline:

# Data preprocessing and augmentation

Model training (CNN + Transfer Learning)

Model evaluation and comparison

Model saving and deployment

User-friendly Streamlit interface

# Technologies Used

Programming Language: Python

Deep Learning: TensorFlow, Keras

# Models Used:

Custom CNN

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

# Web Framework: Streamlit

Visualization: Matplotlib, Seaborn

Version Control: Git & GitHub

# Dataset Description

The dataset consists of fish images categorized into folders by species.

Each folder name represents a class label.

The dataset is loaded using TensorFlow’s ImageDataGenerator for efficient preprocessing.

# Preprocessing Techniques:

Image rescaling (0–1)

Data augmentation (rotation, zoom, horizontal flip)

Model Training

Images are resized and normalized.

CNN model trained from scratch.

Transfer learning models are fine-tuned on the fish dataset.

Adam optimizer and categorical cross-entropy loss are used.

Validation accuracy and loss are monitored during training.

# Model Evaluation

The models are evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Training and validation performance is visualized using accuracy and loss curves.

# Deployment (Streamlit App)

The Streamlit application allows users to:

Upload a fish image

Predict the fish species

View prediction confidence scores

# How it works:

User uploads an image

Image is preprocessed and resized

Trained model predicts the class

Result is displayed on the UI
