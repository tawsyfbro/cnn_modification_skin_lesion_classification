# Skin Lesion Classifier: Convolutional Neural Network for Dermatological Diagnosis

This repository contains a Jupyter Notebook that demonstrates the development of a Convolutional Neural Network (CNN) model for the classification of skin lesions. The model is trained and optimized to accurately identify different types of skin conditions using the HAM10000 dataset.

---

## Overview
Skin lesion classification is an important task in the field of dermatology, as it can aid in the early detection and diagnosis of various skin conditions, including melanoma, a potentially fatal form of skin cancer. In this project, we explore the use of deep learning techniques, specifically a CNN model, to automate the skin lesion classification process.

---

##NOTE

The original dataset was not uploaded to GitHUb due to Computational and Storage Limitations. 


## Key Features
- **Dataset Preprocessing and Augmentation**: The code includes steps to sort the HAM10000 dataset by diagnosis, resize the images, and apply data augmentation techniques to enhance the model's performance.
- **CNN Model Development**: The notebook outlines the construction of a CNN model with multiple convolutional, pooling, and dense layers, along with techniques to mitigate overfitting, such as batch normalization, dropout, and regularization.
- **Model Training and Evaluation**: The code demonstrates the training process of the CNN model, including the use of the Adam optimizer, categorical cross-entropy loss, and accuracy metric. The model's performance is evaluated using metrics like precision, recall, and F1-score.
- **Visualization**: The notebook includes a function to plot the training and validation accuracy and loss curves, providing insights into the model's learning process.

---

## Usage
To use this code, you will need to have the following requirements installed:

- Python 3.x
- NumPy
- OpenCV
- Scikit-learn
- TensorFlow
- Keras
- Pandas
- Matplotlib


