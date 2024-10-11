# Plant-Disease-Detection-Using-CNN
<br>

This repository contains a project focused on detecting plant diseases using deep learning techniques, specifically Convolutional Neural Networks (CNN). The model is trained on images of plant leaves and aims to classify them into healthy or diseased categories.
<br>

## Table of Contents
<br>
1. Overview
<br>
2. Methodology
<br>
3. Results
<br>
4. Technologies Used
<br>
5. Setup
<br>
6. Usage
<br>
7. Future Work
<br>
8. Contributors
<br>
9. License
<br>

## Overview
<br>

Plant diseases can cause significant damage to agriculture, leading to reduced crop yield and food security concerns. Traditional methods of disease detection are time-consuming and require expert knowledge. This project implements a CNN-based model for automatic and efficient detection of plant diseases from leaf images.
<br>

## Abstract
<br>
The model was trained on a dataset of healthy and diseased plant leaves. It utilizes a CNN architecture and achieves over 90% accuracy on the validation and test datasets. The approach presented here demonstrates the effectiveness of deep learning for disease detection in plants, offering a valuable tool for agricultural diagnostics.
<br>

## Methodology
<br>

### Dataset
<br>
The dataset comprises images of plant leaves, both healthy and diseased, from various species. An 85-15% split was used for training and testing, with further splitting for validation. Data augmentation techniques such as resizing, standardization, and random shuffling were employed to improve model generalization.
<br>

### Model Architecture
<br>

The CNN model consists of:
<br>

- Four convolutional layers with ReLU activation and batch normalization
  <br>
- Max-pooling layers
  <br>
- Fully connected layers for classification
  <br>
- Dropout layers to prevent overfitting
  <br>
- The model was trained using the Adam optimizer and cross-entropy loss function. The training was conducted over 5 epochs, with performance tracked via accuracy and loss metrics on training, validation, and test sets.
  <br>

## Results
<br>

- Train Accuracy: 96.91%
<br>

- Validation Accuracy: 92.70%
<br>

- Test Accuracy: 92.08%
<br>

- While the model achieved good accuracy, there was a slight indication of overfitting in the later epochs. A confusion matrix and loss curves show that the model performed well overall but had difficulty distinguishing between certain similar classes of diseases.
<br>

For more detailed analysis, please refer to the Result Analysis section of the report.
<br>

### Technologies Used
<br>

- Python
  <br>
- PyTorch for model building and training
  <br>
- Jupyter Notebook for experimentation
  <br>
- Google Colab for cloud-based GPU training
  <br>
- Pandas, NumPy, and Matplotlib for data handling and visualization
  <br>
  
## Setup
<br>

### Clone the repository:
<br>

git clone https://github.com/yourusername/plant-disease-detection.git
<br>

### Install dependencies:
<br>

pip install -r requirements.txt
<br>

### Download the dataset: 
<br>

The dataset used can be found at Mendeley Data. Download the images and place them in the data/ directory.
<br>

### Run the training notebook:
<br>

jupyter notebook Plant_Disease_Detection.ipynb
<br>

### Usage
<br>

The project is set up to allow users to train the model on a dataset of plant leaves or use a pre-trained model to classify new images. To classify new images, run the provided predict.py script or use the trained model provided in the notebook.
<br>

## Future Work
<br>

- Improved Generalization: Exploring advanced data augmentation techniques to improve model robustness.
<br>
- Class Imbalance: Addressing the class imbalance in the dataset to enhance accuracy.
<br>
- Transfer Learning: Experimenting with transfer learning from models trained on large
