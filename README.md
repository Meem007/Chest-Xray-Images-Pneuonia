# Chest-Xray-Images-Pneuonia
This repository contains code for detecting pneumonia in chest X-ray images using two different deep learning approaches: Convolutional Neural Network (CNN) and ResNet-50. The models are trained, evaluated, and compared for their performance on the task of pneumonia detection.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Loading](#data-loading)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Comparison](#comparison)
- [Conclusion](#conclusion)

## Introduction

Pneumonia is a common respiratory infection that can be diagnosed using chest X-ray images. This project aims to develop and compare two different deep learning models for detecting pneumonia in chest X-ray images: a Convolutional Neural Network (CNN) and the ResNet-50 architecture.

## Dependencies

Make sure you have the following libraries installed:

- Python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- TensorFlow
- Keras
- OpenCV
- skimage

You can install these libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python scikit-image
```

## Data Loading

The dataset consists of chest X-ray images categorized into different folders for training, validation, and testing. The code loads the images, preprocesses them, and organizes them into dataframes.

## Data Preprocessing

Images are resized to a consistent size of 224x224 pixels and normalized to values between 0 and 1. Labels are encoded numerically using label encoding, and one-hot encoding is applied to the labels for model training.

## Model Building

Two models are built: a CNN and a ResNet-50 architecture. The CNN model consists of several convolutional and pooling layers, followed by fully connected layers. The ResNet-50 model uses a pre-trained ResNet-50 base with added dense layers.

## Model Training

The models are trained using the training data and validated using the validation data. Training performance is monitored for each epoch, and the best performing epoch is selected based on validation loss/accuracy.

## Model Evaluation

The models are evaluated on the test dataset, and their accuracy is reported. Additionally, confusion matrices are generated to visualize the model's performance.

## Comparison

The performance of the CNN and ResNet-50 models is compared based on their test accuracy. The CNN model achieves higher accuracy (71%) compared to the ResNet-50 model (67%).

## Conclusion

This repository demonstrates the implementation and comparison of two different deep learning models for detecting pneumonia in chest X-ray images. The CNN model outperforms the ResNet-50 model in terms of accuracy for pneumonia detection. The comparison results provide insights into the effectiveness of different deep learning approaches for medical image classification tasks.
