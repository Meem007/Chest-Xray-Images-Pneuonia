# Chest X-ray Images Pneumonia 

This repository contains code for analyzing the Chest X-Ray Images of Pneumonia dataset sourced from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The dataset consists of various X-ray images of chest radiographs, including cases with pneumonia.

## Table of Contents

1. **Introduction**
2. **Dependencies**
3. **Data Loading**
4. **Data Preprocessing**
5. **Model Building and Training**
6. **Model Evaluation**
7. **Comparison of CNN and ResNet-50**
8. **Conclusion**

## 1. Introduction

This repository contains Python code for performing a comprehensive analysis of the Chest X-Ray Images of Pneumonia dataset. The main goals of this analysis are:

- Load and preprocess the dataset.
- Build and train a Convolutional Neural Network (CNN) model.
- Evaluate the CNN model on validation and test datasets.
- Compare the performance of the CNN model with a ResNet-50 model.

## 2. Dependencies

The following libraries are used for this analysis:

- Python
- Numpy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn
- TensorFlow

## 3. Data Loading

The dataset is divided into three main categories: training, validation, and test sets. Images are loaded using OpenCV, converted from BGR to RGB, and resized to a consistent size (224x224).

## 4. Data Preprocessing

- Images are normalized to have pixel values in the range [0, 1].
- Labels are converted to categorical format.
- The data is split into training, validation, and test sets.

## 5. Model Building and Training

A CNN model is constructed with several convolutional and pooling layers, followed by fully connected layers. The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is trained on the training set and validated on the validation set for a specified number of epochs.

## 6. Model Evaluation

The model's performance is evaluated on the test set, and accuracy metrics are recorded. A confusion matrix is generated to visualize the model's predictions and the actual labels.

## 7. Comparison of CNN and ResNet-50

Both a CNN model and a ResNet-50 model are built and trained on the same dataset. Their performances on the validation and test sets are compared. The CNN model outperforms the ResNet-50 model in terms of accuracy.

## 8. Conclusion

In this analysis, we have demonstrated how to load, preprocess, build, train, and evaluate a CNN model for classifying chest X-ray images with pneumonia. The model's accuracy on the test set is reported, and a comparison is made with a ResNet-50 model. The CNN model shows better performance, achieving an accuracy of 71%, compared to ResNet-50's accuracy of 67%.

Feel free to explore and modify the code to further enhance the analysis or experiment with different models and architectures.


