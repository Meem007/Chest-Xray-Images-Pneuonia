# Chest X-ray Images Pneumonia Detection

This repository contains code for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNN) and the ResNet-50 model. The dataset used for this project is collected from Kaggle and can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Comparison of CNN and ResNet-50](#comparison-of-cnn-and-resnet-50)
- [Conclusion](#conclusion)

## Introduction

Pneumonia is a lung infection that can be identified through chest X-ray images. This repository presents two deep learning approaches – CNN and ResNet-50 – for automating the diagnosis of pneumonia from chest X-ray images.

## Dependencies

The following libraries are used in this project:

- OpenCV
- Matplotlib
- NumPy
- Pandas
- Seaborn
- Scikit-learn
- TensorFlow

You can install them using the following command:

```bash
pip install opencv-python matplotlib numpy pandas seaborn scikit-learn tensorflow
```

## Data Preprocessing

- Load images from the `train`, `validation`, and `test` directories.
- Convert images to RGB format and resize them to a consistent size (e.g., 224x224).
- Create DataFrames for each dataset containing image arrays and labels.
- Normalize pixel values to [0, 1] and convert labels to categorical format.
- Split data into training, validation, and test sets.

## Model Building

### CNN Model

- Create a Sequential model with Convolutional and Pooling layers.
- Flatten the data and add Dense layers for classification.

### ResNet-50 Model

- Load the ResNet-50 model with pre-trained weights and without the top classification layer.
- Add custom Dense layers for classification on top of the pre-trained model.

## Model Training and Evaluation

- Train both the CNN and ResNet-50 models on the training data.
- Evaluate the models on the validation and test datasets.
- Plot training history including accuracy and loss curves.
- Calculate and visualize the confusion matrix for model performance.

## Comparison of CNN and ResNet-50

Both models were trained and evaluated on the same dataset to predict pneumonia from chest X-ray images. After training, the models were compared based on their validation and test accuracies.

The CNN model achieved an accuracy of 71% on the test dataset, while the ResNet-50 model achieved an accuracy of 67%. This suggests that the CNN model performs better in this context.

## Conclusion

This project demonstrates the application of Convolutional Neural Networks and the ResNet-50 model for pneumonia detection in chest X-ray images. The results show that the CNN model outperforms the ResNet-50 model in terms of accuracy. This repository serves as a starting point for further exploration and improvement of pneumonia detection using deep learning techniques.
