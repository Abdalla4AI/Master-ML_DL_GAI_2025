


# MNIST and Classification
This repository contains implementations and experiments using the MNIST dataset, a collection of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels, and the corresponding label represents the digit. MNIST is a foundational dataset in machine learning, often referred to as the "Hello World" of the field. It is widely used for testing and developing classification algorithms.

## Google Colab Notebook
| Chapter 3 Classification | MNIST 98% Accuracy |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/03_classification.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb#scrollTo=h-tIl3el_v7x)|

## Contents of This Repository:
Data Preprocessing: Loading and preparing the MNIST dataset for model training.
Classification Models: Implementations using various algorithms, including:
  - Logistic Regression
  - Support Vector Machines (SVMs)
  - Neural Networks (CNNs)

**Evaluation:** Performance metrics such as Cross-Validation, Confusion Matrix, Confusion Matrix, Precision/Recall Trade-off and ROC Curve.

**Visualization:** Insights into model predictions and feature representations.

**Classification:** as will study Classification type like Multiclass, Multilabel, Multioutput Classification, Dummy (ie. random) classifier, KNN classifier

**Advanced Techniques:** Using Data Augmentation and Convolutional Neural Networks (CNNs) for improved accuracy.

## **Exercise solutions:**
1. **An MNIST Classifier:** I wonder how an MNIST classifier achieves over **99%** accuracy using a **CNN**, while only achieving **97%** accuracy with the help of the **k-nearest neighbors (KNN)** algorithm.
2. **Tackling the Titanic Dataset:** The goal is to predict whether or not a passenger survived based on attributes such as age, sex, passenger class, embarkation point, and so on. We achieved over **98%** accuracy using **Random Forests** algorism, and approximately **87%** using **KNN** with slight improvements over the **~80%** achieved using **Logistic Regression**.


TODO:
=============
- [ ] Train **Tackle the Titanic dataset** using features engineering like ['AgeBucket', 'RelativesOnboard'].
- [ ] Spam classifier.
