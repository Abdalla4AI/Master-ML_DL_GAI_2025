


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
  - Neural Networks (MLPs and CNNs)

**Evaluation:** Performance metrics such as Cross-Validation, Confusion Matrix, Confusion Matrix, Precision/Recall Trade-off and ROC Curve.

**Visualization:** Insights into model predictions and feature representations.

**Classification:** We study Classification type like Multiclass, Multilabel, Multioutput Classification, Dummy (ie. random) classifier, KNN classifier

**Advanced Techniques:** Using Data Augmentation and Convolutional Neural Networks (CNNs) for improved accuracy.

**Exercise solutions:**
1. **An MNIST Classifier:** With Over **97%** Accuracy with help of k-nearest neighbors (KNN) algorithm.
2. I wounder how An MNIST Classifier rich over **99%** accuracy using **CNN**.
3. **Tackle the Titanic dataset:** The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on. We rich over **98%** accuracy using **Random Forests**, **~87%** using KNN Slight improvement and **~80%** using Logistic Regression.


TODO:
=============
- [ ] Train **Tackle the Titanic dataset** using features engineering like ['AgeBucket', 'RelativesOnboard'].
- [ ] Spam classifier.
