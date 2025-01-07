# Master ML DL GAI 2025
Welcome to Our Group for Mastering Machine Learning, Deep Learning, and Generative AI!
</br>
</br>
</br>


![](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/images/Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.jpg)


</br>

# [Chapter 1 Introduction](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/wiki/1.-Home)
</br>

Implementing "Polynomial Regression model" algorism to distinguiture results among algorisms mentioned in chaper one, here modified "Chapter 1 – The Machine Learning landscape notebook", [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/01_the_machine_learning_landscape.ipynb)
</br>

   - [Chapter1: Exercises & Note 1](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/wiki/2.-Chapter1:-Exercises-&-Note-1)</br>
   - [Chapter1: Exercises & Note 2](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/wiki/3.-Chaper1,-Exercises-&-Note-2)


# [Chapter 2 End-to-End Machine Learning Project Workflow](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/week02_EndToEndProject/README.md)
</br>

| Full Tutorial | Enhaced Google Colab  |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/02_end_to_end_machine_learning_project.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/02_Regression_Models_for_California_Housing_Price_Prediction-Copy1.ipynb)|

This notes outlines the process of executing an end-to-end machine learning (ML) project based on Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron, enriched with insights from modern ML practices.


# [Chapter 3 Classification](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/week03_Classification/Readme.md)

This repository contains classification algorithms and their implementations, along with experiments using the MNIST dataset. I am exploring how to achieve approximately **99%** accuracy using **Convolutional Neural Networks (CNNs)**.

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0–9). Each image is 28x28 pixels, and its corresponding label represents the digit. Often referred to as the "Hello World" of machine learning, MNIST is a foundational dataset widely used for testing and developing classification algorithms.


| Chapter 3 Classification | MNIST ~99% Accuracy |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/03_classification.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb#scrollTo=h-tIl3el_v7x)|


# Chapter 4: Training Models:
In this chapter, We will go deeper into how things work, and understanding what’s under the hood will help us debug issues and perform error analysis more efficiently.


| Training Models | Exercise solutions |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/04_training_linear_models.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/04_Chapter04_Exercise_solutions.ipynb)|

# Chapter 5: Support Vector Machines:
A Support Vector Machine (SVM) is a powerful and versatile machine learning model used for classification, regression, and outlier detection, handling both linear and nonlinear data. It is especially effective for classifying complex, small to medium-sized datasets. SVMs work by finding the optimal hyperplane that separates different classes in the feature space, maximizing the margin between the classes. This chapter will explain the fundamental concepts behind SVMs, their practical applications, and how they function, making them an essential tool in machine learning.

**In underline **Exercise solutions** notebooks will found some usefull technic like:**
1. Simplest way to use the pickle module to save and load a scikit-learn model locally.
2. Grid search is a common method to optimize hyperparameters in machine learning models. It systematically searches through a specified subset of hyperparameters to find the best combination based on cross-validation performance.
3. To optimize the hyperparameters of an SVM (Support Vector Machine) for the MNIST dataset using Grid Search, we can follow these steps. We'll use SVC (Support Vector Classification) from Scikit-Learn, and apply GridSearchCV to tune the C, kernel, and gamma parameters, which are crucial for SVM performance.

**In a Jupyter Notebook Markdown Tips, you can write Markdown in a cell to include rich text, such as headings, lists, links, images, and code formatting**

| Exercise solutions | Jupyter Notebook Markdown Tips |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/05_support_vector_machines.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/Jupyter_notebook_how_to_write_markdown.ipynb)|


# Chapter 6: Decision Trees:

In this Chapter, we will start by discussing how to train, validate, and make predictions with decision trees. Then we will go through the CART training algorithm used by `Scikit-Learn`, we will discuss how to regularize trees and use them in regression tasks. Finally, we will discuss some of the limitations of decision trees.

# **Decision Tree (DT) Overview**

## **1. General Concepts**
- Decision Trees can handle **multi-output classification** problems.
- They serve as the **base algorithm for Random Forest** models.
- The **CART (Classification And Regression Trees)** algorithm is used in `sklearn` to construct decision trees, generating **binary trees** with "True" and "False" leaves.
- The algorithm selects the **best feature (k) and threshold (t)** at each node based on **impurity** (e.g., Gini impurity).
- The goal is to minimize impurity using the following cost function:

  \[
  J(k, t) = \frac{\text{samples left}}{\text{total samples}} \times \text{Gini left} + \frac{\text{samples right}}{\text{total samples}} \times \text{Gini right}
  \]

- The tree stops growing when:
  1. It reaches the `max_depth` limit.
  2. No further splits can be made.

---

## **2. Algorithm Characteristics**
- **Greedy Algorithm:** Decision Trees use a greedy approach to find the best split at each step.
- **Pre-sorting (Hyperparameter):** If set to `True`, sorting is performed before training for faster results (useful for small datasets of around 3,000-4,000 samples).

---

## **3. Gini Impurity vs. Entropy**
- **Gini impurity (default):** Faster computation, commonly used.
- **Entropy Calculation:**
  
  \[
  E = -\sum P_i \log_2 P_i
  \]

  - Entropy balances the tree better than Gini impurity.
  - To use entropy in `sklearn`, set `criterion="entropy"`.
- Both methods produce similar results, but entropy can create a more balanced tree.

---

## **4. Overfitting & Regularization**
- Decision Trees can achieve **100% accuracy** on training data, leading to **overfitting** and poor generalization.
- DTs are **non-parametric**, meaning we don’t predefine the number of parameters.
- **Regularization Methods:**
  - Use hyperparameters like:
    - `max_depth`: Limits tree depth (default = None).
    - `min_samples_split`: Minimum samples required to split a node.
    - `min_samples_leaf`: Minimum samples required in a leaf node.
    - `min_weight_fraction_leaf`: Similar to `min_samples_leaf` but in fractional terms.
    - `max_leaf_nodes`: Limits the number of leaf nodes.
    - `max_features`: Restricts the number of features considered at each split.
  - **Regularization Rules:**
    - Decreasing **max** hyperparameters = **more regularization**.
    - Increasing **min** hyperparameters = **more regularization**.
- Another approach is **pruning**, where nodes are removed **after training** if they don’t significantly impact accuracy.

---

## **5. Decision Trees in Regression**
- DTs can also be used for **regression** tasks, predicting continuous values.
- They are **not sensitive to feature scaling**, but they are sensitive to **feature locations** (PCA can help with dimensionality reduction).

---

## **6. Randomness and Variability**
- Decision Trees are highly sensitive to changes in data; small variations can alter splits.
- Using **random_state** ensures reproducibility (e.g., `random_state=42`).
- Random Forest models are more **stable and accurate** than individual Decision Trees.

---

## **7. Other Decision Tree Variants**
- **ID3 (Iterative Dichotomiser 3):** Unlike CART, it can generate more than two child nodes per split.
- **Interpretability:** Decision Trees are highly interpretable models, known as **"White-box algorithms"**, whereas models like CNNs and Random Forests are considered **"Black-box"** models.

---

## **8. Decision Boundary**
- **Tree depth** determines the number of decision levels (horizontal splits).
- The decision boundary is defined by feature splits, forming **rectangular regions** in feature space.

---


| Chapter 6 Decision Tree | Note & Exercise solutions |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/06_decision_trees_ori.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/06.Decision_Trees.ipynb)|

---

\[
  J(k, t) = \left( \frac{\text{samples left}}{\text{total samples}} \times \text{Gini left} \right) + \left( \frac{\text{samples right}}{\text{total samples}} \times \text{Gini right} \right)
\]
