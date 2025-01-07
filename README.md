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

# Decision Tree Notes:

- Multioutput classification.
- It is base algorism for Random Forest
- CART Classification And Regression Trees, sklearn use model or algorism CART to calc tree.
  it generate binary trees. (true and false leaves).
  CART do select feature (k) and threshold (t) to classification first node. (name it imurity) and select minimum impurity (gini).
- Cost function: (t>k)
	min J(k, t) = weighted average = ((min left / min) * gini-left) + ((min right / min) * gini-right)
	for each node. will stop when:
	1. rich max_depth variable.
	2. no value will divided.
- Greedy algorism: find minimum path to next node.
- Presort hyperparameter: if we want fast training? we set presort to true. sorting data if data small 3 to 4 thousands.
- Entropy: e = for k=1 to n: -1 * P * (log2 P). calc for all nodes in trees. P=> propability
	e = -1 * (49/54) * log2((49/54)) - (5/54) * log2((5/54)) .....
	for data (0, 49, 5).
- We can use gini or entropy, result are same. entropy balance tree more than gini.
- gini is default in algorism, and it faster than entropy.
- to change to entropy use hyperparameter intropion='entropy', in sklearn.
- DT: can fit data to rich accuracy 100%, it become overfitting. so no data generization.
- DT: it is non-parameters. we not know what they or how many
- To reduce overfitting we do regularization.
- regularization: use hyperparameter like max_depth(default = 'none').
	min_sample_split: in node has 30 sample if it less than min_sample_split? stop DT, no generate leaves.
	min_sample_leaf: leave count not less than this hyperparameter.
	min_weight_fraction_leaf: same as min_sample_leaf but value in fraction number.
	max_leaf_nodes: max leaf number in trees. (in last depth leaf)
	max_features: select random number of features.
- if reduce any 'max' hyperparameter? then we do regularization.
- if increase any 'min'  hyperparameter? then we do regularization.
- Another way is normal training DT without mis with hyperparameter to make regularization. after training we do pruning.
- Pruning: for each node with leaf, remove node only and check if model result not effect. finally we get best model.
- DT can use in regression (regression use in serial results). ex (x1,x2 features and we need predict y)
- Scaling not problem. the problem in location of data. for location we use PCA
- PCA: do features reduction, reduce dimension.
- DT: sensitive to any change in data. will change in gini values. 
- Random_state: if we have selective random? like DT will pickup specific (for example random features), or specific path on nodes/leaves.
	for example random_state = 42.
	We use this hyperparameter in other model/algorism. not executed to DT.
- Sensitive variation in data Random forst better than DT, DT has fail more than randon forst. 


- Note: ID3 another model, generate more than 2 leaves for one tree.
- Interpretation: we know how model internally work.
- Decision trees called "Whitebox algorism", 
- Note: Blackbox model like CNN, RForst,..
- DT: output is probability, like (0, 49, 3) for all 3 classis. results: relosa = 0/54, versi = 49/54, verginika = 5/54
      


## Decision boundary:

- Depth: how many horizontal level.



| Chapter 6 Decision Tree | Note & Exercise solutions |
|:-:|:-:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/06_decision_trees_ori.ipynb)|  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/Colab/06.Decision_Trees.ipynb)|

