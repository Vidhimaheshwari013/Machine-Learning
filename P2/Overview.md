# Mushroom Edibility Classification using Decision Tree

## About Decision Tree

A **Decision Tree** is a supervised machine learning algorithm used for **classification and regression**. It makes decisions by repeatedly splitting the dataset based on feature values, forming a tree-like structure of **nodes, branches, and leaf nodes**.

For classification, the algorithm selects the best feature and split at each node to make the classes as pure as possible. In this project, **Entropy** is used as the criterion for selecting the best splits. Lower entropy indicates greater purity of the resulting groups.

## Project Overview

This project uses a **Decision Tree Classifier** to predict whether a mushroom belongs to a particular class (such as edible or poisonous) using the **Mushroom Edibility Dataset**.

### Project Workflow

1. Load the mushroom dataset and remove the `SampleID` column.
2. Separate the dataset into **features (X)** and **target class (y)**.
3. Convert categorical values into numerical values using **Label Encoding**.
4. Split the data into **80% training and 20% testing** sets using stratified sampling.
5. Create and train a **Decision Tree Classifier using Entropy**.
6. Predict the classes for the test dataset.
7. Evaluate the model using:

   * Confusion Matrix
   * Accuracy
   * Error Rate
   * Recall / Sensitivity
   * Specificity
   * F1-Score
   * AUC
8. Perform **5-Fold Stratified Cross-Validation**.
9. Visualize the trained Decision Tree.

### Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

### Output

The project produces classification metrics, a confusion matrix, 5-fold cross-validation results, and a visual representation of the **Decision Tree built using Entropy**.
