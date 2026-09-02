# Mushroom Edibility Classification using Naive Bayes

## About Naive Bayes

**Naive Bayes** is a supervised machine learning classification algorithm based on **Bayes' Theorem**. It calculates the probability of each possible class based on the given input features and predicts the class with the highest probability.

It is called "naive" because it assumes that the features are **independent of each other** given the class. This makes the algorithm simple, fast, and effective for many classification problems.

In this project, **Categorical Naive Bayes (`CategoricalNB`)** is used because the mushroom dataset contains categorical features.

## Project Overview

This project uses the **Naive Bayes Classification algorithm** to classify mushrooms into their respective classes using the **Mushroom Edibility Dataset**.

### Project Workflow

1. Load the mushroom dataset and remove the `SampleID` column.
2. Separate the dataset into **features (X)** and **target class (y)**.
3. Convert categorical features into numerical values using **Label Encoding**.
4. Split the dataset into **80% training and 20% testing** data using stratified sampling.
5. Create and train a **Categorical Naive Bayes** model.
6. Predict the classes for the test dataset.
7. Evaluate the model using:

   * Confusion Matrix
   * Accuracy
   * Error Rate
   * Recall / Sensitivity
   * Specificity
   * F1-Score
   * AUC
8. Perform **5-Fold Stratified Cross-Validation** to evaluate model performance.

### Technologies Used

* Python
* Pandas
* Scikit-learn

### Output

The project produces classification performance metrics, a confusion matrix, AUC score, and **5-fold cross-validation accuracy** for evaluating the Naive Bayes model.
