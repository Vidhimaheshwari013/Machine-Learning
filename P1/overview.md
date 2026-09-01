# 🌾 Crop Production Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting **future crop production using Machine Learning techniques** based on historical agricultural data obtained from **FAOSTAT (Food and Agriculture Organization of the United Nations)**.

The project uses historical crop production data to identify trends and relationships between different agricultural variables and production. Machine Learning models are then applied to the historical data to estimate and predict crop production for future years.

The main objective of this project is to demonstrate how historical agricultural data can be analyzed using mathematical and Machine Learning models to support **future crop production forecasting and agricultural decision-making**.

---

## 🎯 Objectives

The major objectives of this project are:

* To analyze historical crop production data obtained from FAOSTAT.
* To preprocess and organize the agricultural dataset for Machine Learning.
* To implement different regression-based Machine Learning models and compare **Linear Regression, Polynomial Regression, and Multivariate Regression** approaches.
* To visualize actual production, model predictions, and future forecasts using graphs.

---
## 🤖 Machine Learning Models Used

Three regression approaches are implemented in this project.

### 1. Linear Regression

Linear Regression is used to model the relationship between the input variable and crop production using a straight-line relationship.

The basic hypothesis function used in the project is:

```text
h(w,b) = wx + b
```

where:

* `h(w,b)` = predicted crop production
* `w` = model weight/coefficient
* `b` = bias/intercept
* `x` = input variable, such as year

The model learns the values of `w` and `b` from the historical dataset and uses them to estimate crop production.

---

### 2. Polynomial Regression

Crop production may not always increase or decrease in a perfectly straight-line pattern. Therefore, Polynomial Regression is used to capture possible nonlinear trends in the historical data.

A polynomial hypothesis can be represented as:

```text
h(w,b) = w₀ + w₁x + w₂x² + ... + wₙxⁿ
```

The polynomial model allows the prediction curve to follow the changing trend of crop production more effectively than a simple straight-line model.

Different polynomial degrees can be tested to determine how well the model represents the historical data.

---

### 3. Multivariate Regression

The Multivariate Regression model uses more than one input variable to predict crop production.
