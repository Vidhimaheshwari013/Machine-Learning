# SVM Classification – Loan Approval Prediction

## Overview
This project implements **Support Vector Machine (SVM)** for classifying loan applications as **Approved (Y)** or **Not Approved (N)** using a loan approval dataset.

## Methodology
- Removed the `ApplicantID` identifier.
- Encoded categorical features using **One-Hot Encoding**.
- Applied **StandardScaler** for feature scaling.
- Split the dataset into **80% training (480 samples)** and **20% testing (120 samples)**.
- Implemented an **RBF-kernel SVM classifier**.
- Applied **SelectKBest** and **RFE** for feature selection.
- Performed **5-Fold Stratified Cross-Validation**.

## Evaluation Metrics
The model was evaluated using:
- Accuracy and Error Rate
- True Positive (TP), True Negative (TN)
- False Positive (FP), False Negative (FN)
- Recall
- Specificity
- Precision
- F1-Score
- AUC

## Initial SVM Results

| Metric | Result |
|---|---:|
| Accuracy | 74.17% |
| Error | 25.83% |
| TP | 38 |
| TN | 51 |
| FP | 16 |
| FN | 15 |
| Recall | 71.70% |
| Specificity | 76.12% |
| Precision | 70.37% |
| F1-Score | 71.03% |
| AUC | 81.30% |

## Conclusion
The SVM classifier achieved **74.17% accuracy** and an **AUC of 81.30%** on the test data. Feature selection and cross-validation were also performed to evaluate the effectiveness and reliability of the classification model.
