# ============================================================
# NAIVE BAYES CLASSIFICATION
# Mushroom Edibility Dataset
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("11_mushroom_edibility.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# 2. REMOVE SAMPLE ID
# ------------------------------------------------------------

df = df.drop("SampleID", axis=1)

# ------------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

# ------------------------------------------------------------
# 4. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

print("\nClasses:")
print(target_encoder.classes_)

# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 6. CREATE NAIVE BAYES MODEL
# ------------------------------------------------------------

model = CategoricalNB()

# ------------------------------------------------------------
# 7. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 8. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# Probability for AUC
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 9. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)

print("\nTN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)

# ------------------------------------------------------------
# 10. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("CLASSIFICATION METRICS")
print("======================================")

print("Accuracy =", accuracy)
print("Accuracy (%) =", accuracy * 100)

# ------------------------------------------------------------
# 11. ERROR
# ------------------------------------------------------------

error = 1 - accuracy

print("Error =", error)
print("Error (%) =", error * 100)

# ------------------------------------------------------------
# 12. RECALL / SENSITIVITY
# ------------------------------------------------------------

recall = recall_score(y_test, y_pred)

print("Recall =", recall)

# ------------------------------------------------------------
# 13. SPECIFICITY
# ------------------------------------------------------------

specificity = TN / (TN + FP)

print("Specificity =", specificity)

# ------------------------------------------------------------
# 14. F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(y_test, y_pred)

print("F1-Score =", f1)

# ------------------------------------------------------------
# 15. AUC
# ------------------------------------------------------------

auc = roc_auc_score(y_test, y_prob)

print("AUC =", auc)

# ------------------------------------------------------------
# 16. K-FOLD CROSS VALIDATION
# ------------------------------------------------------------

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n======================================")
print("5-FOLD CROSS VALIDATION")
print("======================================")

print("Fold Accuracies:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())