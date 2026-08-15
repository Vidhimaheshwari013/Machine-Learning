# ============================================================
# RICE PRODUCTION PREDICTION USING MACHINE LEARNING
# FAOSTAT DATA
#
# LINEAR:
# h(w,b) = w*x + b
#
# POLYNOMIAL:
# h(w,b) = w2*x^2 + w1*x + b
#
# MULTIVARIATE:
# h(w,b) = w1*x1 + w2*x2 + w3*x3 + b
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("rice.csv")

print("\nOriginal Dataset:")
print(df.head())


# ============================================================
# 2. CONVERT FAOSTAT DATA INTO YEAR-WISE FORMAT
# ============================================================

data = df.pivot_table(
    index="Year",
    columns="Element",
    values="Value",
    aggfunc="first"
).reset_index()

# Arrange in sequential chronological order
data = data.sort_values("Year").reset_index(drop=True)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

print("\nColumns in Dataset:")
print(data.columns.tolist())

print("\nDataset Size:")
print(data.shape)

print("\nFirst Year:", data["Year"].min())
print("Last Year :", data["Year"].max())


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

data = data.dropna(
    subset=[
        "Year",
        "Area harvested",
        "Yield",
        "Production"
    ]
).reset_index(drop=True)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
#
# IMPORTANT:
# No random shuffling.
#
# First 80%  -> Training
# Last 20%   -> Testing
#
# ============================================================

train_size = int(len(data) * 0.80)

train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()


print("\n============================================")
print("TRAINING AND TESTING DATA")
print("============================================")

print(
    "Training Period:",
    train["Year"].min(),
    "-",
    train["Year"].max()
)

print(
    "Testing Period :",
    test["Year"].min(),
    "-",
    test["Year"].max()
)

print(
    "Training Observations:",
    len(train)
)

print(
    "Testing Observations:",
    len(test)
)


# ============================================================
# 6. TARGET VARIABLE
# ============================================================

y_train = train["Production"].values
y_test = test["Production"].values


# ============================================================
# 7. LINEAR REGRESSION
#
# Hypothesis:
#
# h(w,b) = w*x + b
#
# x = Year
# w = weight
# b = bias
# ============================================================

X_train_linear = train[["Year"]]
X_test_linear = test[["Year"]]

linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)


# Get weight and bias
w_linear = linear_model.coef_[0]
b_linear = linear_model.intercept_


# Hypothesis function
def h_linear(x, w, b):

    return w * x + b


# Test prediction
linear_test_prediction = h_linear(
    test["Year"].values,
    w_linear,
    b_linear
)


print("\n============================================")
print("LINEAR REGRESSION")
print("============================================")

print("Hypothesis:")
print("h(w,b) = w*x + b")

print("\nw =", w_linear)
print("b =", b_linear)

print("\nEquation:")

print(
    f"h(w,b) = ({w_linear:.4f})x + "
    f"({b_linear:.4f})"
)


# ============================================================
# 8. POLYNOMIAL REGRESSION
#
# Hypothesis:
#
# h(w,b) = w2*x^2 + w1*x + b
#
# x = Year
# ============================================================

poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly_features.fit_transform(
    train[["Year"]]
)

X_test_poly = poly_features.transform(
    test[["Year"]]
)


poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)


# Get weights and bias
w1_poly = poly_model.coef_[0]
w2_poly = poly_model.coef_[1]
b_poly = poly_model.intercept_


# Hypothesis function
def h_polynomial(x, w1, w2, b):

    return (
        w2 * x**2
        + w1 * x
        + b
    )


# Test prediction
poly_test_prediction = h_polynomial(
    test["Year"].values,
    w1_poly,
    w2_poly,
    b_poly
)


print("\n============================================")
print("POLYNOMIAL REGRESSION")
print("============================================")

print("Hypothesis:")
print("h(w,b) = w2*x^2 + w1*x + b")

print("\nw1 =", w1_poly)
print("w2 =", w2_poly)
print("b  =", b_poly)

print("\nEquation:")

print(
    f"h(w,b) = ({w2_poly:.8f})x^2 "
    f"+ ({w1_poly:.4f})x "
    f"+ ({b_poly:.4f})"
)


# ============================================================
# 9. MULTIVARIATE LINEAR REGRESSION
#
# Hypothesis:
#
# h(w,b) = w1*x1 + w2*x2 + w3*x3 + b
#
# x1 = Year
# x2 = Area harvested
# x3 = Yield
# ============================================================

X_train_multi = train[
    [
        "Year",
        "Area harvested",
        "Yield"
    ]
]

X_test_multi = test[
    [
        "Year",
        "Area harvested",
        "Yield"
    ]
]


multi_model = LinearRegression()

multi_model.fit(
    X_train_multi,
    y_train
)


# Get weights and bias
w1_multi = multi_model.coef_[0]
w2_multi = multi_model.coef_[1]
w3_multi = multi_model.coef_[2]

b_multi = multi_model.intercept_


# Hypothesis function
def h_multivariate(
    x1,
    x2,
    x3,
    w1,
    w2,
    w3,
    b
):

    return (
        w1 * x1
        + w2 * x2
        + w3 * x3
        + b
    )


# Test prediction
multi_test_prediction = h_multivariate(
    test["Year"].values,
    test["Area harvested"].values,
    test["Yield"].values,
    w1_multi,
    w2_multi,
    w3_multi,
    b_multi
)


print("\n============================================")
print("MULTIVARIATE LINEAR REGRESSION")
print("============================================")

print("Hypothesis:")
print(
    "h(w,b) = w1*x1 + w2*x2 + w3*x3 + b"
)

print("\nw1 =", w1_multi)
print("w2 =", w2_multi)
print("w3 =", w3_multi)
print("b  =", b_multi)

print("\nEquation:")

print(
    f"h(w,b) = ({w1_multi:.4f})x1 "
    f"+ ({w2_multi:.6f})x2 "
    f"+ ({w3_multi:.4f})x3 "
    f"+ ({b_multi:.4f})"
)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + model_name)
    print("--------------------------------")

    print(
        "MAE  :",
        round(mae, 2)
    )

    print(
        "RMSE :",
        round(rmse, 2)
    )

    print(
        "R2   :",
        round(r2, 4)
    )

    return mae, rmse, r2


linear_scores = evaluate_model(
    "Linear Regression",
    y_test,
    linear_test_prediction
)

poly_scores = evaluate_model(
    "Polynomial Regression",
    y_test,
    poly_test_prediction
)

multi_scores = evaluate_model(
    "Multivariate Regression",
    y_test,
    multi_test_prediction
)


# ============================================================
# 11. MODEL COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MAE": [
        linear_scores[0],
        poly_scores[0],
        multi_scores[0]
    ],

    "RMSE": [
        linear_scores[1],
        poly_scores[1],
        multi_scores[1]
    ],

    "R2": [
        linear_scores[2],
        poly_scores[2],
        multi_scores[2]
    ]
})


print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

print(
    results.round(4).to_string(index=False)
)


# ============================================================
# 12. RETRAIN MODELS USING ALL HISTORICAL DATA
# ============================================================
#
# After testing, use complete historical data
# to predict future years.
#
# ============================================================


# ------------------------------------------------------------
# FINAL LINEAR MODEL
# ------------------------------------------------------------

final_linear = LinearRegression()

final_linear.fit(
    data[["Year"]],
    data["Production"]
)

w_linear_final = final_linear.coef_[0]
b_linear_final = final_linear.intercept_


# ------------------------------------------------------------
# FINAL POLYNOMIAL MODEL
# ------------------------------------------------------------

final_poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_all_poly = final_poly_features.fit_transform(
    data[["Year"]]
)

final_poly = LinearRegression()

final_poly.fit(
    X_all_poly,
    data["Production"]
)

w1_poly_final = final_poly.coef_[0]
w2_poly_final = final_poly.coef_[1]
b_poly_final = final_poly.intercept_


# ------------------------------------------------------------
# FINAL MULTIVARIATE MODEL
# ------------------------------------------------------------

final_multi = LinearRegression()

final_multi.fit(
    data[
        [
            "Year",
            "Area harvested",
            "Yield"
        ]
    ],
    data["Production"]
)

w1_multi_final = final_multi.coef_[0]
w2_multi_final = final_multi.coef_[1]
w3_multi_final = final_multi.coef_[2]

b_multi_final = final_multi.intercept_


# ============================================================
# 13. FUTURE YEARS
# ============================================================

future_years = np.array([
    2027,
    2028,
    2029,
    2030
])


# ============================================================
# 14. PREDICT FUTURE AREA HARVESTED
# ============================================================
#
# Multivariate model needs future Area.
# Area is estimated from its historical trend.
# ============================================================

area_model = LinearRegression()

area_model.fit(
    data[["Year"]],
    data["Area harvested"]
)

future_area = area_model.predict(
    pd.DataFrame({"Year": future_years})
)

# ============================================================
# 15. PREDICT FUTURE YIELD
# ============================================================
#
# Multivariate model needs future Yield.
# Yield is estimated from its historical trend.
# ============================================================

yield_model = LinearRegression()

yield_model.fit(
    data[["Year"]],
    data["Yield"]
)

future_yield = yield_model.predict(
    future_years.reshape(-1, 1)
)


# ============================================================
# 16. FUTURE PREDICTIONS
# ============================================================


# ------------------------------------------------------------
# LINEAR
#
# h(w,b) = w*x + b
# ------------------------------------------------------------

future_linear = h_linear(
    future_years,
    w_linear_final,
    b_linear_final
)


# ------------------------------------------------------------
# POLYNOMIAL
#
# h(w,b) = w2*x^2 + w1*x + b
# ------------------------------------------------------------

future_poly = h_polynomial(
    future_years,
    w1_poly_final,
    w2_poly_final,
    b_poly_final
)


# ------------------------------------------------------------
# MULTIVARIATE
#
# h(w,b) = w1*x1 + w2*x2 + w3*x3 + b
# ------------------------------------------------------------

future_multi = h_multivariate(
    future_years,
    future_area,
    future_yield,
    w1_multi_final,
    w2_multi_final,
    w3_multi_final,
    b_multi_final
)


# ============================================================
# 17. CREATE FUTURE PREDICTION TABLE
# ============================================================

future_results = pd.DataFrame({

    "Year": future_years,

    "Linear Prediction": future_linear,

    "Polynomial Prediction": future_poly,

    "Multivariate Prediction": future_multi
})


# ============================================================
# 18. DISPLAY FUTURE PREDICTIONS
# ============================================================

print("\n\n============================================")
print("      FUTURE RICE PRODUCTION PREDICTION")
print("              2027 - 2030")
print("============================================")

print(
    future_results.to_string(
        index=False,
        formatters={
            "Linear Prediction":
                lambda x: f"{x:,.0f}",

            "Polynomial Prediction":
                lambda x: f"{x:,.0f}",

            "Multivariate Prediction":
                lambda x: f"{x:,.0f}"
        }
    )
)


# ============================================================
# 20. ONE COMBINED GRAPH
# ============================================================
#
# This graph contains:
#
# 1. Actual historical production
# 2. Linear model test prediction
# 3. Polynomial model test prediction
# 4. Multivariate model test prediction
# 5. Linear future prediction
# 6. Polynomial future prediction
# 7. Multivariate future prediction
#
# ALL IN ONE GRAPH
# ============================================================

plt.figure(figsize=(14, 8))


# ------------------------------------------------------------
# ACTUAL HISTORICAL DATA
# ------------------------------------------------------------

plt.plot(
    data["Year"],
    data["Production"] / 1_000_000,
    marker="o",
    markersize=3,
    label="Actual Production"
)


# ------------------------------------------------------------
# LINEAR TEST PREDICTION
# ------------------------------------------------------------

plt.plot(
    test["Year"],
    linear_test_prediction / 1_000_000,
    linestyle="--",
    label="Linear Test Prediction"
)


# ------------------------------------------------------------
# POLYNOMIAL TEST PREDICTION
# ------------------------------------------------------------

plt.plot(
    test["Year"],
    poly_test_prediction / 1_000_000,
    linestyle="--",
    label="Polynomial Test Prediction"
)


# ------------------------------------------------------------
# MULTIVARIATE TEST PREDICTION
# ------------------------------------------------------------

plt.plot(
    test["Year"],
    multi_test_prediction / 1_000_000,
    linestyle="--",
    label="Multivariate Test Prediction"
)


# ------------------------------------------------------------
# LINEAR FUTURE PREDICTION
# ------------------------------------------------------------

plt.plot(
    future_years,
    future_linear / 1_000_000,
    marker="o",
    linestyle=":",
    label="Linear Future Prediction"
)


# ------------------------------------------------------------
# POLYNOMIAL FUTURE PREDICTION
# ------------------------------------------------------------

plt.plot(
    future_years,
    future_poly / 1_000_000,
    marker="s",
    linestyle=":",
    label="Polynomial Future Prediction"
)


# ------------------------------------------------------------
# MULTIVARIATE FUTURE PREDICTION
# ------------------------------------------------------------

plt.plot(
    future_years,
    future_multi / 1_000_000,
    marker="^",
    linestyle=":",
    label="Multivariate Future Prediction"
)


# ------------------------------------------------------------
# GRAPH FORMATTING
# ------------------------------------------------------------

plt.xlabel(
    "Year",
    fontsize=12
)

plt.ylabel(
    "Rice Production (Million Tonnes)",
    fontsize=12
)

plt.title(
    "Rice Production: Actual, Test Predictions and Future Forecast",
    fontsize=14
)

plt.axvline(
    x=test["Year"].min(),
    linestyle="--",
    label="Train/Test Boundary"
)

plt.grid(True)

plt.legend(
    fontsize=9
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. SAVE FUTURE PREDICTIONS
# ============================================================

future_results.to_csv(
    "rice_future_predictions_2027_2030.csv",
    index=False
)

print("\n============================================")
print("Prediction file saved successfully!")
print("File: rice_future_predictions_2027_2030.csv")
print("============================================")