# Rice Production Prediction Using Machine Learning

This project uses **FAOSTAT rice production data** to predict future rice production using three machine learning approaches:

* **Linear Regression** – uses Year as the input variable.
* **Polynomial Regression** – uses Year with a quadratic relationship.
* **Multivariate Linear Regression** – uses Year, Area Harvested, and Yield as input variables.

### Project Workflow

1. Load and preprocess the FAOSTAT dataset.
2. Convert the data into a year-wise format.
3. Remove missing values.
4. Split the historical data into **80% training and 20% testing** sets.
5. Train Linear, Polynomial, and Multivariate Regression models.
6. Evaluate the models using **MAE, RMSE, and R² score**.
7. Retrain the models using all available historical data.
8. Forecast rice production for **2027–2030**.
9. Generate a combined graph comparing actual data, test predictions, and future forecasts.
10. Save the future predictions as `rice_future_predictions_2027_2030.csv`.

### Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Output

The project provides model performance comparisons, future rice production predictions for **2027–2030**, and a visualization of historical and predicted production.

