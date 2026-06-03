# =========================================
# House Price Prediction using ML
# Algorithm : Linear Regression
# =========================================

# STEP 1 — Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# STEP 1 - Read dataset
train_data = pd.read_csv("train.csv")

# STEP 2 - Select features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]

X = train_data[features]
y = train_data["SalePrice"]

# STEP 3 - Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4 - Train model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5 - Predict
predictions = model.predict(X_test)

# STEP 6 - Error check
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# STEP 7 - Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# STEP 8 - Read test dataset
test_data = pd.read_csv("test.csv")

# STEP 9 - Predict test data prices
test_features = test_data[features].fillna(0)

test_predictions = model.predict(test_features)

# STEP 10 - Create submission file
submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions
})

# STEP 11 - Save CSV
submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")