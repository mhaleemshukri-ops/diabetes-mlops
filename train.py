import json
import os

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Load diabetes dataset
Xy = load_diabetes(as_frame=True)

# Separate input features and target
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# Inspect the dataset
print("Dataset shape:", X.shape)
print("\nFeature names:")
print(X.columns.tolist())

print("\nFirst five rows:")
print(X.head())

print("\nTarget values:")
print(y.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Check the number of observations in each set
print("\nTraining observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])

# Create ML pipeline (v0.1)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

# Train the model using training data (only)
model.fit(X_train, y_train)

print("\nBaseline model (v0.1) trained successfully!")
print(model)

# Predict progression for the test observationz
y_pred = model.predict(X_test)

# CalculateRMSE
rmse = root_mean_squared_error(y_test, y_pred)

# show the evaluation result
print("\nBaseline Model Evaluation (v0.1)")
print(f"Test RMSE: {rmse:.4f}")

# Create folder for the model and the results
os.makedirs("models", exist_ok=True)

# Save the already trained model
joblib.dump(model, "models/model_v0.1.joblib")

# Save the RMSE result
metrics = {"version": "v0.1", "rmse": round(float(rmse), 4)}

with open("models/metrics_v0.1.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel and metrics saved!")

# Try Ridge Regression for the other version v0.2
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", Ridge(alpha=1.0))
])

# Train the Ridge-model
ridge_model.fit(X_train, y_train)

# Test the Ridge-model
ridge_pred = ridge_model.predict(X_test)
ridge_rmse = root_mean_squared_error(y_test, ridge_pred)

print("\nRidge Regression Evaluation")
print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Baseline RMSE: {rmse:.4f}")

# Trying different Ridge alpha values
print("\nTesting different Ridge settings:")

for alpha in [0.1, 1, 10, 100]:
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", Ridge(alpha=alpha))
    ])

    ridge.fit(X_train, y_train)
    predictions = ridge.predict(X_test)

    score = root_mean_squared_error(y_test, predictions)

    print(f"Alpha: {alpha} | RMSE: {score:.4f}")

    

# Comparing both Ridge models using cross validation
print("\nRidge cross-validation results:")

for alpha in [0.1, 1, 10, 100]:
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", Ridge(alpha=alpha))
    ])

    scores = cross_val_score(
        ridge,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    print(f"Alpha: {alpha} | Average RMSE: {-scores.mean():.4f}")


    # Final model for version v0.2
model_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", Ridge(alpha=10))
])

model_v2.fit(X_train, y_train)

# Evaluate the newer model
pred_v2 = model_v2.predict(X_test)
rmse_v2 = root_mean_squared_error(y_test, pred_v2)

print("\nFinal v0.2 results:")
print(f"v0.1 RMSE: {rmse:.4f}")
print(f"v0.2 RMSE: {rmse_v2:.4f}")

# Save the new model
joblib.dump(model_v2, "models/model_v0.2.joblib")

# Save also the results
metrics_v2 = {
    "version": "v0.2",
    "model": "Ridge",
    "alpha": 10,
    "rmse": round(float(rmse_v2), 4)
}

with open("models/metrics_v0.2.json", "w") as f:
    json.dump(metrics_v2, f, indent=4)

print("v0.2 model and metrics saved!")
