import joblib
from sklearn.datasets import load_diabetes

# Load v0.2 model(saved)
model = joblib.load("models/model_v0.1.joblib")

# take one patient from the diabetes dataset
data = load_diabetes(as_frame=True)
patient = data.frame.drop(columns=["target"]).iloc[[0]]

# Predict disease progression
prediction = model.predict(patient)

print("Predicted progression:", prediction[0])
print("Actual progression:", data.frame["target"].iloc[0])