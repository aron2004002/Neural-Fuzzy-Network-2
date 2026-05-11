import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset

df = pd.read_csv("DATA/diabetes.csv")


# Features and target

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Feature scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# Train model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Prediction

predictions = model.predict(X_test)


# Accuracy

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Save model and scaler

joblib.dump(model, "MODEL/saved_model.pkl")
joblib.dump(scaler, "MODEL/scaler.pkl")

print("Model Saved Successfully")