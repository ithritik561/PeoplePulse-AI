import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load employee data
data = pd.read_csv("data/employees.csv")

print("Dataset loaded successfully!")
print("Total employees:", len(data))


# 2. Convert text columns into numbers
categorical_columns = [
    "Department",
    "JobRole",
    "OverTime",
    "Workload"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder


# 3. Convert Attrition target into 0/1
target_encoder = LabelEncoder()
data["Attrition"] = target_encoder.fit_transform(data["Attrition"])

# 4. Remove EmployeeID because it is only an identifier
X = data.drop(columns=["EmployeeID", "Attrition"])
y = data["Attrition"]


# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Create AI model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# 7. Train model
model.fit(X_train, y_train)


# 8. Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# 9. Save model
joblib.dump(model, "models/attrition_model.pkl")

# Save encoders too
joblib.dump(encoders, "models/encoders.pkl")

print("\nModel saved in models/attrition_model.pkl")
print("Encoders saved in models/encoders.pkl")