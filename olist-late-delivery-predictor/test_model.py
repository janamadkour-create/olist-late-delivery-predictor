import joblib
import json

model = joblib.load(
    "artifacts/late_delivery_model.joblib"
)

with open("artifacts/model_metadata.json", "r") as f:
    metadata = json.load(f)

with open("artifacts/feature_schema.json", "r") as f:
    schema = json.load(f)

print("Model Loaded Successfully")
print("Model Type:", type(model).__name__)
print("Number of Features:", len(schema["features"]))
print("Target:", metadata["target"])
print("Thresholds:", metadata["thresholds"])