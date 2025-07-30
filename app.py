from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing files
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")
numeric_cols = joblib.load("model/numeric_cols.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Churn Prediction API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure all expected features exist (fill missing with 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training set
    input_df = input_df[feature_names]

    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = round(model.predict_proba(input_df)[0][1], 2)

    return jsonify({
        "churn_prediction": int(prediction),
        "churn_probability": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)