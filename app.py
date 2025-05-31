from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("student_graduation_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])[features]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    return jsonify({
        "prediction": int(pred),
        "prob_graduate": float(proba[1]),
        "prob_dropout": float(proba[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
