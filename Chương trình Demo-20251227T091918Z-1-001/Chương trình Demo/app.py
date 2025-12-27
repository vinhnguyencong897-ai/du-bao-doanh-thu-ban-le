# app.py
"""
Flask web app that loads model.bin and serves a small HTML form.
It auto-detects the required number of features from the trained model.
Run: python app.py
"""

from flask import Flask, request, render_template, redirect, url_for, flash
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

MODEL_PATH = 'model.bin'

# Load model
model = None
num_features = 0

if os.path.exists(MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # READ INPUT SHAPE FROM MODEL
    try:
        num_features = model.n_features_in_
    except:
        num_features = 0


@app.route('/', methods=['GET'])
def index():
    """
    Render homepage with dynamic number of input fields
    """
    if model is None:
        flash('Model not found. Please train the model first.', 'danger')
        return render_template('index.html', prediction=None, num_features=0)

    return render_template('index.html', prediction=None, num_features=num_features)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("Model not found. Train model first!", "danger")
        return redirect(url_for("index"))

    try:
        # collect all features dynamically
        features = []
        for i in range(num_features):
            value = float(request.form.get(f'feature{i+1}', '0'))
            features.append(value)

        X = np.array([features])
        pred = model.predict(X)[0]

        return render_template("index.html",
                               prediction=round(float(pred), 4),
                               num_features=num_features,
                               inputs=features)

    except ValueError:
        flash("Invalid numeric input.", "danger")
        return redirect(url_for("index"))


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
