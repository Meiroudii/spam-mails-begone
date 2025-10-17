from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

MODEL_PATH = "saved_model.joblib"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # accept form or JSON
    if request.is_json:
        data = request.get_json()
        text = data.get("text","")
    else:
        text = request.form.get("text","")

    if not text:
        return jsonify({"error":"no text provided"}), 400

    probs = model.predict_proba([text])[0]  # [prob_ham, prob_spam] or vice-versa
    pred = model.predict([text])[0]         # 0 or 1

    # Determine which column is spam depending on model.classes_
    # Many pipelines map classes_ = [0,1] where 1==spam based on training script mapping
    classes = model.classes_
    # find index of label 1 (spam)
    try:
        spam_idx = list(classes).index(1)
    except ValueError:
        # fallback: assume classes_[1] is spam
        spam_idx = 1 if len(classes) > 1 else 0

    spam_prob = float(probs[spam_idx])
    label = "spam" if pred == 1 else "ham"
    return jsonify({
        "label": label,
        "spam_probability": spam_prob,
        "text": text
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
