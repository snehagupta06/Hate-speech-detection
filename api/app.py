import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "model", "svm_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl"))

app = Flask(__name__)

# helper to get top contributing features
def top_contributing_features(text, vectorizer, model, top_n=5):
    X = vectorizer.transform([text])
    coef = model.coef_[0]  # binary classifier coefficients
    feat_names = np.array(vectorizer.get_feature_names_out())
    contrib = (X.toarray()[0]) * coef

    pos_idx = np.argsort(-contrib)[:top_n]
    neg_idx = np.argsort(contrib)[:top_n]

    top_pos = [(feat_names[i], float(round(contrib[i], 4))) for i in pos_idx]
    top_neg = [(feat_names[i], float(round(contrib[i], 4))) for i in neg_idx]

    return top_pos, top_neg

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Transform input
        X_tfidf = vectorizer.transform([text])

        # Let model decide (don’t override threshold)
        pred = int(model.predict(X_tfidf)[0])

        # Map labels based on your dataset (1 = Hate, 0 = Non-Hate)
        label = "Hate Speech" if pred == 1 else "Non-Hate Speech"

        # Explainability (optional)
        top_pos, top_neg = top_contributing_features(text, vectorizer, model)

        return jsonify({
            "input": text,
            "prediction": pred,       # 0 or 1
            "label": label,           # Human-readable
            "top_positive_features": top_pos,
            "top_negative_features": top_neg
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
