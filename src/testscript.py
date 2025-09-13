import joblib
import os
import numpy as np

def top_contributing_features(text, vectorizer, model, n=5):
    """Return top positive and negative contributing features for explanation."""
    # Transform input
    X_tfidf = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get coefficients (works with linear models like SVM, LogisticRegression)
    coefs = model.coef_[0] * X_tfidf.toarray()[0]

    # Sort features
    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]

    top_pos = [(feature_names[i], round(coefs[i], 4)) for i in top_pos_idx if coefs[i] > 0]
    top_neg = [(feature_names[i], round(coefs[i], 4)) for i in top_neg_idx if coefs[i] < 0]

    return top_pos, top_neg

# Load saved model + vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "svm_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl"))

# Some test sentences
samples = [
    "i like you",
    "i love you",
    "you are amazing",
    "have a nice day",
    "i hate you",
    "fuck you",
    "you stupid idiot"
]

# Transform & predict
X = vectorizer.transform(samples)
preds = model.predict(X)

# Print results
for text, p in zip(samples, preds):
    print(f"{text:20} --> {p}")

print(top_contributing_features("i like you", vectorizer, model))
print(top_contributing_features("i hate you", vectorizer, model))
