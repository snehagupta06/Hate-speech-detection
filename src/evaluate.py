import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

print("⏳ Loading dataset...")
df = pd.read_csv("../data/cleaned_balanced.csv")

# Fix NaN issue: replace missing with empty string
X = df["cleaned"].fillna("").astype(str)
y = df["Label"]

print("⏳ Loading model and vectorizer...")
model = joblib.load("../model/svm_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")

print("⏳ Transforming features...")
X_tfidf = vectorizer.transform(X)

print("⏳ Evaluating model...")
y_pred = model.predict(X_tfidf)

acc = accuracy_score(y, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")

# Classification report
report = classification_report(y, y_pred)
print(report)

# Save classification report to file
with open("../reports/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Not Hate", "Hate"],
            yticklabels=["Not Hate", "Hate"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("../reports/confusion_matrix.png")
plt.tight_layout()
plt.close()

# --- ROC Curve ---
y_prob = model.decision_function(X_tfidf)  # for SVM
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../reports/roc_curve.png")
plt.close()

print("📊 Evaluation complete!")
print("📝 Saved classification report to reports/classification_report.txt")
print("📊 Saved confusion matrix plot to reports/confusion_matrix.png")
