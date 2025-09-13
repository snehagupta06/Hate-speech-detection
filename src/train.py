# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("⏳ Loading cleaned dataset...")
df = pd.read_csv("../data/cleaned_balanced.csv")
df["cleaned"] = df["cleaned"].fillna("")
X = df["cleaned"]
y = df["Label"]

print("⏳ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⏳ Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=20000, ngram_range=(1, 3)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("⏳ Training SVM model...")
model = LinearSVC(class_weight="balanced")
model.fit(X_train_tfidf, y_train)

print("✅ Training complete!")

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "../model/svm_model.pkl")
joblib.dump(vectorizer, "../model/tfidf_vectorizer.pkl")

# Ensure reports folder exists
os.makedirs("../reports", exist_ok=True)

# Save classification report
with open("../reports/training_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(classification_report(y_test, y_pred))

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Training/Test Split)")
plt.tight_layout()
plt.savefig("../reports/train_confusion_matrix.png")
plt.close()

# Save Precision, Recall, F1 bar chart
metrics = ["precision", "recall", "f1-score"]
classes = ["0 (Non-Hate)", "1 (Hate)"]

plt.figure(figsize=(8, 6))
for i, metric in enumerate(metrics):
    values = [report_dict["0"][metric], report_dict["1"][metric]]
    plt.bar([x + i*0.25 for x in range(len(classes))], values, width=0.25, label=metric)

plt.xticks([x + 0.25 for x in range(len(classes))], classes)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score by Class")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/train_metrics.png")
plt.close()

print("\n✅ Model & vectorizer saved as svm_model.pkl and tfidf_vectorizer.pkl")
print("✅ Training report saved to reports/training_report.txt")
print("✅ Confusion matrix saved to reports/train_confusion_matrix.png")
print("✅ Metrics bar chart saved to reports/train_metrics.png")


