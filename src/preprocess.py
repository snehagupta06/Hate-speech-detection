# preprocess.py
import pandas as pd
import re

print("⏳ Loading dataset...")
df = pd.read_csv("HateSpeechDatasetBalanced.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    return text.strip()

print("⏳ Cleaning text...")
df["cleaned"] = df["Content"].apply(clean_text)

df.to_csv("cleaned_balanced.csv", index=False)

print("✅ Preprocessing done! Saved as cleaned_balanced.csv")
print(df.head())
