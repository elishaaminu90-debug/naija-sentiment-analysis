"""
Step 2: Preprocessing for Multilingual Nigerian Sentiment Analysis
Run this in Google Colab, after Step 1 (needs afrisenti_combined.csv).

Install first (run once in a Colab cell):
    !pip install transformers scikit-learn pandas
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

# ---------------------------------------------------------
# 1. Load the combined dataset from Step 1
# ---------------------------------------------------------
df = pd.read_csv("afrisenti_combined.csv")
print(f"Loaded {len(df)} total examples")

# ---------------------------------------------------------
# 2. Clean text
#    IMPORTANT: we do NOT strip accents/diacritics — Yoruba's
#    tonal marks (e.g. Ìlerí) carry real linguistic meaning.
#    We only remove URLs and normalize whitespace.
# ---------------------------------------------------------
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

# Drop any rows that became empty after cleaning
before = len(df)
df = df[df["clean_tweet"].str.len() > 0].reset_index(drop=True)
print(f"Dropped {before - len(df)} empty rows after cleaning")

# ---------------------------------------------------------
# 3. Encode labels
# ---------------------------------------------------------
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# ---------------------------------------------------------
# 4. Stratified train/val/test split
#    Stratify on language + label combined so every
#    language/class combination is represented in all splits.
# ---------------------------------------------------------
df["strata"] = df["language"] + "_" + df["label"]

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["strata"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["strata"], random_state=42
)

print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ---------------------------------------------------------
# 5. Per-language class weights (for the loss function later)
#    This is how we handle Pidgin's small neutral class
#    without discarding data.
# ---------------------------------------------------------
print("\nPer-language class weights (for weighted loss):")
class_weights_by_lang = {}
for lang in df["language"].unique():
    lang_train = train_df[train_df["language"] == lang]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=lang_train["label_id"],
    )
    class_weights_by_lang[lang] = dict(zip([0, 1, 2], weights))
    print(f"  {lang}: {class_weights_by_lang[lang]}")

# ---------------------------------------------------------
# 6. Tokenize with AfroXLMR
#    AfroXLMR-base was the best-performing model under 270M
#    params in the original AfriSenti benchmark paper, and
#    it's realistic to fine-tune on a free Colab T4 GPU.
# ---------------------------------------------------------
MODEL_NAME = "Davlan/afro-xlmr-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Max length: covers the 95th percentile of tweet lengths
# across all 4 languages (Pidgin/Yoruba run longer than Hausa/Igbo)
MAX_LENGTH = 128

def tokenize_batch(texts):
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

# ---------------------------------------------------------
# 7. Save everything for Step 3 (model training)
# ---------------------------------------------------------
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

import json
with open("class_weights.json", "w") as f:
    json.dump(class_weights_by_lang, f, indent=2)

print("\nSaved: train_split.csv, val_split.csv, test_split.csv, class_weights.json")
print(f"Tokenizer ready: {MODEL_NAME}, max_length={MAX_LENGTH}")
