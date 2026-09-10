"""
Step 4: Evaluation for Multilingual Nigerian Sentiment Analysis
Run this in Google Colab, after Step 3 (needs the saved model +
test_split.csv from Step 2).

Install first (run once in a Colab cell):
    !pip install transformers scikit-learn pandas torch
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "./afroxlmr-sentiment-best"
MAX_LENGTH = 128
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ---------------------------------------------------------
# 1. Load model + test set
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

test_df = pd.read_csv("test_split.csv")
print(f"Evaluating on {len(test_df)} test examples\n")

# ---------------------------------------------------------
# 2. Run predictions in batches
# ---------------------------------------------------------
def predict_batch(texts, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
    return all_preds

test_df["predicted"] = predict_batch(test_df["clean_tweet"].tolist())

# ---------------------------------------------------------
# 3. Overall metrics
# ---------------------------------------------------------
overall_f1 = f1_score(test_df["label_id"], test_df["predicted"], average="macro")
print(f"Overall Macro F1 on test set: {overall_f1:.4f}\n")
print("Overall classification report:")
print(classification_report(
    test_df["label_id"], test_df["predicted"],
    target_names=["negative", "neutral", "positive"]
))

# ---------------------------------------------------------
# 4. Per-language breakdown — this is the important part.
#    We specifically want to see Pidgin's neutral-class
#    performance, since that's the known weak spot.
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("PER-LANGUAGE BREAKDOWN")
print("=" * 60)

per_lang_results = {}
for lang in test_df["language"].unique():
    lang_df = test_df[test_df["language"] == lang]
    lang_f1 = f1_score(lang_df["label_id"], lang_df["predicted"], average="macro")
    per_lang_results[lang] = lang_f1

    print(f"\n--- {lang} (n={len(lang_df)}) ---")
    print(f"Macro F1: {lang_f1:.4f}")
    print(classification_report(
        lang_df["label_id"], lang_df["predicted"],
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
    ))

# ---------------------------------------------------------
# 5. Confusion matrix per language (visual check on where
#    Pidgin neutral predictions are actually going)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, lang in zip(axes, test_df["language"].unique()):
    lang_df = test_df[test_df["language"] == lang]
    cm = confusion_matrix(lang_df["label_id"], lang_df["predicted"], labels=[0, 1, 2])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"],
    )
    ax.set_title(lang)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# ---------------------------------------------------------
# 6. Save results summary
# ---------------------------------------------------------
summary_df = pd.DataFrame(
    list(per_lang_results.items()), columns=["language", "macro_f1"]
)
summary_df.to_csv("per_language_results.csv", index=False)
print("\nSaved per_language_results.csv and confusion_matrices.png")
