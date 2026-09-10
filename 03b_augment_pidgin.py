"""
Step 3b: Data Augmentation for Pidgin Neutral Class (via Back-Translation)
Run this in Google Colab BEFORE re-running Step 3 (training).
Only touches train_split.csv — val/test are left untouched to avoid
data leakage from near-duplicate augmented examples.

Install first (run once in a Colab cell):
    !pip install transformers sentencepiece pandas
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# NLLB language codes: Nigerian Pidgin and English
PCM = "pcm_Latn"
ENG = "eng_Latn"

def translate(texts, src_lang, tgt_lang, batch_size=16):
    tokenizer.src_lang = src_lang
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=128,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ---------------------------------------------------------
# 1. Load training split, isolate Pidgin neutral examples
# ---------------------------------------------------------
train_df = pd.read_csv("train_split.csv")

pidgin_neutral = train_df[
    (train_df["language"] == "Nigerian Pidgin") & (train_df["label"] == "neutral")
].copy()
print(f"Found {len(pidgin_neutral)} Pidgin neutral examples in training split")

# ---------------------------------------------------------
# 2. Back-translate: Pidgin -> English -> Pidgin
#    This round trip naturally paraphrases the text while
#    (usually) preserving the original meaning/sentiment.
# ---------------------------------------------------------
print("Translating Pidgin -> English...")
english_versions = translate(pidgin_neutral["clean_tweet"].tolist(), PCM, ENG)

print("Translating English -> Pidgin (back)...")
augmented_texts = translate(english_versions, ENG, PCM)

# ---------------------------------------------------------
# 3. Build augmented rows and filter out near-duplicates
#    (back-translation sometimes returns text identical to
#    the original, which wouldn't add any value)
# ---------------------------------------------------------
augmented_df = pidgin_neutral.copy()
augmented_df["clean_tweet"] = augmented_texts

before = len(augmented_df)
augmented_df = augmented_df[
    augmented_df["clean_tweet"].str.lower() != pidgin_neutral["clean_tweet"].str.lower().values
]
print(f"Dropped {before - len(augmented_df)} examples that came back unchanged")

# ---------------------------------------------------------
# 4. Combine with original training data and save
# ---------------------------------------------------------
train_df_augmented = pd.concat([train_df, augmented_df], ignore_index=True)

train_df_augmented.to_csv("train_split_augmented.csv", index=False)

print(f"\nOriginal train size: {len(train_df)}")
print(f"Augmented train size: {len(train_df_augmented)}")
print(f"Pidgin neutral count now: "
      f"{len(train_df_augmented[(train_df_augmented['language']=='Nigerian Pidgin') & (train_df_augmented['label']=='neutral')])}")
print("\nSaved to train_split_augmented.csv")
print("Next: re-run Step 2's class-weight calculation on this new file,")
print("then re-run Step 3 training using train_split_augmented.csv instead of train_split.csv")
