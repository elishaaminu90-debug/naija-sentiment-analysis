"""
Recompute class weights after Pidgin neutral augmentation.
Run this in Colab before re-training.
"""

import json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

train_df = pd.read_csv("train_split_augmented.csv")

print("Per-language class weights (recomputed on augmented data):")
class_weights_by_lang = {}
for lang in train_df["language"].unique():
    lang_train = train_df[train_df["language"] == lang]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=lang_train["label_id"],
    )
    class_weights_by_lang[lang] = dict(zip([0, 1, 2], weights))
    print(f"  {lang}: {class_weights_by_lang[lang]}")

with open("class_weights_augmented.json", "w") as f:
    json.dump(class_weights_by_lang, f, indent=2)

print("\nSaved class_weights_augmented.json")
