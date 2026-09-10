"""
Step 3: Fine-tune AfroXLMR for Multilingual Nigerian Sentiment Analysis
Run this in Google Colab with a T4 GPU (Runtime > Change runtime type > GPU).

Install first (run once in a Colab cell):
    !pip install transformers datasets scikit-learn torch accelerate
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, classification_report

MODEL_NAME = "Davlan/afro-xlmr-base"
MAX_LENGTH = 128
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------
# 1. Load splits + class weights from Step 2
# ---------------------------------------------------------
train_df = pd.read_csv("train_split_augmented.csv")
val_df = pd.read_csv("val_split.csv")

with open("class_weights_augmented.json") as f:
    class_weights_by_lang = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------
# 2. Dataset: tokenizes text and attaches a per-example
#    sample weight looked up from (language, label)
# ---------------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, class_weights_by_lang):
        # Guard against NaN/empty values (e.g. a back-translation that
        # returned blank text) — the tokenizer requires plain strings.
        df = df.copy()
        df["clean_tweet"] = df["clean_tweet"].fillna("").astype(str)
        df = df[df["clean_tweet"].str.strip().str.len() > 0].reset_index(drop=True)

        self.texts = df["clean_tweet"].tolist()
        self.labels = df["label_id"].tolist()
        self.languages = df["language"].tolist()
        self.tokenizer = tokenizer
        self.class_weights_by_lang = class_weights_by_lang

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        lang = self.languages[idx]
        label = self.labels[idx]
        weight = self.class_weights_by_lang[lang][str(label)]
        item["sample_weight"] = torch.tensor(weight, dtype=torch.float)
        return item


train_dataset = SentimentDataset(train_df, tokenizer, class_weights_by_lang)
val_dataset = SentimentDataset(val_df, tokenizer, class_weights_by_lang)

# ---------------------------------------------------------
# 3. Custom Trainer: applies per-example sample weights to
#    the loss instead of a single global class-weight vector,
#    since weights differ by language AND class.
# ---------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weight")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_example_loss = loss_fct(logits, labels)
        weighted_loss = (per_example_loss * sample_weights).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ---------------------------------------------------------
# 4. Metrics: macro F1 (fair across classes, doesn't get
#    inflated by the majority class)
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}


# ---------------------------------------------------------
# 5. Model + training config
#    Batch size 16 and fp16 are chosen to fit a free-tier
#    Colab T4 GPU (16GB) comfortably.
# ---------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

# ---------------------------------------------------------
# GPU check — fp16 requires a GPU, and CPU training will be
# dramatically slower for a model this size.
# ---------------------------------------------------------
has_gpu = torch.cuda.is_available()
if not has_gpu:
    print("\n" + "=" * 60)
    print("WARNING: No GPU detected — training will run on CPU.")
    print("This will take MUCH longer (hours instead of minutes).")
    print("Go to Runtime > Change runtime type > T4 GPU, then")
    print("restart the runtime and re-run all cells.")
    print("=" * 60 + "\n")

training_args = TrainingArguments(
    output_dir="./afroxlmr-sentiment-augmented",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    fp16=has_gpu,  # fp16 only works on GPU
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False,  # keep sample_weight — Trainer strips
                                   # any column it doesn't recognize as
                                   # a model input by default
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# 6. Train
# ---------------------------------------------------------
trainer.train()

# Save the best model for Step 4 (evaluation) and the app
trainer.save_model("./afroxlmr-sentiment-augmented-best")
tokenizer.save_pretrained("./afroxlmr-sentiment-augmented-best")
print("\nModel saved to ./afroxlmr-sentiment-augmented-best")
print("Copy this folder to Google Drive before your Colab session ends!")
