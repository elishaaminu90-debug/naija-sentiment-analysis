"""
Step 1: Data Exploration for Multilingual Nigerian Sentiment Analysis
Run this in Google Colab (same setup you use for TAI).

DATA SOURCE UPDATE: Switched from NaijaSenti to AfriSenti — the
follow-up benchmark by the same research group (used for SemEval 2023),
covering the same 4 Nigerian languages but with more annotated data
and includes Pidgin data collected specifically to address earlier
class-imbalance issues. We also combine train+dev+test splits here
since we'll create our own splits later for reproducibility.

NOTE: Same as before — we skip the deprecated HF `datasets` script
loader and pull the TSVs directly from GitHub.

Install first (run once in a Colab cell):
    !pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LANGUAGES = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "pcm": "Nigerian Pidgin",
    "yor": "Yoruba",
}

BASE_URL = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data"
SPLITS = ["train", "dev", "test"]

all_data = {}

for code, name in LANGUAGES.items():
    print(f"\nLoading {name} ({code})...")
    split_dfs = []
    for split in SPLITS:
        url = f"{BASE_URL}/{code}/{split}.tsv"
        try:
            df_split = pd.read_csv(url, sep="\t")
            df_split.columns = [c.strip().lower() for c in df_split.columns]
            if "tweet" not in df_split.columns:
                df_split = df_split.rename(columns={df_split.columns[0]: "tweet", df_split.columns[1]: "label"})
            df_split["source_split"] = split
            split_dfs.append(df_split)
        except Exception as e:
            print(f"  Could not load {split} split: {e}")

    df = pd.concat(split_dfs, ignore_index=True)
    df["language"] = name
    all_data[code] = df
    print(f"  {len(df)} examples total (train+dev+test combined)")
    print(f"  Label distribution:\n{df['label'].value_counts()}")

# Combine everything into one dataframe for a full picture
combined = pd.concat(all_data.values(), ignore_index=True)

# --- 1. Class balance per language ---
plt.figure(figsize=(10, 5))
sns.countplot(data=combined, x="language", hue="label")
plt.title("Sentiment Label Distribution by Language")
plt.savefig("label_distribution.png")
plt.show()

# --- 2. Tweet length distribution ---
combined["tweet_length"] = combined["tweet"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 5))
sns.boxplot(data=combined, x="language", y="tweet_length")
plt.title("Tweet Length (word count) by Language")
plt.savefig("tweet_length.png")
plt.show()

print("\nAverage tweet length by language:")
print(combined.groupby("language")["tweet_length"].mean())

# --- 3. Quick peek at a few examples per language ---
for code, name in LANGUAGES.items():
    print(f"\n--- {name} examples ---")
    print(all_data[code][["tweet", "label"]].sample(3, random_state=42))

# Save combined dataset locally for the next step (preprocessing)
combined.to_csv("afrisenti_combined.csv", index=False)
print("\nSaved combined dataset to afrisenti_combined.csv")
