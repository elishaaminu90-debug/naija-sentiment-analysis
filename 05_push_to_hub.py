"""
Push the trained model to Hugging Face Hub.
Run this in Colab after training.

Install first:
    !pip install huggingface_hub

You'll need a Hugging Face account + access token (from
huggingface.co/settings/tokens, "write" permission).
"""

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This will prompt you to paste your HF token
login()

MODEL_PATH = "afroxlmr-sentiment-augmented-best"
HUB_MODEL_NAME = "elishaaminu90/naija-sentiment-afroxlmr"  # change username if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.push_to_hub(HUB_MODEL_NAME)
tokenizer.push_to_hub(HUB_MODEL_NAME)

print(f"\nModel pushed to: https://huggingface.co/{HUB_MODEL_NAME}")
print("Your Streamlit app can now load it directly with:")
print(f'  AutoModelForSequenceClassification.from_pretrained("{HUB_MODEL_NAME}")')
