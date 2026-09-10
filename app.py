"""
Gradio app for Multilingual Nigerian Sentiment Analysis.
Built for Hugging Face Spaces on the free ZeroGPU tier.
"""

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "elishaaminu90/naija-sentiment-afroxlmr"
MAX_LENGTH = 128
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LANGUAGES = ["Hausa", "Igbo", "Nigerian Pidgin", "Yoruba"]

# Known limitation, from evaluation — Pidgin neutral F1 is only 0.31
# even after attempting data augmentation. Flag these rather than
# hiding the weakness.
KNOWN_WEAK_SPOT = ("Nigerian Pidgin", "Neutral")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


@spaces.GPU
def analyze_sentiment(text, language):
    if not text or not text.strip():
        return "Please enter some text.", None, ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    pred_id = int(torch.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    confidence = float(probs[pred_id])

    prob_dict = {ID2LABEL[i]: float(probs[i]) for i in range(3)}

    warning = ""
    if (language, pred_label) == KNOWN_WEAK_SPOT or confidence < 0.5:
        warning = (
            "⚠️ Lower confidence prediction. Nigerian Pidgin's 'Neutral' "
            "class has limited training data (only ~800 examples vs. "
            "thousands for other classes), so predictions in this specific "
            "combination are less reliable — a known, documented limitation "
            "of the training data, not a bug."
        )

    result_text = f"**Prediction: {pred_label}** (Confidence: {confidence:.1%})"

    return result_text, prob_dict, warning


with gr.Blocks(title="Nigerian Sentiment Analyzer") as demo:
    gr.Markdown("# 🇳🇬 Multilingual Nigerian Sentiment Analyzer")
    gr.Markdown(
        "Sentiment classification for Hausa, Igbo, Nigerian Pidgin, and "
        "Yoruba, fine-tuned from AfroXLMR-base on the AfriSenti dataset."
    )

    with gr.Row():
        language_input = gr.Dropdown(choices=LANGUAGES, label="Language of the text", value="Hausa")

    text_input = gr.Textbox(label="Enter text to analyze", lines=3)
    analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

    result_output = gr.Markdown()
    prob_output = gr.Label(label="Confidence breakdown")
    warning_output = gr.Markdown()

    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=[text_input, language_input],
        outputs=[result_output, prob_output, warning_output],
    )

    gr.Markdown(
        "---\n"
        "*Model: AfroXLMR-base fine-tuned on AfriSenti | "
        "Overall test macro F1: 0.763 | "
        "Per-language F1 — Hausa: 0.78, Igbo: 0.77, Yoruba: 0.73, Pidgin: 0.60*"
    )

demo.launch()
