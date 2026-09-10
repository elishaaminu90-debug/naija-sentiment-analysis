# Multilingual Nigerian Sentiment Analysis

Sentiment classification (positive/negative/neutral) across four Nigerian
languages — Hausa, Igbo, Nigerian Pidgin, and Yorùbá — fine-tuning
[AfroXLMR-base](https://huggingface.co/Davlan/afro-xlmr-base) on the
[AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) dataset.

**Live demo:** https://huggingface.co/spaces/elishaaminu90/naija-sentiment-analyzer
**Model on the Hub:** https://huggingface.co/elishaaminu90/naija-sentiment-afroxlmr

## Results

| Language | Macro F1 |
|---|---|
| Hausa | 0.784 |
| Igbo | 0.768 |
| Yorùbá | 0.734 |
| Nigerian Pidgin | 0.603 |
| **Overall** | **0.763** |

### Known limitation

Nigerian Pidgin's neutral class is underrepresented in the source data
(~800 training examples after augmentation, vs. thousands for other
classes), giving it a much lower F1 (0.31) than every other class/language
combination. This was investigated directly:

- Tried back-translation augmentation (Pidgin → English → Pidgin via
  NLLB-200) to roughly double the class size — this made **no measurable
  difference** to the F1 score, suggesting the ceiling here is a genuine
  property of the data (short, informal Pidgin text where "neutral" and
  "mildly negative" overlap heavily) rather than something more synthetic
  data can fix.
- The deployed app surfaces a confidence warning for predictions in this
  specific language/class combination rather than hiding the weakness.

## Repository structure

```
├── app.py                  # Deployed Gradio app (live on HF Spaces, ZeroGPU)
├── requirements.txt        # Dependencies for the app
└── pipeline/                # End-to-end training pipeline (run in Colab)
    ├── 00_setup_drive.py         # Mount Drive for persistence across sessions
    ├── 01_explore_data.py        # Load AfriSenti, check class balance
    ├── 02_preprocess.py          # Clean text, stratified split, class weights
    ├── 03b_augment_pidgin.py     # Back-translation augmentation attempt
    ├── 02b_recompute_weights.py  # Recompute weights after augmentation
    ├── 03_train_model.py         # Fine-tune AfroXLMR with weighted loss
    ├── 04_evaluate.py            # Per-language evaluation + confusion matrices
    └── 05_push_to_hub.py         # Push trained model to Hugging Face Hub
```

## Methodology

1. **Data**: AfriSenti (SemEval 2023) — chosen over the earlier NaijaSenti
   release for its larger size and better class coverage per language.
2. **Preprocessing**: URLs stripped, whitespace normalized. Diacritics
   preserved deliberately — Yorùbá's tonal marks carry real linguistic
   meaning and are not noise.
3. **Split**: 80/10/10 train/val/test, stratified on language + label
   jointly so every language/class combination is represented in all
   three splits.
4. **Class imbalance**: per-language, per-class weights computed with
   `sklearn`'s balanced weighting, applied via a custom weighted
   cross-entropy loss (a single global class-weight vector doesn't work
   here since imbalance severity differs sharply by language).
5. **Model**: AfroXLMR-base — the best-performing model under 270M
   parameters in the original AfriSenti benchmark paper, fine-tuned for
   4 epochs with early stopping on validation macro F1.
6. **Evaluation**: macro F1 (not accuracy) both overall and per-language,
   plus confusion matrices, to make sure minority-class and minority-
   language performance isn't hidden behind aggregate numbers.

## Deployment

The app is built with Gradio and deployed on Hugging Face Spaces using
the free **ZeroGPU** tier (dynamic GPU allocation per request). The model
loads directly from the Hub rather than being bundled into the Space repo.
