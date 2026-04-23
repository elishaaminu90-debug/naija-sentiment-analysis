
# app.py - Sentiment & Emotion Analyzer (Enhanced)


# STEP 1: IMPORT LIBRARIES
import streamlit as st
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

# STEP 2: LOAD MODELS

@st.cache_resource(show_spinner=True)
def load_models():
    # Sentiment analysis model
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Emotion detection model
    emotion_detector = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    
    # Summarization model fix for Transformers 5.x
    summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer_model_obj = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer = pipeline(
        "text2text-generation",
        model=summarizer_model_obj,
        tokenizer=summarizer_tokenizer
    )

    return sentiment_analyzer, emotion_detector, summarizer

sentiment_model, emotion_model, summarizer_model = load_models()


# STEP 3: HELPER FUNCTIONS

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return result['label'], result['score']

def detect_emotion(text):
    scores = emotion_model(text)[0]
    top_emotion = max(scores, key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score']

def generate_summary(text, max_length=60, min_length=25):
    summary = summarizer_model(f"summarize: {text}", max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['generated_text']

# Highlight words
positive_words = ['love','great','excellent','good','amazing','happy','fantastic','awesome']
negative_words = ['hate','bad','terrible','awful','worst','sad','disappointing','angry']

def highlight_words(text):
    words = text.split()
    highlighted = []
    for w in words:
        lw = w.lower().strip('.,!?')
        if lw in positive_words:
            highlighted.append(f"<span style='color:#00FF00;font-weight:bold'>{w}</span>")
        elif lw in negative_words:
            highlighted.append(f"<span style='color:#FF4500;font-weight:bold'>{w}</span>")
        else:
            highlighted.append(w)
    return " ".join(highlighted)

# Emojis
sentiment_emojis = {"POSITIVE":"😁","NEGATIVE":"😡","NEUTRAL":"😐"}
emotion_emojis = {"joy":"😊","anger":"😡","fear":"😨","surprise":"😲","sadness":"😢","neutral":"😐"}

def display_box(label, score, emoji, color):
    st.markdown(
        f"<div style='border-radius:10px; padding:10px; margin:5px 0; background-color:{color}; color:white; font-weight:bold; font-size:18px'>{emoji} {label} ({score:.2f})</div>",
        unsafe_allow_html=True
    )


# STEP 4: STREAMLIT UI - Dark Mode

st.set_page_config(page_title="AI Sentiment & Emotion Analyzer", page_icon="🖤", layout="wide")

st.markdown("""
<style>
.reportview-container, .main { background-color: #121212; color: #FFFFFF; }
.css-1d391kg { background-color: #1E1E1E; }
.stButton>button { background-color: #1E90FF; color: white; font-weight: bold; }
.stDataFrame table { color: white; background-color: #1E1E1E; }
span { font-weight:bold; }
h1, h2, h3 { color:#00FFFF; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("AI Sentiment Analyzer 🌑")
st.sidebar.markdown("Analyze text sentiment, emotions, and generate summaries in Dark Mode!")
mode = st.sidebar.selectbox("Select Mode", ["Single Text Analysis", "Multi Text Analysis", "CSV Upload Analysis"])

st.title("Sentiment & Emotion Analyzer (Dark Mode)")


# SESSION STATE HISTORY

if 'history' not in st.session_state:
    st.session_state['history'] = []


# SINGLE TEXT ANALYSIS

if mode == "Single Text Analysis":
    user_input = st.text_area("Enter text here:", height=150)

    # Live typing analysis
    if user_input.strip() != "":
        sentiment_label, sentiment_score = analyze_sentiment(user_input)
        emotion_label, emotion_score = detect_emotion(user_input)
        summary = generate_summary(user_input)
        highlighted_text = highlight_words(user_input)

        # Store history
        st.session_state.history.append({
            "Text": user_input,
            "Sentiment": f"{sentiment_emojis.get(sentiment_label,'')} {sentiment_label} ({sentiment_score:.2f})",
            "Emotion": f"{emotion_emojis.get(emotion_label,'')} {emotion_label} ({emotion_score:.2f})",
            "Summary": summary
        })

        st.subheader("Analysis Results")
        display_box(sentiment_label, sentiment_score, sentiment_emojis.get(sentiment_label,""),
                    "#008000" if sentiment_label=="POSITIVE" else "#FF4500" if sentiment_label=="NEGATIVE" else "#808080")
        display_box(emotion_label, emotion_score, emotion_emojis.get(emotion_label,""), "#1E90FF")
        st.markdown("### Summary:")
        st.markdown(summary)
        st.markdown("### Highlighted Text:")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # History
        st.markdown("### Analysis History")
        st.dataframe(pd.DataFrame(st.session_state.history))


# MULTI-TEXT ANALYSIS

if mode == "Multi Text Analysis":
    multi_input = st.text_area("Enter multiple sentences (one per line):", height=200)
    if st.button("Analyze Multi Text"):
        if multi_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            lines = [line for line in multi_input.split("\n") if line.strip() != ""]
            results = []
            for line in lines:
                sentiment, s_score = analyze_sentiment(line)
                emotion, e_score = detect_emotion(line)
                summary = generate_summary(line)
                results.append({"Text":line,
                                "Sentiment":f"{sentiment_emojis.get(sentiment,'')} {sentiment} ({s_score:.2f})",
                                "Emotion":f"{emotion_emojis.get(emotion,'')} {emotion} ({e_score:.2f})",
                                "Summary":summary})
            df_results = pd.DataFrame(results)
            st.markdown("### Analysis Results")
            st.dataframe(df_results)

           
            # Advanced Charts
           
            st.markdown("### Sentiment Distribution")
            sns.set_style("darkgrid")
            fig, ax = plt.subplots()
            df_results['Sentiment_Label'] = df_results['Sentiment'].apply(lambda x: x.split()[1])
            sns.countplot(x='Sentiment_Label', data=df_results, palette="viridis", ax=ax)
            st.pyplot(fig)

            st.markdown("### Emotion Word Cloud")
            all_emotions = " ".join(df_results['Emotion_Label'].tolist())
            wc = WordCloud(width=800, height=400, background_color='black', colormap='plasma').generate(all_emotions)
            fig2, ax2 = plt.subplots(figsize=(10,5))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)


# CSV UPLOAD ANALYSIS

if mode == "CSV Upload Analysis":
    uploaded_file = st.file_uploader("Upload a CSV file (must have a 'text' column)", type=["csv"])
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        if 'text' not in df_csv.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            results_csv = []
            for idx, row in df_csv.iterrows():
                text = str(row['text'])
                sentiment, s_score = analyze_sentiment(text)
                emotion, e_score = detect_emotion(text)
                summary = generate_summary(text)
                results_csv.append({
                    "Text": text,
                    "Sentiment": f"{sentiment_emojis.get(sentiment,'')} {sentiment} ({s_score:.2f})",
                    "Emotion": f"{emotion_emojis.get(emotion,'')} {emotion} ({e_score:.2f})",
                    "Summary": summary
                })
            df_results_csv = pd.DataFrame(results_csv)
            st.markdown("### CSV Analysis Results")
            st.dataframe(df_results_csv)

            towrite = io.BytesIO()
            df_results_csv.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button(label="Download Results CSV", data=towrite,
                               file_name="sentiment_analysis_results.csv", mime="text/csv")