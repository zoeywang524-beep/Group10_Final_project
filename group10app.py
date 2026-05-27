import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import io
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, VitsModel
from huggingface_hub import login

st.set_page_config(page_title="GamePulse AI", page_icon="🎮", layout="wide")

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500;600&display=swap");
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.game-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
}
.game-header h1 {
    font-family: "Rajdhani", sans-serif; font-size: 3rem; font-weight: 700;
    color: #fff; letter-spacing: 3px; margin: 0;
    text-shadow: 0 0 30px rgba(99,202,255,0.6);
}
.game-header p { color: rgba(255,255,255,0.6); font-size: 1rem; margin-top: 0.5rem; }
.game-header .badge {
    display: inline-block; background: rgba(99,202,255,0.15);
    border: 1px solid rgba(99,202,255,0.4); color: #63caff;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.75rem; letter-spacing: 1px; margin-top: 0.6rem;
}
.step-label {
    font-family: "Rajdhani", sans-serif; font-size: 1.4rem; font-weight: 700;
    letter-spacing: 1.5px; color: #63caff; text-transform: uppercase; margin-bottom: 0.3rem;
}
.result-card {
    background: #1a1a2e; border: 1px solid rgba(99,202,255,0.2);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; color: #e0e0e0;
}
.result-card:hover { border-color: rgba(99,202,255,0.5); }
.report-box {
    background: #0d1117; border-left: 4px solid #63caff;
    border-radius: 8px; padding: 1.2rem 1.5rem;
    color: #c9d1d9; font-size: 0.95rem; line-height: 1.75;
}
.rec-box { border-radius: 10px; padding: 1rem 1.4rem; font-size: 0.95rem; line-height: 1.7; }
.stat-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem; color: #c9d1d9; }
.runtime-box {
    background: #161b22; border: 1px solid rgba(99,202,255,0.2);
    border-radius: 10px; padding: 1rem; text-align: center;
}
.runtime-box .label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.runtime-box .value { font-family: "Rajdhani", sans-serif; font-size: 2.2rem; font-weight: 700; color: #63caff; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_MODEL_NAME = "zoeywww/cardiffnlp-sentiment-3class-finetuned"
TTS_MODEL_NAME = "kakao-enterprise/vits-ljs"
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_EMOJI = {"Positive": "✅", "Negative": "❌", "Neutral": "➖"}
LABEL_COLOR = {"Positive": "#2ea043", "Negative": "#da3633", "Neutral": "#e3b341"}

if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])


def get_recommendation(counts, total):
    neg_pct = counts.get("Negative", 0) / total * 100
    pos_pct = counts.get("Positive", 0) / total * 100
    neu_pct = counts.get("Neutral", 0) / total * 100
    if neg_pct >= 40:
        return ("🚨", "#da3633", "Urgent Issue Review",
                "Negative feedback accounts for " + str(round(neg_pct)) + "% of comments. "
                "The community management team should immediately investigate recurring player concerns "
                "and prioritise fixes in the next patch cycle.")
    elif pos_pct >= 60:
        return ("🎉", "#2ea043", "Healthy Community Response",
                "Positive feedback accounts for " + str(round(pos_pct)) + "% of comments. "
                "The team may leverage these highlights in the next community update, "
                "social channels, or marketing summary.")
    elif neu_pct >= 60:
        return ("📊", "#e3b341", "Stable — Monitor Normally",
                "Neutral feedback dominates at " + str(round(neu_pct)) + "%. "
                "No significant emotional signal detected. Maintain standard monitoring cadence "
                "and watch for emerging trends.")
    else:
        dominant = max(counts, key=counts.get)
        dominant_pct = round(counts[dominant] / total * 100)
        return ("📋", "#58a6ff", "Mixed Community Reaction",
                "Sentiment is distributed across categories. Dominant signal: "
                + dominant + " (" + str(dominant_pct) + "%). "
                "Conduct a deeper qualitative review of specific feedback threads.")


def generate_written_report(counts, total, dominant):
    neg_pct = counts.get("Negative", 0) / total * 100
    pos_pct = counts.get("Positive", 0) / total * 100
    neu_pct = counts.get("Neutral", 0) / total * 100
    neg_c = counts.get("Negative", 0)
    pos_c = counts.get("Positive", 0)
    neu_c = counts.get("Neutral", 0)
    comment_word = "comments" if total > 1 else "comment"
    if dominant == "Positive":
        tone = "predominantly positive"
        action = ("The community team is advised to amplify player highlights through official channels "
                  "and maintain current content and update cadence.")
    elif dominant == "Negative":
        tone = "predominantly negative"
        action = ("Immediate attention is recommended. The product team should triage top complaints, "
                  "communicate a response roadmap, and prioritise high-impact fixes in the upcoming patch.")
    elif dominant == "Neutral":
        tone = "largely neutral"
        action = ("No urgent action required. Standard monitoring is advised, with particular attention "
                  "to shifts in sentiment following forthcoming updates.")
    else:
        tone = "mixed"
        action = ("A targeted qualitative review of negative threads is recommended alongside continued "
                  "monitoring of positive engagement signals.")
    dist = ("Positive: " + str(pos_c) + " (" + str(round(pos_pct)) + "%) | "
            "Negative: " + str(neg_c) + " (" + str(round(neg_pct)) + "%) | "
            "Neutral: " + str(neu_c) + " (" + str(round(neu_pct)) + "%)")
    dominant_pct = round(counts[dominant] / total * 100)
    report = (
        "Community Sentiment Report\n\n"
        "Analysis Date: " + time.strftime("%B %d, %Y") + "\n"
        "Total Comments Analysed: " + str(total) + "\n"
        "Sentiment Distribution: " + dist + "\n\n"
        "Summary: Based on " + str(total) + " player " + comment_word + " collected from the gaming community, "
        "overall sentiment is " + tone + ". The dominant category is " + dominant + ", accounting for "
        + str(dominant_pct) + "% of all feedback.\n\n"
        "Recommended Action: " + action + "\n\n"
        "This report was automatically generated by GamePulse AI for internal community team review. "
        "For full comment-level detail, refer to the classification table above."
    )
    return report


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    if "HF_TOKEN" in st.secrets:
        login(token=st.secrets["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, token=st.secrets.get("HF_TOKEN"))
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME, token=st.secrets.get("HF_TOKEN"))
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading TTS model...")
def load_tts_model():
    if "HF_TOKEN" in st.secrets:
        login(token=st.secrets["HF_TOKEN"])
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME, token=st.secrets.get("HF_TOKEN"))
    tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME, token=st.secrets.get("HF_TOKEN"))
    tts_model.eval()
    return tts_tokenizer, tts_model


def classify_sentiment(texts, tokenizer, model):
    results = []
    start_time = time.time()
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
        pred_id = int(np.argmax(probs))
        label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id]) * 100
        results.append({"comment": text, "sentiment": label, "confidence": confidence})
    runtime = time.time() - start_time
    return results, runtime


def generate_speech(text, tts_tokenizer, tts_model):
    start_time = time.time()
    plain = text.replace("**", "").replace("*", "").replace("#", "")
    inputs = tts_tokenizer(plain, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs)
    waveform = output.waveform[0].numpy()
    sampling_rate = tts_model.config.sampling_rate
    waveform_int16 = (waveform * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sampling_rate, waveform_int16)
    audio_bytes = buffer.getvalue()
    runtime = time.time() - start_time
    return audio_bytes, runtime


# ── Header
st.markdown("""
<div class="game-header">
  <h1>🎮 GAMEPULSE AI</h1>
  <p>Gaming Community Sentiment Voice Reporter</p>
  <span class="badge">ISOM5240 &nbsp;·&nbsp; NLP PIPELINE &nbsp;·&nbsp; POWERED BY TRANSFORMERS</span>
</div>
""", unsafe_allow_html=True)

# ── Step 1
st.markdown("<div class='step-label'>📝 Step 1 — Enter Player Comments</div>", unsafe_allow_html=True)
st.markdown("Paste player feedback below — one comment per line. Supports reviews, forum posts, and in-game chat exports.")

default_text = (
    "The graphics update looks amazing, but matchmaking is still terrible.\n"
    "I love the new event rewards and the seasonal battlepass!\n"
    "The server lag makes ranked mode completely unplayable.\n"
    "The patch is okay, nothing that exciting honestly.\n"
    "Finally fixed the hitbox bug — best update in months."
)

user_input = st.text_area(label="Player Comments", value=default_text, height=180, label_visibility="collapsed")

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_button = st.button("🚀  ANALYSE COMMENTS", type="primary", use_container_width=True)
with col_info:
    st.caption("Pipeline 1: Fine-tuned RoBERTa (3-class sentiment)  ·  Pipeline 2: VITS Text-to-Speech")

    unsafe_allow_html=True)
