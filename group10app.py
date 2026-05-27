import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import io
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, VitsModel
from huggingface_hub import login

st.set_page_config(page_title="EcoPulse AI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500;600&display=swap");
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.app-header {
    background: linear-gradient(135deg, #0b192c, #1a365d, #004b49);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
}
.app-header h1 {
    font-family: "Rajdhani", sans-serif; font-size: 3rem; font-weight: 700;
    color: #fff; letter-spacing: 3px; margin: 0;
    text-shadow: 0 0 30px rgba(0, 210, 255, 0.6);
}
.app-header p { color: rgba(255,255,255,0.7); font-size: 1rem; margin-top: 0.5rem; }
.app-header .badge {
    display: inline-block; background: rgba(0, 210, 255, 0.15);
    border: 1px solid rgba(0, 210, 255, 0.4); color: #00d2ff;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.75rem; letter-spacing: 1px; margin-top: 0.6rem;
}
.step-label {
    font-family: "Rajdhani", sans-serif; font-size: 1.4rem; font-weight: 700;
    letter-spacing: 1.5px; color: #00d2ff; text-transform: uppercase; margin-bottom: 0.3rem;
}
.result-card {
    background: #111827; border: 1px solid rgba(0, 210, 255, 0.2);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; color: #e5e7eb;
}
.result-card:hover { border-color: rgba(0, 210, 255, 0.5); }
.report-box {
    background: #0f172a; border-left: 4px solid #00d2ff;
    border-radius: 8px; padding: 1.2rem 1.5rem;
    color: #cbd5e1; font-size: 0.95rem; line-height: 1.75;
}
.rec-box { border-radius: 10px; padding: 1rem 1.4rem; font-size: 0.95rem; line-height: 1.7; }
.stat-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem; color: #cbd5e1; }
.runtime-box {
    background: #1e293b; border: 1px solid rgba(0, 210, 255, 0.2);
    border-radius: 10px; padding: 1rem; text-align: center;
}
.runtime-box .label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.runtime-box .value { font-family: "Rajdhani", sans-serif; font-size: 2.2rem; font-weight: 700; color: #00d2ff; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_MODEL_NAME = "zoeywww/cardiffnlp-sentiment-3class-finetuned"
TTS_MODEL_NAME = "kakao-enterprise/vits-ljs"
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_EMOJI = {"Positive": "✅", "Negative": "🚨", "Neutral": "➖"}
LABEL_COLOR = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}

if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])


def get_recommendation(counts, total):
    neg_pct = counts.get("Negative", 0) / total * 100
    pos_pct = counts.get("Positive", 0) / total * 100
    neu_pct = counts.get("Neutral", 0) / total * 100
    if neg_pct >= 40:
        return ("🚨", "#ef4444", "Urgent Service Escalation",
                "Negative feedback accounts for " + str(round(neg_pct)) + "% of today's comments. "
                "The customer recovery team should immediately review recent orders for potential logistics delays, "
                "damaged packaging, or widespread defect issues. Prioritize refunds and outbound outreach.")
    elif pos_pct >= 60:
        return ("🎉", "#10b981", "Healthy Customer Satisfaction",
                "Positive feedback accounts for " + str(round(pos_pct)) + "% of comments. "
                "Customer satisfaction remains high. The marketing team may leverage these highlights "
                "for promotional materials and maintain current service level agreements (SLAs).")
    elif neu_pct >= 60:
        return ("📊", "#f59e0b", "Stable Operations — Monitor Normally",
                "Neutral feedback dominates at " + str(round(neu_pct)) + "%. "
                "Most reviews reflect standard transactional inquiries with no significant emotional signal. "
                "Maintain standard monitoring cadence.")
    else:
        dominant = max(counts, key=counts.get)
        dominant_pct = round(counts[dominant] / total * 100)
        return ("📋", "#3b82f6", "Mixed Operational Feedback",
                "Sentiment is distributed across categories. Dominant signal: "
                + dominant + " (" + str(dominant_pct) + "%). "
                "Recommend filtering by product category to identify if dissatisfaction is isolated to specific SKUs.")


def generate_written_report(counts, total, dominant):
    neg_pct = counts.get("Negative", 0) / total * 100
    pos_pct = counts.get("Positive", 0) / total * 100
    neu_pct = counts.get("Neutral", 0) / total * 100
    neg_c = counts.get("Negative", 0)
    pos_c = counts.get("Positive", 0)
    neu_c = counts.get("Neutral", 0)
    comment_word = "reviews" if total > 1 else "review"

    if dominant == "Positive":
        tone = "predominantly positive"
        action = ("The customer service team is advised to acknowledge highly rated reviews "
                  "and monitor for ongoing product quality consistency.")
    elif dominant == "Negative":
        tone = "predominantly negative"
        action = ("Immediate attention is required. The support team must triage top complaints, "
                  "cross-check warehouse and shipping logs, and escalate critical product defects to procurement.")
    elif dominant == "Neutral":
        tone = "largely neutral"
        action = ("No urgent intervention is required. Continue to meet standard response time metrics.")
    else:
        tone = "mixed"
        action = ("A targeted review of negative tickets is recommended, cross-referencing specific SKUs "
                  "or regional delivery hubs to isolate operational bottlenecks.")

    dist = ("Positive: " + str(pos_c) + " (" + str(round(pos_pct)) + "%) | "
            "Negative: " + str(neg_c) + " (" + str(round(neg_pct)) + "%) | "
            "Neutral: " + str(neu_c) + " (" + str(round(neu_pct)) + "%)")
    dominant_pct = round(counts[dominant] / total * 100)

    report = (
        "E-Commerce Customer Sentiment Report\n\n"
        "Analysis Date: " + time.strftime("%B %d, %Y") + "\n"
        "Total Reviews Analysed: " + str(total) + "\n"
        "Sentiment Distribution: " + dist + "\n\n"
        "Summary: Based on " + str(total) + " customer " + comment_word + " processed today, "
        "overall sentiment is " + tone + ". The dominant category is " + dominant + ", accounting for "
        + str(dominant_pct) + "% of total feedback.\n\n"
        "Recommended Action: " + action + "\n\n"
        "This report was automatically generated by EcoPulse AI for the Customer Support Management Team. "
        "For ticket-level detail, refer to the dashboard classification logs."
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
    plain = plain[:500]
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


with st.spinner("Initialising AI models... Please wait on first load."):
    sent_tokenizer, sent_model = load_sentiment_model()
    tts_tokenizer, tts_model = load_tts_model()

st.markdown("""
<div class="app-header">
  <h1>🛒 EcoPulse AI</h1>
  <p>E-Commerce Customer Support Voice Reporter</p>
  <span class="badge">ISOM5240 &nbsp;·&nbsp; NLP PIPELINE &nbsp;·&nbsp; POWERED BY TRANSFORMERS</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='step-label'>📝 Step 1 — Enter Customer Reviews</div>", unsafe_allow_html=True)
st.markdown("Paste Amazon customer feedback below — one review per line. Supports product reviews, support tickets, and post-purchase surveys.")

default_text = (
    "The shipping was incredibly fast and the packaging kept everything safe. Five stars!\n"
    "Item arrived defective. Customer service is completely ignoring my emails.\n"
    "The product works as described, but the setup instructions were a bit confusing.\n"
    "I received the wrong color, and now I have to pay for return shipping. Terrible experience.\n"
    "Great value for the price. I've already recommended this to my friends."
)

user_input = st.text_area(label="Customer Reviews", value=default_text, height=180, label_visibility="collapsed")

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_button = st.button("🚀  ANALYSE REVIEWS", type="primary", use_container_width=True)
with col_info:
    st.caption("Pipeline 1: Fine-tuned Sentiment Classifier  ·  Pipeline 2: VITS Text-to-Speech")

st.divider()

if run_button:
    raw_comments = [c.strip() for c in user_input.strip().split("\n") if c.strip()]
    if len(raw_comments) == 0:
        st.warning("Please enter at least one review.")
        st.stop()

    st.markdown("<div class='step-label'>📊 Step 2 — Sentiment Classification</div>", unsafe_allow_html=True)
    st.caption("Model: " + SENTIMENT_MODEL_NAME)
    with st.spinner("Classifying customer reviews..."):
        sentiment_results, p1_runtime = classify_sentiment(raw_comments, sent_tokenizer, sent_model)

    for i, res in enumerate(sentiment_results):
        color = LABEL_COLOR.get(res["sentiment"], "#3b82f6")
        emoji = LABEL_EMOJI.get(res["sentiment"], "")
        bar_pct = round(res["confidence"])
        st.markdown(
            "<div class='result-card'>"
            "<div style='display:flex; justify-content:space-between; align-items:center;'>"
            "<span style='color:#e5e7eb;'><b>" + str(i+1) + ".</b> " + res["comment"] + "</span>"
            "<span style='color:" + color + "; font-weight:600; white-space:nowrap; margin-left:1rem;'>"
            + emoji + " " + res["sentiment"] + " (" + str(bar_pct) + "%)</span>"
            "</div></div>",
            unsafe_allow_html=True)

    st.caption("⏱ Classification runtime: " + str(round(p1_runtime, 3)) + "s for " + str(len(raw_comments)) + " reviews")

    df = pd.DataFrame(sentiment_results)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Analysis Results (CSV)", csv, "ecopulse_review_analysis.csv", "text/csv")

    st.divider()

    st.markdown("<div class='step-label'>📈 Step 3 — Customer Sentiment Overview</div>", unsafe_allow_html=True)
    counts = {}
    for res in sentiment_results:
        lbl = res["sentiment"]
        counts[lbl] = counts.get(lbl, 0) + 1
    total = len(sentiment_results)
    dominant = max(counts, key=counts.get)

    col_stats, col_chart = st.columns([1, 1])
    with col_stats:
        for lbl in ["Positive", "Negative", "Neutral"]:
            count = counts.get(lbl, 0)
            pct = count / total * 100
            color = LABEL_COLOR.get(lbl, "#3b82f6")
            emoji = LABEL_EMOJI.get(lbl, "")
            st.markdown(
                "<div class='stat-row'>"
                "<span style='width:90px; color:" + color + "; font-weight:600;'>" + emoji + " " + lbl + "</span>"
                "<div style='flex:1; background:#1e293b; border-radius:6px; height:10px;'>"
                "<div style='width:" + str(round(pct)) + "%; background:" + color + "; height:10px; border-radius:6px;'></div>"
                "</div>"
                "<span style='width:60px; text-align:right; color:#94a3b8;'>"
                + str(count) + " (" + str(round(pct)) + "%)</span>"
                "</div>",
                unsafe_allow_html=True)
    with col_chart:
        chart_df = pd.DataFrame({"Count": list(counts.values())}, index=list(counts.keys()))
        st.bar_chart(chart_df, color="#00d2ff")
    st.divider()

    st.markdown("<div class='step-label'>💼 Step 4 — Support Team Action Plan</div>", unsafe_allow_html=True)
    rec_icon, rec_color, rec_title, rec_body = get_recommendation(counts, total)
    st.markdown(
        "<div class='rec-box' style='background:" + rec_color + "18; border:1px solid " + rec_color + "55;'>"
        "<div style='font-weight:700; font-size:1rem; color:" + rec_color + "; margin-bottom:0.4rem;'>"
        + rec_icon + " " + rec_title + "</div>"
        "<div style='color:#cbd5e1;'>" + rec_body + "</div>"
        "</div>",
        unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='step-label'>📄 Step 5 — Written Daily Briefing</div>", unsafe_allow_html=True)
    written_report = generate_written_report(counts, total, dominant)
    report_html = written_report.replace("\n", "<br>")
    st.markdown("<div class='report-box'>" + report_html + "</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='step-label'>🔊 Step 6 — Audio Dashboard Brief (Pipeline 2)</div>", unsafe_allow_html=True)
    st.caption("Model: " + TTS_MODEL_NAME + "  ·  Generating spoken version of the daily support report")
    with st.spinner("Synthesising audio briefing..."):
        audio_bytes, p2_runtime = generate_speech(written_report, tts_tokenizer, tts_model)
    st.audio(audio_bytes, format="audio/wav")
    st.caption("⏱ TTS runtime: " + str(round(p2_runtime, 2)) + "s")
    st.divider()

    st.markdown("<div class='step-label'>⚡ Pipeline Runtime Summary</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(
            "<div class='runtime-box'>"
            "<div class='label'>Pipeline 1 · Sentiment Classification</div>"
            "<div class='value'>" + str(round(p1_runtime, 3)) + "s</div>"
            "<div style='color:#94a3b8; font-size:0.78rem;'>" + SENTIMENT_MODEL_NAME + "</div>"
            "</div>",
            unsafe_allow_html=True)
    with col_p2:
        st.markdown(
            "<div class='runtime-box'>"
            "<div class='label'>Pipeline 2 · Text-to-Speech</div>"
            "<div class='value'>" + str(round(p2_runtime, 2)) + "s</div>"
            "<div style='color:#94a3b8; font-size:0.78rem;'>" + TTS_MODEL_NAME + "</div>"
            "</div>",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.78rem; padding:1rem 0;'>"
    "EcoPulse AI &nbsp;·&nbsp; ISOM5240 Group Project &nbsp;·&nbsp; "
    "Pipeline 1: Text Sentiment Classification &nbsp;·&nbsp; Pipeline 2: VITS TTS"
    "</div>",
    unsafe_allow_html=True)
