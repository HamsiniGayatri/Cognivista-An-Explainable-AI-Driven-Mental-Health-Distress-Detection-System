import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from textblob import TextBlob

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="XAI Mental Health Distress Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 XAI Mental Health Distress Detector")
st.markdown(
    """
This application utilizes a **Hybrid NLP Machine Learning Model** combined with 
**Explainable AI (XAI)** to screen text inputs for emotional distress indicators.
"""
)
st.caption("⚠️ **Disclaimer:** This application is intended for educational and screening support only and does not provide a formal medical diagnosis.")

# ============================================================
# 2. LOAD TRAINED MODEL ARTIFACTS
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "Models")

@st.cache_resource
def load_ml_components():
    try:
        tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        hybrid_model = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.pkl"))
        return tfidf, scaler, hybrid_model
    except Exception as e:
        st.error(f"Error loading model files from `{MODELS_DIR}`: {e}")
        return None, None, None

tfidf, scaler, hybrid_model = load_ml_components()

# Feature columns sequence expected by the scaler
FEATURE_COLUMNS = [
    "text_length", "word_count", "num_urls", "num_emojis", 
    "num_special_chars", "num_excess_punct", "avg_word_length", 
    "stopword_ratio", "type_token_ratio", "polarity", "subjectivity", 
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", 
    "has_suicidal_keyword", "has_stress_keyword", "has_help_keyword"
]

# ============================================================
# 3. FEATURE EXTRACTION PIPELINE
# ============================================================
def extract_features_from_text(text):
    words = text.split()
    word_count = len(words)
    text_length = len(text)

    num_urls = len(re.findall(r"https?://\S+|www\.\S+", text))
    num_emojis = len(re.findall(r"[^\x00-\x7F]", text))
    num_special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    num_excess_punct = len(re.findall(r"[!?.,]{2,}", text))
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    stopwords = {
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", 
        "it", "they", "the", "a", "an", "and", "or", "but", "if", "is", "are", 
        "was", "were", "to", "of", "in", "on", "for", "with", "at"
    }

    stopword_count = sum(1 for word in words if word.lower().strip(".,!?") in stopwords)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0

    unique_words = set(word.lower() for word in words)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        pos_tags = blob.tags
        total_tags = len(pos_tags)

        if total_tags > 0:
            noun_ratio = sum(1 for _, tag in pos_tags if tag.startswith(("NN", "NP"))) / total_tags
            verb_ratio = sum(1 for _, tag in pos_tags if tag.startswith("VB")) / total_tags
            adj_ratio = sum(1 for _, tag in pos_tags if tag.startswith("JJ")) / total_tags
            adv_ratio = sum(1 for _, tag in pos_tags if tag.startswith("RB")) / total_tags
        else:
            noun_ratio = verb_ratio = adj_ratio = adv_ratio = 0
    except Exception:
        polarity = subjectivity = noun_ratio = verb_ratio = adj_ratio = adv_ratio = 0

    text_lower = text.lower()
    suicidal_keywords = ["suicide", "suicidal", "kill myself", "end my life", "take my life", "want to die", "wish i was dead", "overdose", "self harm"]
    stress_keywords = ["stress", "stressed", "overwhelmed", "pressure", "anxious", "anxiety", "worried", "panic", "tired", "exhausted"]
    help_keywords = ["help", "therapy", "therapist", "counselling", "counseling", "support", "doctor", "mental health"]

    has_suicidal_keyword = int(any(k in text_lower for k in suicidal_keywords))
    has_stress_keyword = int(any(k in text_lower for k in stress_keywords))
    has_help_keyword = int(any(k in text_lower for k in help_keywords))

    features_df = pd.DataFrame(
        [[
            text_length, word_count, num_urls, num_emojis, num_special_chars, 
            num_excess_punct, avg_word_length, stopword_ratio, type_token_ratio, 
            polarity, subjectivity, noun_ratio, verb_ratio, adj_ratio, adv_ratio, 
            has_suicidal_keyword, has_stress_keyword, has_help_keyword
        ]],
        columns=FEATURE_COLUMNS
    )
    return features_df

# ============================================================
# 4. REMEDIAL GUIDANCE GENERATOR
# ============================================================
def generate_suggestions(predicted_class):
    if predicted_class == "Suicidal":
        return [
            "🚨 **Immediate Support:** If you or someone you know is struggling or in crisis, help is available. Reach out to local emergency services immediately.",
            "📞 Connect with a 24/7 crisis helpline or professional support network.",
            "🤝 Talk confidentially to someone you trust about what you are experiencing."
        ]
    elif predicted_class == "Depression":
        return [
            "💬 Consider sharing what you are going through with a friend, family member, or healthcare professional.",
            "🌱 Focus on minimal, achievable daily goals rather than trying to carry everything at once.",
            "🚶 Gentle physical activity and stable sleep routines can provide steady grounding."
        ]
    elif predicted_class == "Anxiety":
        return [
            "🫁 Practice slow, deep-breathing cycles or mindful grounding techniques to center yourself.",
            "🧘 Segment stressful responsibilities into smaller, predictable action items.",
            "💤 Prioritize intentional rest and mental breaks throughout your schedule."
        ]
    else:
        return [
            "✅ Text patterns reflect minimal current distress signals.",
            "☀️ Continue prioritizing self-care, consistent wellness habits, and social support systems."
        ]

# ============================================================
# 5. USER INTERFACE LAYOUT & EVALUATION
# ============================================================
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📥 Input Content")
    user_input = st.text_area(
        "Enter text or post content for distress analysis:",
        height=220,
        placeholder="Type or paste context here..."
    )
    analyze_btn = st.button("🔍 Run Text Analysis", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.warning("Please provide valid text before executing analysis.")
    elif tfidf is None or scaler is None or hybrid_model is None:
        st.error("Missing model pipeline artifacts. Please check that `.pkl` files exist inside the `Models` directory.")
    else:
        # Pipeline Feature Transformations
        text_tfidf = tfidf.transform([user_input])
        features_df = extract_features_from_text(user_input)
        features_scaled = scaler.transform(features_df)
        hybrid_input = hstack([text_tfidf, features_scaled]).tocsr()

        # Model Inference
        prediction = hybrid_model.predict(hybrid_input)[0]
        probs = hybrid_model.predict_proba(hybrid_input)[0]
        class_probs = dict(zip(hybrid_model.classes_, probs))
        confidence = max(probs)

        with col2:
            st.subheader("📊 Screening Results")
            
            color_map = {
                "Normal": "green",
                "Anxiety": "orange",
                "Depression": "orange",
                "Suicidal": "red"
            }
            badge_color = color_map.get(prediction, "blue")
            
            st.markdown(f"### Predicted Classification: :{badge_color}[**{prediction}**]")
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")

            st.markdown("#### Probability Distribution")
            for cls_name, prob in class_probs.items():
                st.write(f"**{cls_name}**: {prob*100:.1f}%")
                st.progress(float(prob))

        # ============================================================
        # 6. EXPLAINABLE AI (XAI) & SUPPORT ADVICE TABS
        # ============================================================
        st.divider()
        st.subheader("💡 Explainable AI (XAI) & Insights")

        tab_xai, tab_sug = st.tabs(["🔍 Model Explanation (Feature Attribution)", "🌱 Actionable Guidance"])

        with tab_xai:
            try:
                tfidf_names = tfidf.get_feature_names_out().tolist()
                all_feature_names = tfidf_names + FEATURE_COLUMNS
                cls_idx = list(hybrid_model.classes_).index(prediction)
                coefficients = hybrid_model.coef_[cls_idx]
                sample_vec = hybrid_input.toarray()[0]
                contributions = sample_vec * coefficients

                explanation_df = pd.DataFrame({
                    "Feature": all_feature_names,
                    "Contribution": contributions
                })

                explanation_df = explanation_df[explanation_df["Contribution"] != 0]
                explanation_df["Abs_Contrib"] = explanation_df["Contribution"].abs()
                explanation_df = explanation_df.sort_values(by="Abs_Contrib", ascending=False)

                supporting = explanation_df[explanation_df["Contribution"] > 0].head(5)
                opposing = explanation_df[explanation_df["Contribution"] < 0].head(5)

                col_sup, col_opp = st.columns(2)
                with col_sup:
                    st.markdown("##### 🟢 Features Supporting Prediction")
                    if not supporting.empty:
                        st.dataframe(supporting[["Feature", "Contribution"]], use_container_width=True)
                    else:
                        st.write("No major supporting factors found.")

                with col_opp:
                    st.markdown("##### 🔴 Features Opposing Prediction")
                    if not opposing.empty:
                        st.dataframe(opposing[["Feature", "Contribution"]], use_container_width=True)
                    else:
                        st.write("No major opposing factors found.")
            except Exception:
                st.info("Local attribution explanation unavailable for this configuration sample.")

        with tab_sug:
            suggestions = generate_suggestions(prediction)
            for item in suggestions:
                st.markdown(f"- {item}")