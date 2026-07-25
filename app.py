import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from textblob import TextBlob

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Mental Health Distress Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make metrics and containers look more professional
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 5% 10%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CONSTANTS & KEYWORDS (Untouched)
# ============================================================
FEATURE_COLUMNS = [
    "text_length", "word_count", "num_urls", "num_emojis", 
    "num_special_chars", "num_excess_punct", "avg_word_length", 
    "stopword_ratio", "type_token_ratio", "polarity", 
    "subjectivity", "noun_ratio", "verb_ratio", "adj_ratio", 
    "adv_ratio", "has_suicidal_keyword", "has_stress_keyword", 
    "has_help_keyword"
]

SUICIDAL_KEYWORDS = ["suicide", "suicidal", "kill myself", "end my life", "take my life", "want to die", "wish i was dead", "overdose", "self harm"]
STRESS_KEYWORDS = ["stress", "stressed", "overwhelmed", "pressure", "anxious", "anxiety", "worried", "panic", "tired", "exhausted"]
HELP_KEYWORDS = ["help", "therapy", "therapist", "counselling", "counseling", "support", "doctor", "mental health"]

# ============================================================
# 3. LOAD MODELS (Untouched)
# ============================================================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "Models")
    
    try:
        tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        hybrid_model = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.pkl"))
        return tfidf, scaler, hybrid_model
    except Exception as e:
        st.error(f"Error loading models from {MODELS_DIR}: {e}")
        return None, None, None

tfidf, scaler, hybrid_model = load_models()

# ============================================================
# 4. FEATURE EXTRACTION (Untouched)
# ============================================================
def extract_features_from_text(text):
    text = text.strip()
    words = text.split()
    word_count = len(words)
    text_length = len(text)

    num_urls = len(re.findall(r"https?://\S+|www\.\S+", text))
    num_emojis = len(re.findall(r"[^\x00-\x7F]", text))
    num_special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    num_excess_punct = len(re.findall(r"[!?.,]{2,}", text))

    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    stopwords = {"i","me","my","myself","we","our","you","your","he","she","it","they","the","a","an","and","or","but","if","is","are","was","were","to","of","in","on","for","with","at"}
    stopword_count = sum(1 for word in words if word.lower().strip(string.punctuation) in stopwords) if word_count > 0 else 0
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0

    unique_words = set(word.lower() for word in words)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0

    try:
        blob = TextBlob(text)
        polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        blob, polarity, subjectivity = None, 0, 0

    noun_ratio = verb_ratio = adj_ratio = adv_ratio = 0
    try:
        if blob:
            pos_tags = blob.tags
            total_tags = len(pos_tags)
            if total_tags > 0:
                noun_ratio = sum(1 for _, t in pos_tags if t.startswith(("NN", "NP"))) / total_tags
                verb_ratio = sum(1 for _, t in pos_tags if t.startswith("VB")) / total_tags
                adj_ratio = sum(1 for _, t in pos_tags if t.startswith("JJ")) / total_tags
                adv_ratio = sum(1 for _, t in pos_tags if t.startswith("RB")) / total_tags
    except:
        pass

    text_lower = text.lower()
    has_suicidal_keyword = int(any(k in text_lower for k in SUICIDAL_KEYWORDS))
    has_stress_keyword = int(any(k in text_lower for k in STRESS_KEYWORDS))
    has_help_keyword = int(any(k in text_lower for k in HELP_KEYWORDS))

    features = pd.DataFrame([[
        text_length, word_count, num_urls, num_emojis, num_special_chars, num_excess_punct, 
        avg_word_length, stopword_ratio, type_token_ratio, polarity, subjectivity, 
        noun_ratio, verb_ratio, adj_ratio, adv_ratio, has_suicidal_keyword, 
        has_stress_keyword, has_help_keyword
    ]], columns=FEATURE_COLUMNS)

    return features

# ============================================================
# 5. SUGGESTIONS & XAI (Untouched)
# ============================================================
def generate_suggestions(predicted_class):
    if predicted_class == "Suicidal":
        return [
            "The model detected language patterns associated with severe emotional distress.",
            "If you are in immediate danger, seek immediate help from local emergency services.",
            "Consider contacting a qualified mental health professional for appropriate support."
        ]
    elif predicted_class == "Depression":
        return [
            "The model detected language patterns associated with depressive distress.",
            "Try maintaining regular sleep, physical activity, and supportive social connections.",
            "If these feelings persist, consider speaking with a professional."
        ]
    elif predicted_class == "Anxiety":
        return [
            "The model detected language patterns associated with anxiety or stress.",
            "Consider relaxation techniques such as slow breathing or taking breaks.",
            "Maintaining healthy sleep habits and talking to someone you trust may help."
        ]
    else:
        return [
            "Your text does not show strong indicators of significant mental health distress.",
            "Continue maintaining healthy routines, adequate sleep, and physical activity."
        ]

def explain_prediction(new_text_tfidf, new_features, predicted_class):
    try:
        tfidf_feature_names = tfidf.get_feature_names_out().tolist()
        all_feature_names = tfidf_feature_names + FEATURE_COLUMNS
        class_index = list(hybrid_model.classes_).index(predicted_class)
        coefficients = hybrid_model.coef_[class_index]
        
        tfidf_values = new_text_tfidf.toarray()[0]
        engineered_values = scaler.transform(new_features)[0]
        all_values = np.concatenate([tfidf_values, engineered_values])
        
        contributions = all_values * coefficients
        
        explanation = pd.DataFrame({
            "Feature": all_feature_names,
            "Value": all_values,
            "Contribution": contributions
        })
        
        # Keep active features and sort
        explanation = explanation[explanation["Value"].abs() > 0.000001]
        explanation["Absolute Contribution"] = explanation["Contribution"].abs()
        explanation = explanation.sort_values("Absolute Contribution", ascending=False).head(15)
        
        return explanation[["Feature", "Contribution"]]
    except Exception as e:
        return pd.DataFrame()

# ============================================================
# 6. PROFESSIONAL UI LAYOUT
# ============================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 XAI Detector")
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This application uses a Hybrid NLP machine learning model combined with Explainable AI (XAI) to detect signs of mental health distress in text.
    
    **Features Used:**
    - TF-IDF Vectorization
    - Sentiment Polarity (TextBlob)
    - Part-of-Speech Ratios
    - Keyword Detection
    """)
    st.markdown("---")
    st.warning("**Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")

# Check if models loaded successfully
if tfidf is None or scaler is None or hybrid_model is None:
    st.error("Cannot proceed. Models failed to load. Please ensure your .pkl files are in a 'Models' directory relative to this script.")
    st.stop()

# --- MAIN PAGE HEADER ---
st.title("Mental Health Distress Analysis")
st.markdown("Enter text below to evaluate language patterns for signs of emotional distress.")

# --- TABS FOR WORKFLOW ---
tab_single, tab_batch = st.tabs(["📝 Single Text Analysis", "📁 Batch Dataset Processing"])

# ============================================================
# TAB 1: SINGLE ANALYSIS
# ============================================================
with tab_single:
    with st.form("analysis_form"):
        user_input = st.text_area("Input Text:", height=150, placeholder="Paste text here to analyze...")
        analyze_btn = st.form_submit_button("🔍 Analyze Text", type="primary")

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text before analysis.")
        else:
            with st.spinner("Analyzing text patterns..."):
                # Run Model Pipeline (Untouched)
                new_text_tfidf = tfidf.transform([user_input])
                new_features = extract_features_from_text(user_input)
                new_features_scaled = scaler.transform(new_features)
                new_hybrid = hstack([new_text_tfidf, new_features_scaled]).tocsr()
                
                predicted_class = hybrid_model.predict(new_hybrid)[0]
                
                if hasattr(hybrid_model, "predict_proba"):
                    probs = hybrid_model.predict_proba(new_hybrid)[0]
                    prob_dict = {str(c): float(p) for c, p in zip(hybrid_model.classes_, probs)}
                    confidence = max(prob_dict.values())
                else:
                    prob_dict = {}
                    confidence = 0.0

                explanation_df = explain_prediction(new_text_tfidf, new_features, predicted_class)
                suggestions = generate_suggestions(predicted_class)

            # --- DISPLAY RESULTS DASHBOARD ---
            st.markdown("### 📊 Analysis Results")
            
            # Top Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Category", predicted_class)
            with m2:
                st.metric("Confidence Score", f"{confidence * 100:.1f}%")
            with m3:
                st.metric("Words Analyzed", len(user_input.split()))

            st.markdown("<br>", unsafe_allow_html=True) # Spacer

            # Nested Tabs for deep-dive information
            res_tab1, res_tab2, res_tab3 = st.tabs(["📈 Class Probabilities", "🧠 Explainable AI (XAI)", "💡 Recommendations"])
            
            with res_tab1:
                st.markdown("#### Diagnosis Probability Breakdown")
                st.markdown("The model's confidence across all possible categories:")
                if prob_dict:
                    # Professional progress bars instead of raw charts
                    for label, prob in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True):
                        col_text, col_bar = st.columns([1, 4])
                        with col_text:
                            st.write(f"**{label}**")
                        with col_bar:
                            st.progress(prob, text=f"{prob * 100:.1f}%")
            
            with res_tab2:
                st.markdown("#### Feature Contributions")
                st.markdown("Which specific words or features pushed the model toward this prediction (Supporting > 0, Opposing < 0).")
                if not explanation_df.empty:
                    xai_col1, xai_col2 = st.columns([2, 1])
                    with xai_col1:
                        explanation_df = explanation_df.set_index("Feature")
                        st.bar_chart(explanation_df["Contribution"])
                    with xai_col2:
                        st.dataframe(
                            explanation_df,
                            use_container_width=True
                        )
                else:
                    st.info("No local explanation available for this input.")

            with res_tab3:
                st.markdown("#### Actionable Steps")
                # Dynamic styling based on severity
                if predicted_class == "Suicidal":
                    for sug in suggestions:
                        st.error(f"🚨 {sug}")
                elif predicted_class in ["Anxiety", "Depression"]:
                    for sug in suggestions:
                        st.warning(f"⚠️ {sug}")
                else:
                    for sug in suggestions:
                        st.success(f"✅ {sug}")


# ============================================================
# TAB 2: BATCH ANALYSIS
# ============================================================
with tab_batch:
    st.markdown("### Upload a CSV or Excel file for bulk analysis")
    uploaded_file = st.file_uploader("Choose a dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.dataframe(df.head(), use_container_width=True)
            st.info("Dataset loaded successfully. To process this entire dataset, loop through the target text column applying the `extract_features_from_text` and model `.predict` functions row-by-row.")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")