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
    page_title="XAI Mental Health Distress Detector",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# 2. CONSTANTS & KEYWORDS
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
# 3. LOAD MODELS (CACHED)
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
# 4. FEATURE EXTRACTION
# ============================================================
def extract_features_from_text(text):
    # Ensure exact same stripping as Tkinter to prevent prediction mismatches
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
# 5. SUGGESTIONS & XAI
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
# 6. MAIN UI
# ============================================================
st.title("🧠 XAI Mental Health Distress Detector")
st.markdown("Hybrid NLP Model + XAI + Suggestions")

# Check if models loaded successfully
if tfidf is None or scaler is None or hybrid_model is None:
    st.error("Cannot proceed. Models failed to load. Please ensure your .pkl files are in a 'Models' directory relative to this script.")
    st.stop()

# Input area
user_input = st.text_area("Enter text for analysis:", height=150)

col1, col2 = st.columns([1, 1])

with col1:
    analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
with col2:
    uploaded_file = st.file_uploader("📤 Upload Dataset (CSV/Excel) for Batch Analysis", type=["csv", "xlsx"])

if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter some text before analysis.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. TF-IDF
            new_text_tfidf = tfidf.transform([user_input])
            
            # 2. Engineered Features
            new_features = extract_features_from_text(user_input)
            
            # 3. Scale Features
            new_features_scaled = scaler.transform(new_features)
            
            # 4. Hybrid Features
            new_hybrid = hstack([new_text_tfidf, new_features_scaled]).tocsr()
            
            # 5. Prediction
            predicted_class = hybrid_model.predict(new_hybrid)[0]
            
            # 6. Probabilities
            if hasattr(hybrid_model, "predict_proba"):
                probs = hybrid_model.predict_proba(new_hybrid)[0]
                prob_dict = {str(c): float(p) for c, p in zip(hybrid_model.classes_, probs)}
                confidence = max(prob_dict.values())
            else:
                prob_dict = {}
                confidence = 0.0

            # 7. XAI
            explanation_df = explain_prediction(new_text_tfidf, new_features, predicted_class)
            
            # 8. Suggestions
            suggestions = generate_suggestions(predicted_class)

        st.divider()
        
        # Display Results Layout
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("Analysis Results")
            st.metric("Predicted Category", predicted_class)
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
            
            st.markdown("### Class Probabilities")
            if prob_dict:
                # Convert to dataframe for nice Streamlit bar chart
                prob_df = pd.DataFrame.from_dict(prob_dict, orient='index', columns=['Probability'])
                st.bar_chart(prob_df)
                
            st.markdown("### Remedial Suggestions")
            for i, sug in enumerate(suggestions, 1):
                st.info(f"{i}. {sug}")

        with res_col2:
            st.subheader("XAI Explanation")
            st.markdown("Features driving this specific prediction (Supporting > 0, Opposing < 0):")
            
            if not explanation_df.empty:
                # Plot horizontal bar chart for XAI
                explanation_df = explanation_df.set_index("Feature")
                st.bar_chart(explanation_df["Contribution"])
                st.dataframe(explanation_df, use_container_width=True)
            else:
                st.write("No local explanation available.")

# File upload logic (Batch Processing)
if uploaded_file is not None:
    st.divider()
    st.subheader("Batch Dataset Analysis")
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("Dataset Preview:", df.head())
        st.info("To process this entire dataset, loop through the target text column applying the `extract_features_from_text` and model `.predict` functions row-by-row.")
        # Implementation for batch processing would go here
    except Exception as e:
        st.error(f"Error reading file: {e}")