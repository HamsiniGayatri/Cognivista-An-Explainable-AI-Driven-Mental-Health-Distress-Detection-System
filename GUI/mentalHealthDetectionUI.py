# #welcome, detect your mental stability.

# #1.Give a title for the project first assign title and assign a meaning full header

# import tkinter;
# from tkinter import messagebox
# from tkinter import scrolledtext;
# from tkinter import filedialog;
# import openpyxl
# #root is just a user defined variable
# root = tkinter.Tk()
# #set filepath
# file_path = r"C:\Users\Lenovo\Downloads\UserLog.xlsx"
# workbook = openpyxl.load_workbook(file_path)
# sheet = workbook["Registration"]


# #Tk() is a class inside a package
# root.configure(bg="#E6E6FA")
# root.title("XAI MENTAL HEALTH DISTRESS DETECTOR")
# header = tkinter.Label(root,text="MENTAL HEALTH DISTRESS DETECTOR", font=("Arial", 18, "bold"), bg="#E6E6FA",fg="#000000")
# header.pack(pady=(50,20))

# #create a frame
# risk_frame = tkinter.Frame(root, bg="#E6E6FA")
# risk_frame.pack(pady=5)
# #Risk score, to display risk score we should first designate label then single line text box
# riskValue = tkinter.Label(risk_frame,text="Risk Score:",font=("italic",12),fg="red",bg="#E6E6FA")
# riskValue.pack(side="left", padx=10)
# TextLine1 = tkinter.Entry(risk_frame)
# TextLine1.pack(side="left")

# #remidial suggestions function

# def show_remedial_suggestions():
#     remedial_window = tkinter.Toplevel(root)
#     remedial_window.title("Remedial Suggestions")
#     remedial_window.geometry("500x400")
#     remedial_window.configure(bg="#F5F5F5")

#     label = tkinter.Label(remedial_window, text="Remedial Suggestions for Mental Health", font=("Arial", 12, "bold"), bg="#F5F5F5", fg="#333")
#     label.pack(pady=15)

#     suggestions_text = "Suggestions"

#     text_widget = tkinter.Text(remedial_window, wrap="word", bg="white", font=("Arial", 12))
#     text_widget.insert("1.0", suggestions_text)
#     text_widget.config(state=tkinter.DISABLED)
#     text_widget.pack(padx=20, pady=10, fill="both", expand=True)
    
    
# #remidial suggestions button

# remedial_button = tkinter.Button(root, text="Remedial Suggestions", command=show_remedial_suggestions,bg="#90EE90", font=("Arial", 12, "bold"), width=18)
# remedial_button.pack(pady=(10, 20))

# #text box beside a text box with scroll bar 
# text_frame = tkinter.Frame(root, bg="#E6E6FA")
# text_frame.pack(pady=5)
# text_scroll1 = scrolledtext.ScrolledText(text_frame, wrap=tkinter.WORD, height=20, width=80, bg='white')
# text_scroll1.pack(side="left", padx=25, pady=20)

# text_scroll2 = scrolledtext.ScrolledText(text_frame, wrap=tkinter.WORD, height=20, width=80, bg='lightgrey', state=tkinter.DISABLED)
# text_scroll2.pack(side="left", pady=20)

# #function to show submited dialog box and clear all function
# def Submitted():
#    textBox = text_scroll1.get("1.0", "end-1c")
#    if textBox.strip(): 
#         sheet.append([textBox])
#         workbook.save(file_path)
#         messagebox.showinfo("Done", "Saved Successfully!")
#         text_scroll1.delete("1.0", "end")
#    else:
#         messagebox.showwarning("Oops", "Please fill all the fields")
# def ClearAll():
#     text_scroll1.delete("1.0", "end")
#     text_scroll2.config(state=tkinter.NORMAL)
#     text_scroll2.delete("1.0", "end")
#     text_scroll2.config(state=tkinter.DISABLED)
#     TextLine1.delete(0, tkinter.END)
#     status_label.config(text="")
            
# #button-submit btn
# button_frame = tkinter.Frame(root, bg="#E6E6FA")
# button_frame.pack(pady=5)
# SubmitButton = tkinter.Button(button_frame,text="Submit",height=1,width=10,bg="yellow",command=Submitted)
# SubmitButton.pack(side="left",padx=30)
# ClearButton = tkinter.Button(button_frame,text="Clear",height=1,width=10,bg="lightgrey",command=ClearAll)
# ClearButton.pack(side="left")

# #create an upload button
# def upload_file():
#     file_path = filedialog.askopenfilename(
#         title="Select a File",
#         filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
#     )
#     if file_path:
#         print("File selected:", file_path)
#         # Example: read csv using pandas
#         # import pandas as pd
#         # df = pd.read_csv(file_path)
#         # print(df.head())
#     else:
#         print("No file selected.")
#     if file_path:
#         # Show success message in red
#         status_label.config(text="File Uploaded Successfully!", fg="red")
#         print("File selected:", file_path)
#     else:
#         status_label.config(text="No file selected", fg="gray")


# # Frame to center upload button and align with above components
# upload_frame = tkinter.Frame(root, bg="#E6E6FA", width=800)
# upload_frame.pack(pady=(35, 10))

# upload_button = tkinter.Button(
#     upload_frame,
#     text="📤Upload Dataset",
#     command=upload_file,
#     bg="lightblue",
#     font=("Arial", 12, "bold"),
#     height=1,
#     width=20
# )
# upload_button.pack(side="left",anchor="center")
# #analyze button
# analyzeButton = tkinter.Button(upload_frame,text="🔍Analyze",height=1,width=20,bg="red",font=("italic",12,"bold"))
# analyzeButton.pack(side="left",padx=10,pady=20)


# status_label = tkinter.Label(
#     root,
#     text="",          
#     font=("Arial", 11, "italic"),
#     bg="#E6E6FA",
#     fg="red"
# )
# status_label.pack()



# root.mainloop()

# ============================================================
# XAI MENTAL HEALTH DISTRESS DETECTOR
# Hybrid NLP Model + XAI + Suggestions + Tkinter GUI
# ============================================================

import os
import re
import string
import tkinter as tk

from tkinter import (
    messagebox,
    filedialog,
    scrolledtext
)

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from textblob import TextBlob


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "Models"
)


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

try:

    tfidf = joblib.load(
        os.path.join(
            MODELS_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "scaler.pkl"
        )
    )

    hybrid_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "hybrid_model.pkl"
        )
    )

    print("Models loaded successfully.")

except Exception as e:

    print(
        "ERROR LOADING MODELS:",
        e
    )

    tfidf = None
    scaler = None
    hybrid_model = None


# ============================================================
# 3. FEATURE COLUMN NAMES
# ============================================================

FEATURE_COLUMNS = [

    "text_length",

    "word_count",

    "num_urls",

    "num_emojis",

    "num_special_chars",

    "num_excess_punct",

    "avg_word_length",

    "stopword_ratio",

    "type_token_ratio",

    "polarity",

    "subjectivity",

    "noun_ratio",

    "verb_ratio",

    "adj_ratio",

    "adv_ratio",

    "has_suicidal_keyword",

    "has_stress_keyword",

    "has_help_keyword"

]


# ============================================================
# 4. KEYWORD LISTS
# ============================================================

SUICIDAL_KEYWORDS = [

    "suicide",

    "suicidal",

    "kill myself",

    "end my life",

    "take my life",

    "want to die",

    "wish i was dead",

    "overdose",

    "self harm"

]


STRESS_KEYWORDS = [

    "stress",

    "stressed",

    "overwhelmed",

    "pressure",

    "anxious",

    "anxiety",

    "worried",

    "panic",

    "tired",

    "exhausted"

]


HELP_KEYWORDS = [

    "help",

    "therapy",

    "therapist",

    "counselling",

    "counseling",

    "support",

    "doctor",

    "mental health"

]


# ============================================================
# 5. FEATURE EXTRACTION
# ============================================================

def extract_features_from_text(text):

    words = text.split()

    word_count = len(words)

    text_length = len(text)


    # --------------------------------------------------------
    # URL COUNT
    # --------------------------------------------------------

    num_urls = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text
        )
    )


    # --------------------------------------------------------
    # EMOJI / NON ASCII COUNT
    # --------------------------------------------------------

    num_emojis = len(
        re.findall(
            r"[^\x00-\x7F]",
            text
        )
    )


    # --------------------------------------------------------
    # SPECIAL CHARACTERS
    # --------------------------------------------------------

    num_special_chars = len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            text
        )
    )


    # --------------------------------------------------------
    # EXCESSIVE PUNCTUATION
    # --------------------------------------------------------

    num_excess_punct = len(
        re.findall(
            r"[!?.,]{2,}",
            text
        )
    )


    # --------------------------------------------------------
    # AVERAGE WORD LENGTH
    # --------------------------------------------------------

    if word_count > 0:

        avg_word_length = (

            sum(
                len(word)
                for word in words
            )

            / word_count

        )

    else:

        avg_word_length = 0


    # --------------------------------------------------------
    # STOPWORD RATIO
    # --------------------------------------------------------

    stopwords = {

        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "at"

    }


    if word_count > 0:

        stopword_count = sum(

            1

            for word in words

            if word.lower().strip(
                string.punctuation
            )
            in stopwords

        )

        stopword_ratio = (

            stopword_count

            / word_count

        )

    else:

        stopword_ratio = 0


    # --------------------------------------------------------
    # TYPE TOKEN RATIO
    # --------------------------------------------------------

    unique_words = set(

        word.lower()

        for word in words

    )


    if word_count > 0:

        type_token_ratio = (

            len(unique_words)

            / word_count

        )

    else:

        type_token_ratio = 0


    # --------------------------------------------------------
    # TEXTBLOB SENTIMENT
    # --------------------------------------------------------

    try:

        blob = TextBlob(text)

        polarity = (
            blob.sentiment.polarity
        )

        subjectivity = (
            blob.sentiment.subjectivity
        )

    except Exception:

        blob = None

        polarity = 0

        subjectivity = 0


    # --------------------------------------------------------
    # POS RATIOS
    # --------------------------------------------------------

    noun_ratio = 0

    verb_ratio = 0

    adj_ratio = 0

    adv_ratio = 0


    try:

        if blob is not None:

            pos_tags = blob.tags

            total_tags = len(
                pos_tags
            )


            if total_tags > 0:

                noun_count = sum(

                    1

                    for _, tag
                    in pos_tags

                    if tag.startswith(
                        ("NN", "NP")
                    )

                )


                verb_count = sum(

                    1

                    for _, tag
                    in pos_tags

                    if tag.startswith(
                        "VB"
                    )

                )


                adj_count = sum(

                    1

                    for _, tag
                    in pos_tags

                    if tag.startswith(
                        "JJ"
                    )

                )


                adv_count = sum(

                    1

                    for _, tag
                    in pos_tags

                    if tag.startswith(
                        "RB"
                    )

                )


                noun_ratio = (

                    noun_count

                    / total_tags

                )


                verb_ratio = (

                    verb_count

                    / total_tags

                )


                adj_ratio = (

                    adj_count

                    / total_tags

                )


                adv_ratio = (

                    adv_count

                    / total_tags

                )

    except Exception:

        pass


    # --------------------------------------------------------
    # KEYWORD FEATURES
    # --------------------------------------------------------

    text_lower = text.lower()


    has_suicidal_keyword = int(

        any(

            keyword in text_lower

            for keyword
            in SUICIDAL_KEYWORDS

        )

    )


    has_stress_keyword = int(

        any(

            keyword in text_lower

            for keyword
            in STRESS_KEYWORDS

        )

    )


    has_help_keyword = int(

        any(

            keyword in text_lower

            for keyword
            in HELP_KEYWORDS

        )

    )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    features = pd.DataFrame(

        [[

            text_length,

            word_count,

            num_urls,

            num_emojis,

            num_special_chars,

            num_excess_punct,

            avg_word_length,

            stopword_ratio,

            type_token_ratio,

            polarity,

            subjectivity,

            noun_ratio,

            verb_ratio,

            adj_ratio,

            adv_ratio,

            has_suicidal_keyword,

            has_stress_keyword,

            has_help_keyword

        ]],

        columns=FEATURE_COLUMNS

    )


    return features


# ============================================================
# 6. GENERATE SUGGESTIONS
# ============================================================

def generate_suggestions(
    predicted_class,
    confidence,
    features
):

    suggestions = []


    if predicted_class == "Suicidal":

        suggestions.append(
            "The model detected language patterns "
            "associated with severe emotional distress."
        )

        suggestions.append(
            "If you are in immediate danger or feel "
            "you may harm yourself, seek immediate help "
            "from local emergency services or a trusted person."
        )

        suggestions.append(
            "Consider contacting a qualified mental "
            "health professional for appropriate support."
        )


    elif predicted_class == "Depression":

        suggestions.append(
            "The model detected language patterns that "
            "may be associated with depressive distress."
        )

        suggestions.append(
            "Try maintaining regular sleep, physical "
            "activity, and supportive social connections."
        )

        suggestions.append(
            "If these feelings persist, consider speaking "
            "with a trusted person or mental health professional."
        )


    elif predicted_class == "Anxiety":

        suggestions.append(
            "The model detected language patterns that "
            "may be associated with anxiety or stress."
        )

        suggestions.append(
            "Consider relaxation techniques such as "
            "slow breathing, mindfulness, or taking breaks."
        )

        suggestions.append(
            "Maintaining healthy sleep habits and talking "
            "to someone you trust may also be helpful."
        )


    else:

        suggestions.append(
            "Your text does not show strong indicators "
            "of significant mental health distress according "
            "to the model."
        )

        suggestions.append(
            "Continue maintaining healthy routines, adequate "
            "sleep, physical activity, and supportive social connections."
        )

        suggestions.append(
            "If your feelings change or become difficult "
            "to manage, consider talking to someone you trust."
        )


    return suggestions


# ============================================================
# 7. XAI EXPLANATION
# ============================================================

def explain_prediction(
    new_text_tfidf,
    new_features,
    predicted_class
):

    try:

        # Get TF-IDF feature names
        tfidf_feature_names = (

            tfidf

            .get_feature_names_out()

            .tolist()

        )


        # Combine names
        all_feature_names = (

            tfidf_feature_names

            + FEATURE_COLUMNS

        )


        # Get predicted class index
        class_index = list(

            hybrid_model.classes_

        ).index(

            predicted_class

        )


        # Get model coefficients
        coefficients = (

            hybrid_model

            .coef_[class_index]

        )


        # Get TF-IDF values
        tfidf_values = (

            new_text_tfidf

            .toarray()[0]

        )


        # Scale engineered features
        engineered_values = (

            scaler

            .transform(
                new_features
            )[0]

        )


        # Combine feature values
        all_values = np.concatenate(

            [

                tfidf_values,

                engineered_values

            ]

        )


        # Safety check
        if len(all_feature_names) != len(
            coefficients
        ):

            return pd.DataFrame()


        # Calculate contributions
        contributions = (

            all_values

            * coefficients

        )


        # Create DataFrame
        explanation = pd.DataFrame(

            {

                "feature":
                    all_feature_names,

                "feature_value":
                    all_values,

                "coefficient":
                    coefficients,

                "contribution":
                    contributions

            }

        )


        # Keep only active features
        explanation = explanation[

            explanation[
                "feature_value"
            ].abs() > 0.000001

        ]


        # Sort by absolute contribution
        explanation[

            "absolute_contribution"

        ] = (

            explanation[
                "contribution"
            ].abs()

        )


        explanation = (

            explanation

            .sort_values(

                "absolute_contribution",

                ascending=False

            )

        )


        return explanation.head(20)


    except Exception as e:

        print(
            "XAI Error:",
            e
        )

        return pd.DataFrame()


# ============================================================
# 8. MAIN ML ANALYSIS FUNCTION
# ============================================================

def analyze_text(user_text):

    if not user_text.strip():

        raise ValueError(
            "Please enter some text before analysis."
        )


    if (

        tfidf is None

        or scaler is None

        or hybrid_model is None

    ):

        raise RuntimeError(
            "Trained models could not be loaded."
        )


    # --------------------------------------------------------
    # STEP 1: TF-IDF
    # --------------------------------------------------------

    new_text_tfidf = (

        tfidf

        .transform(
            [user_text]
        )

    )


    # --------------------------------------------------------
    # STEP 2: ENGINEERED FEATURES
    # --------------------------------------------------------

    new_features = (

        extract_features_from_text(
            user_text
        )

    )


    # --------------------------------------------------------
    # STEP 3: SCALE FEATURES
    # --------------------------------------------------------

    new_features_scaled = (

        scaler

        .transform(
            new_features
        )

    )


    # --------------------------------------------------------
    # STEP 4: HYBRID FEATURES
    # --------------------------------------------------------

    new_hybrid = hstack(

        [

            new_text_tfidf,

            new_features_scaled

        ]

    ).tocsr()


    # --------------------------------------------------------
    # STEP 5: PREDICTION
    # --------------------------------------------------------

    predicted_class = (

        hybrid_model

        .predict(
            new_hybrid
        )[0]

    )


    # --------------------------------------------------------
    # STEP 6: CONFIDENCE
    # --------------------------------------------------------

    probabilities = {}


    if hasattr(

        hybrid_model,

        "predict_proba"

    ):

        probability_values = (

            hybrid_model

            .predict_proba(
                new_hybrid
            )[0]

        )


        for label, probability in zip(

            hybrid_model.classes_,

            probability_values

        ):

            probabilities[
                str(label)
            ] = float(
                probability
            )


        confidence = max(

            probabilities.values()

        )


    else:

        confidence = 0.0


    # --------------------------------------------------------
    # STEP 7: XAI
    # --------------------------------------------------------

    explanation = explain_prediction(

        new_text_tfidf,

        new_features,

        predicted_class

    )


    # --------------------------------------------------------
    # STEP 8: SUGGESTIONS
    # --------------------------------------------------------

    suggestions = generate_suggestions(

        predicted_class,

        confidence,

        new_features

    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "prediction":
            predicted_class,

        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "suggestions":
            suggestions,

        "explanation":
            explanation,

        "features":
            new_features

    }


# ============================================================
# 9. DISPLAY ANALYSIS IN GUI
# ============================================================

def run_analysis():

    try:

        user_text = (

            text_scroll1

            .get(
                "1.0",
                "end-1c"
            )

            .strip()

        )


        if not user_text:

            messagebox.showwarning(

                "Input Required",

                "Please enter text to analyze."

            )

            return


        # Run model
        result = analyze_text(

            user_text

        )


        prediction = (

            result[
                "prediction"
            ]

        )


        confidence = (

            result[
                "confidence"
            ]

        )


        probabilities = (

            result[
                "probabilities"
            ]

        )


        suggestions = (

            result[
                "suggestions"
            ]

        )


        explanation = (

            result[
                "explanation"
            ]

        )


        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        TextLine1.delete(

            0,

            tk.END

        )


        TextLine1.insert(

            0,

            f"{confidence * 100:.2f}%"

        )


        # ----------------------------------------------------
        # RIGHT SIDE OUTPUT
        # ----------------------------------------------------

        text_scroll2.config(

            state=tk.NORMAL

        )


        text_scroll2.delete(

            "1.0",

            tk.END

        )


        text_scroll2.insert(

            tk.END,

            "MENTAL HEALTH DISTRESS ANALYSIS\n"

        )

        text_scroll2.insert(

            tk.END,

            "=" * 45

            + "\n\n"

        )


        text_scroll2.insert(

            tk.END,

            f"Predicted Category: {prediction}\n"

        )


        text_scroll2.insert(

            tk.END,

            f"Confidence: {confidence * 100:.2f}%\n\n"

        )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        text_scroll2.insert(

            tk.END,

            "CLASS PROBABILITIES\n"

        )

        text_scroll2.insert(

            tk.END,

            "-" * 30

            + "\n"

        )


        for label, probability in (

            probabilities.items()

        ):

            text_scroll2.insert(

                tk.END,

                f"{label}: "
                f"{probability * 100:.2f}%\n"

            )


        # ----------------------------------------------------
        # SUGGESTIONS
        # ----------------------------------------------------

        text_scroll2.insert(

            tk.END,

            "\nSUGGESTIONS\n"

        )

        text_scroll2.insert(

            tk.END,

            "-" * 30

            + "\n"

        )


        for i, suggestion in enumerate(

            suggestions,

            1

        ):

            text_scroll2.insert(

                tk.END,

                f"{i}. {suggestion}\n\n"

            )


        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        text_scroll2.insert(

            tk.END,

            "\nXAI EXPLANATION\n"

        )

        text_scroll2.insert(

            tk.END,

            "-" * 30

            + "\n"

        )


        if not explanation.empty:

            for _, row in (

                explanation.head(10)

                .iterrows()

            ):

                contribution = (

                    row[
                        "contribution"
                    ]

                )


                direction = (

                    "supporting"

                    if contribution > 0

                    else "opposing"

                )


                text_scroll2.insert(

                    tk.END,

                    f"• {row['feature']} "
                    f"({direction})\n"

                )


        else:

            text_scroll2.insert(

                tk.END,

                "No local explanation available.\n"

            )


        text_scroll2.config(

            state=tk.DISABLED

        )


        # Update status
        status_label.config(

            text="Analysis completed successfully.",

            fg="green"

        )


    except Exception as e:

        messagebox.showerror(

            "Analysis Error",

            f"An error occurred while "
            f"analyzing the text:\n\n{str(e)}"

        )

        print(

            "Analysis Error:",

            e

        )


# ============================================================
# 10. CLEAR FUNCTION
# ============================================================

def clear_all():

    text_scroll1.delete(

        "1.0",

        tk.END

    )


    text_scroll2.config(

        state=tk.NORMAL

    )


    text_scroll2.delete(

        "1.0",

        tk.END

    )


    text_scroll2.config(

        state=tk.DISABLED

    )


    TextLine1.delete(

        0,

        tk.END

    )


    status_label.config(

        text=""

    )


# ============================================================
# 11. UPLOAD DATASET
# ============================================================

def upload_file():

    selected_file = (

        filedialog.askopenfilename(

            title="Select Dataset",

            filetypes=[

                (
                    "CSV Files",

                    "*.csv"

                ),

                (

                    "Excel Files",

                    "*.xlsx"

                ),

                (

                    "All Files",

                    "*.*"

                )

            ]

        )

    )


    if selected_file:

        status_label.config(

            text="File Uploaded Successfully!",

            fg="green"

        )

        print(

            "Selected file:",

            selected_file

        )

    else:

        status_label.config(

            text="No file selected.",

            fg="gray"

        )


# ============================================================
# 12. REMEDIAL SUGGESTIONS WINDOW
# ============================================================

def show_remedial_suggestions():

    remedial_window = tk.Toplevel(

        root

    )


    remedial_window.title(

        "Remedial Suggestions"

    )


    remedial_window.geometry(

        "600x450"

    )


    remedial_window.configure(

        bg="#F5F5F5"

    )


    label = tk.Label(

        remedial_window,

        text=(
            "Mental Health Support Suggestions"
        ),

        font=(

            "Arial",

            14,

            "bold"

        ),

        bg="#F5F5F5",

        fg="#333333"

    )


    label.pack(

        pady=15

    )


    suggestions_text = (

        "General wellbeing suggestions:\n\n"

        "• Maintain a regular sleep schedule.\n\n"

        "• Stay physically active when possible.\n\n"

        "• Maintain supportive social connections.\n\n"

        "• Take regular breaks during stressful activities.\n\n"

        "• Practice relaxation or mindfulness techniques.\n\n"

        "• Talk to someone you trust if you are struggling.\n\n"

        "• Consider seeking professional support when needed.\n\n"

        "Note: This application is an AI-based screening "
        "and educational tool. It does not provide a "
        "medical diagnosis."

    )


    text_widget = tk.Text(

        remedial_window,

        wrap="word",

        bg="white",

        font=(

            "Arial",

            11

        )

    )


    text_widget.insert(

        "1.0",

        suggestions_text

    )


    text_widget.config(

        state=tk.DISABLED

    )


    text_widget.pack(

        padx=20,

        pady=10,

        fill="both",

        expand=True

    )


# ============================================================
# 13. TKINTER MAIN WINDOW
# ============================================================

root = tk.Tk()


root.configure(

    bg="#E6E6FA"

)


root.title(

    "XAI Mental Health Distress Detector"

)


root.geometry(

    "1200x800"

)


# ============================================================
# HEADER
# ============================================================

header = tk.Label(

    root,

    text=(
        "XAI MENTAL HEALTH DISTRESS DETECTOR"
    ),

    font=(

        "Arial",

        20,

        "bold"

    ),

    bg="#E6E6FA",

    fg="#000000"

)


header.pack(

    pady=(30, 20)

)


# ============================================================
# RISK SCORE
# ============================================================

risk_frame = tk.Frame(

    root,

    bg="#E6E6FA"

)


risk_frame.pack(

    pady=5

)


riskValue = tk.Label(

    risk_frame,

    text="Confidence Score:",

    font=(

        "Arial",

        12,

        "bold"

    ),

    fg="red",

    bg="#E6E6FA"

)


riskValue.pack(

    side="left",

    padx=10

)


TextLine1 = tk.Entry(

    risk_frame,

    width=20,

    font=(

        "Arial",

        12

    )

)


TextLine1.pack(

    side="left"

)


# ============================================================
# SUGGESTION BUTTON
# ============================================================

remedial_button = tk.Button(

    root,

    text="Remedial Suggestions",

    command=show_remedial_suggestions,

    bg="#90EE90",

    font=(

        "Arial",

        12,

        "bold"

    ),

    width=22

)


remedial_button.pack(

    pady=(10, 20)

)


# ============================================================
# TEXT INPUT AND OUTPUT
# ============================================================

text_frame = tk.Frame(

    root,

    bg="#E6E6FA"

)


text_frame.pack(

    pady=5

)


text_scroll1 = scrolledtext.ScrolledText(

    text_frame,

    wrap=tk.WORD,

    height=20,

    width=55,

    bg="white",

    font=(

        "Arial",

        11

    )

)


text_scroll1.pack(

    side="left",

    padx=15,

    pady=20

)


text_scroll2 = scrolledtext.ScrolledText(

    text_frame,

    wrap=tk.WORD,

    height=20,

    width=55,

    bg="lightgrey",

    state=tk.DISABLED,

    font=(

        "Arial",

        11

    )

)


text_scroll2.pack(

    side="left",

    padx=15,

    pady=20

)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(

    root,

    bg="#E6E6FA"

)


button_frame.pack(

    pady=5

)


ClearButton = tk.Button(

    button_frame,

    text="Clear",

    height=1,

    width=12,

    bg="lightgrey",

    command=clear_all

)


ClearButton.pack(

    side="left",

    padx=15

)


# ============================================================
# UPLOAD AND ANALYZE
# ============================================================

upload_frame = tk.Frame(

    root,

    bg="#E6E6FA"

)


upload_frame.pack(

    pady=(20, 10)

)


upload_button = tk.Button(

    upload_frame,

    text="Upload Dataset",

    command=upload_file,

    bg="lightblue",

    font=(

        "Arial",

        12,

        "bold"

    ),

    height=1,

    width=20

)


upload_button.pack(

    side="left",

    padx=10

)


analyzeButton = tk.Button(

    upload_frame,

    text="Analyze",

    command=run_analysis,

    height=1,

    width=20,

    bg="red",

    fg="white",

    font=(

        "Arial",

        12,

        "bold"

    )

)


analyzeButton.pack(

    side="left",

    padx=10

)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(

    root,

    text="",

    font=(

        "Arial",

        11,

        "italic"

    ),

    bg="#E6E6FA",

    fg="red"

)


status_label.pack(

    pady=10

)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()