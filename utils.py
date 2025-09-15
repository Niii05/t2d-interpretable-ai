# utils.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------
# 1. Ensure Engineered Features
# -------------------------------
def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features if missing."""
    df = df.copy()

    # Glucose-Insulin Ratio
    if "GlucoseInsulinRatio" not in df.columns:
        df["GlucoseInsulinRatio"] = df["Glucose"] / (df["Insulin"] + 1)

    # Age Group
    if "AgeGroup" not in df.columns:
        df["AgeGroup"] = df["Age"].apply(
            lambda age: "Young" if age < 30 else "Middle" if age < 50 else "Senior"
        )

    # BMI Category
    if "BMICategory" not in df.columns:
        def bmi_category(bmi):
            if pd.isna(bmi):
                return "Unknown"
            elif bmi < 18.5:
                return "Underweight"
            elif bmi < 25:
                return "Normal"
            elif bmi < 30:
                return "Overweight"
            else:
                return "Obese"
        df["BMICategory"] = df["BMI"].apply(bmi_category)

    # Age × BMI Interaction
    if "AgeBMIInteraction" not in df.columns:
        df["AgeBMIInteraction"] = df["Age"] * df["BMI"]

    # Health Risk Score
    if "HealthRiskScore" not in df.columns:
        df["HealthRiskScore"] = df[["Glucose", "Pregnancies", "BMI"]].sum(axis=1)

    # IsHighRisk Flag
    if "IsHighRisk" not in df.columns:
        df["IsHighRisk"] = ((df["Glucose"] > 150) & (df["BMI"] > 30)).astype(int)

    return df


# -------------------------------
# 2. NLP Features from Feedback
# -------------------------------
def nlp_features_from_text(text: str) -> dict:
    """Extract NLP-based features (sentiment, fatigue level) from patient feedback."""
    if not text or text.strip() == "":
        return {"sentiment": 0.0, "fatigue_level": 0}

    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)["compound"]

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Fatigue detection
    fatigue_keywords = ["tired", "exhausted", "weak", "fatigue", "sleepy"]
    fatigue_level = int(any(word in text.lower() for word in fatigue_keywords))

    return {
        "sentiment": (vader_score + polarity) / 2,
        "fatigue_level": fatigue_level,
    }


# -------------------------------
# 3. Lifestyle Suggestions
# -------------------------------
def lifestyle_suggestions(row: pd.Series) -> list:
    """Generate lifestyle suggestions based on patient features."""
    suggestions = []

    if row["Glucose"] > 140:
        suggestions.append("Reduce sugar intake and monitor glucose regularly.")
    if row["BMI"] >= 30:
        suggestions.append("Adopt a regular exercise routine to lower BMI.")
    if row["BloodPressure"] >= 130:
        suggestions.append("Reduce salt intake and monitor blood pressure.")
    if row["Insulin"] > 200:
        suggestions.append("Consult doctor for insulin management.")
    if row.get("fatigue_level", 0) == 1:
        suggestions.append("Improve sleep schedule and balanced diet.")

    if not suggestions:
        suggestions.append("Maintain current lifestyle with regular check-ups.")

    return suggestions
