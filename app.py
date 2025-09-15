# app.py
import json, joblib, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
import shap, lime.lime_tabular
from utils import ensure_features, nlp_features_from_text, lifestyle_suggestions

st.set_page_config(page_title="T2D Interpretable AI", layout="centered")
st.title("🧠 Type-2 Diabetes Risk — Interpretable AI")

@st.cache_resource
def load_model():
    model = joblib.load("artifacts/model.pkl")
    with open("artifacts/feature_names.json") as f:
        feat_names = json.load(f)
    return model, feat_names

model, feat_names = load_model()

st.markdown("**Pipeline:** PIMA/NLP ➜ Feature Engineering ➜ Train/Test ➜ Model+Tuning ➜ SHAP/LIME ➜ Suggestions ➜ UI")

# --- Load training data for background (LIME/SHAP) ---
@st.cache_resource
def load_background():
    df = pd.read_csv("ai_final_dataset.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Is_High_Risk" in df.columns:
        df = df.rename(columns={"Is_High_Risk": "IsHighRisk"})
    df = ensure_features(df)

    for c in feat_names:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_names]

    # Clean background (remove inf/NaN + dups)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0).drop_duplicates()

    # Numeric copy for LIME/SHAP
    df_num = df.copy()
    for col in df_num.select_dtypes(include=["object", "category"]).columns:
        df_num[col] = pd.factorize(df_num[col])[0]

    return df, df_num

bg_df, bg_df_num = load_background()

tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV"])

# --- Single Prediction ---
with tab1:
    st.subheader("Single Patient Input")
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        Glucose = st.number_input("Glucose", 0, 300, 120)
        BloodPressure = st.number_input("BloodPressure", 0, 200, 70)
        SkinThickness = st.number_input("SkinThickness", 0, 100, 20)
        Insulin = st.number_input("Insulin", 0, 900, 80)
    with col2:
        BMI = st.number_input("BMI", 0.0, 70.0, 28.5)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01)
        Age = st.number_input("Age", 10, 100, 42)
        feedback = st.text_area("Patient Feedback (optional)", "I feel tired and thirsty lately")

    row = pd.DataFrame([{
        "Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age
    }])

    row = ensure_features(row)
    row = pd.concat([row, pd.DataFrame([nlp_features_from_text(feedback)])], axis=1)

    for c in feat_names:
        if c not in row.columns:
            row[c] = 0
    row = row[feat_names]

    st.write("Feature preview:", row)

    if st.button("Predict"):
        # Prediction with pipeline
        proba = model.predict_proba(row)[0, 1]
        pred = int(proba >= 0.5)
        st.success(f"**Risk Probability:** {proba:.2%}  |  **Prediction:** {'Diabetes' if pred else 'No Diabetes'}")

        # --- Prepare numeric copy for explanations ---
        row_num = row.copy()
        for col in row_num.select_dtypes(include=["object", "category"]).columns:
            row_num[col] = pd.factorize(row_num[col])[0]

        # --- LIME ---
        st.markdown("### 🔍 LIME (local explanation)")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_df_num.sample(min(300, len(bg_df_num)), random_state=42).values,
            feature_names=row_num.columns.tolist(),
            class_names=["No Diabetes", "Diabetes"],
            mode="classification",
            discretize_continuous=False
        )

        def predict_proba_wrapper(x):
            df = pd.DataFrame(x, columns=row_num.columns)
            return model.predict_proba(df)

        exp = explainer.explain_instance(
            row_num.iloc[0].values,
            predict_proba_wrapper,
            num_features=8
        )
        st.write(pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"]))

        # --- SHAP ---
        st.markdown("### 🧩 SHAP (feature contributions)")
        expl = shap.Explainer(model.predict_proba, bg_df_num.sample(min(200, len(bg_df_num)), random_state=42))
        sv = expl(row_num)

        # Fix: select SHAP values for positive class (Diabetes)
        if sv.values.ndim == 3:  # (n_samples, n_classes, n_features)
            shap_values_class1 = sv.values[0][1]      # first sample, class 1
            base_value_class1 = sv.base_values[0][1]
        else:
            shap_values_class1 = sv.values[0]
            base_value_class1 = sv.base_values[0]

        single_expl = shap.Explanation(
            values=shap_values_class1,
            base_values=base_value_class1,
            data=row_num.iloc[0].values,
            feature_names=row_num.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(single_expl, show=False)
        st.pyplot(fig)

        # --- Lifestyle Suggestions ---
        st.markdown("### 💡 Tailored Lifestyle Suggestions")
        for s in lifestyle_suggestions(row.iloc[0]):
            st.write("•", s)

# --- Batch Prediction ---
with tab2:
    st.subheader("Batch Prediction via CSV")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        if "Is_High_Risk" in df.columns:
            df = df.rename(columns={"Is_High_Risk": "IsHighRisk"})
        df = ensure_features(df)

        for c in feat_names:
            if c not in df.columns:
                df[c] = 0
        df = df[feat_names]

        preds = model.predict(df)
        prob = model.predict_proba(df)[:, 1]
        out = df.copy()
        out["RiskProb"] = prob
        out["Pred"] = preds
        st.write(out.head())

        # --- SHAP Global ---
        st.markdown("### 🧩 SHAP Summary (global)")
        df_num = df.copy()
        for col in df_num.select_dtypes(include=["object", "category"]).columns:
            df_num[col] = pd.factorize(df_num[col])[0]

        expl = shap.Explainer(model.predict_proba, df_num.sample(min(200, len(df_num)), random_state=42))
        sv = expl(df_num.sample(min(50, len(df_num)), random_state=42))

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, show=False)
        st.pyplot(fig)
