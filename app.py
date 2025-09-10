import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")  # For feature importance only

# Page setup
st.set_page_config(page_title="Symptom Analyzer", layout="centered")
st.title("🧠 Symptom Analyzer Dashboard")
st.markdown("""
This tool uses a **Random Forest model** to predict whether a patient has **Flu**, **Allergy**, **Cold**, or **COVID** based on symptoms.  
It also shows how a Decision Tree model weighs each symptom's importance.
""")

# Define symptoms
symptoms = [
    'COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE',
    'STUFFY_NOSE', 'FEVER', 'NAUSEA', 'VOMITING', 'DIARRHEA',
    'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 'LOSS_OF_TASTE',
    'LOSS_OF_SMELL', 'SNEEZING', 'PINK_EYE'
]

# Collect input using checkboxes
st.header("🩺 Enter Patient Symptoms")
user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.checkbox(f"{symptom.replace('_', ' ').title()}")

# Convert input to DataFrame
input_df = pd.DataFrame([{
    symptom: 1 if user_input[symptom] else 0
    for symptom in symptoms
}])

# Prediction logic
if st.button("🔍 Predict Disease"):
    if input_df.sum().sum() == 0:
        st.warning("⚠️ No symptoms selected. Prediction may not be meaningful.")
    else:
        prediction = rf_model.predict(input_df)[0]
        confidence = rf_model.predict_proba(input_df)[0]
        class_labels = rf_model.classes_

        st.success(f"🧬 Predicted Disease: **{prediction}**")

        st.markdown("### 📊 Prediction Confidence")
        for i, label in enumerate(class_labels):
            st.write(f"**{label}**: {confidence[i]:.2f}")

        predicted_index = list(class_labels).index(prediction)
        predicted_confidence = confidence[predicted_index]
        if predicted_confidence < 0.6:
            st.warning("⚠️ Low confidence prediction. Please interpret with caution.")

# Feature importance chart (from Decision Tree)
st.markdown("### 🔍 Symptom Importance (Decision Tree)")
importance = pd.Series(dt_model.feature_importances_, index=input_df.columns)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index, ax=ax)
ax.set_title("Feature Importance in Decision Tree")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Symptoms")
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built by Ali 👑 — Powered by Random Forest & Decision Tree logic.")

