# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

st.title("Test: Streamlit is working!")
st.write("If you see this, the app is loading correctly.")


# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

st.title("🎓 Student Exam Score Prediction")
st.write("Predict a student's exam score based on input features.")

# --- Input fields ---
study_hours = st.number_input("Study Hours per day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
attendance = st.slider("Attendance %", 0, 100, 80)
lunch = st.selectbox("Lunch Type", df['lunch'].unique())
parent_edu = st.selectbox("Parental Education", df['parental level of education'].unique())

# --- Preprocessing ---
X = df[['study time', 'attendance', 'lunch', 'parental level of education']]
y = df['math score']  # predicting math score as an example

# Encode categorical columns
le_lunch = LabelEncoder()
X['lunch'] = le_lunch.fit_transform(X['lunch'])

le_parent = LabelEncoder()
X['parental level of education'] = le_parent.fit_transform(X['parental level of education'])

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Prepare user input for prediction ---
user_input = pd.DataFrame({
    'study time': [study_hours],
    'attendance': [attendance],
    'lunch': [le_lunch.transform([lunch])[0]],
    'parental level of education': [le_parent.transform([parent_edu])[0]]
})

# --- Predict ---
if st.button("Predict Score"):
    prediction = model.predict(user_input)[0]
    st.success(f"Predicted Math Score: {prediction:.2f}")

    # --- Bar chart ---
    fig, ax = plt.subplots()
    ax.bar(['Predicted Score'], [prediction], color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    st.pyplot(fig)

    # --- SHAP explanation ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.write("### Feature Importance (SHAP)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig2)
