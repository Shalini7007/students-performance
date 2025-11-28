# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

st.title("🎓 Student Exam Score Prediction")
st.write("Predict a student's math score based on input features.")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

# Show the first few rows
if st.checkbox("Show dataset preview"):
    st.write(df.head())

# --- Input fields ---
gender = st.selectbox("Gender", df['gender'].unique())
race = st.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
parent_edu = st.selectbox("Parental Education", df['parental level of education'].unique())
lunch = st.selectbox("Lunch Type", df['lunch'].unique())
test_prep = st.selectbox("Test Preparation Course", df['test preparation course'].unique())

# --- Prepare features ---
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
y = df['math score']

# Encode categorical features
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Prepare user input for prediction ---
user_input = pd.DataFrame({
    'gender': [le_dict['gender'].transform([gender])[0]],
    'race/ethnicity': [le_dict['race/ethnicity'].transform([race])[0]],
    'parental level of education': [le_dict['parental level of education'].transform([parent_edu])[0]],
    'lunch': [le_dict['lunch'].transform([lunch])[0]],
    'test preparation course': [le_dict['test preparation course'].transform([test_prep])[0]]
})

# --- Predict ---
if st.button("Predict Math Score"):
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
# redeploy-trigger
