import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Student Predictor - Renuka")
st.title("🎓 AI Student Performance Predictor - PRO")
st.write("Built by Renuka Nallapu | SVIST | Day 7")

# Load and Train Model
@st.cache_data
def load_model():
    df = pd.read_csv("data.csv")
    X = df[["study_hours","attendance","prev_cgpa"]]
    y = df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, acc

model, accuracy = load_model()
st.success(f"✅ Model Trained! Accuracy: {accuracy:.1f}%")

# Inputs
hours = st.slider("Study Hours per Day", 1, 10, 4)
attendance = st.slider("Attendance %", 40, 100, 52)
prev_cgpa = st.slider("Previous CGPA", 5.0, 10.0, 6.67)

if st.button("🔮 Predict with AI"):
    pred = model.predict([[hours, attendance, prev_cgpa]])[0]
    if pred == "High":
        st.balloons()
        st.markdown(f"## :green[🚀 {pred} Performer - 9+ CGPA!]")
    elif pred == "Average":
        st.markdown(f"## :orange[📚 {pred} - 7-8 CGPA Expected]")
    else:
        st.markdown(f"## :red[⚠️ {pred} - Needs Focus]")

    st.write(f"Based on my training data, students like you with {hours}hrs study get: {pred}")

st.caption("Day 7/30 | SVIST CP Lab 2/7 | Now with Real ML Model")
