import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- USER CREDENTIALS ----------------
USER_CREDENTIALS = {
    "user1": "password123",
    "admin": "admin123"
}

def logout():
    st.session_state.logged_in = False
    st.success("Logged out successfully")

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Final Year Project | AI & DS</h4>", unsafe_allow_html=True)
    st.write("---")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

# ---------------- MAIN APP ----------------
if st.session_state.logged_in:
    st.sidebar.button("Logout", on_click=logout)

    model = pickle.load(open("model.pkl", "rb"))

    st.title("🩺 Diabetes Prediction Dashboard")
    st.write("Predict diabetes risk using machine learning.")
    st.write("---")

    # ---------------- SIDEBAR INPUTS ----------------
    st.sidebar.header("Patient Medical Details")

    preg = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
    bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
    skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 79)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("Age", 1, 120, 30)

    # ---------------- VALIDATION ----------------
    if glucose == 0:
        st.warning("⚠️ Glucose level should not be zero")
    if bmi < 10 or bmi > 60:
        st.warning("⚠️ BMI value seems unrealistic")

    # ---------------- SUMMARY ----------------
    st.subheader("📋 Patient Summary")
    st.write(f"""
    - **Age:** {age}
    - **BMI:** {bmi}
    - **Glucose:** {glucose}
    - **Blood Pressure:** {bp}
    """)

    # ---------------- PREDICTION ----------------
    if st.button("Predict Diabetes"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100

        # ---------------- RESULT ----------------
        if prediction == 1:
            st.error(f"⚠️ Person is likely **Diabetic**")
        else:
            st.success(f"✅ Person is **Not Diabetic**")

        st.info(f"📊 **Diabetes Risk Probability:** {probability:.2f}%")

        # ---------------- RISK CHART ----------------
        st.subheader("📊 Risk Visualization")
        fig, ax = plt.subplots()
        ax.bar(["Diabetes Risk"], [probability])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Risk Percentage")
        st.pyplot(fig)

        # ---------------- MEDICAL RECOMMENDATIONS ----------------
        st.subheader("🩺 Medical Recommendations")

        if probability > 70:
            st.write("🔴 **High Risk:** Consult a doctor immediately, maintain strict diet control, and regular exercise.")
        elif probability > 40:
            st.write("🟠 **Moderate Risk:** Lifestyle changes, regular monitoring, and balanced diet recommended.")
        else:
            st.write("🟢 **Low Risk:** Maintain healthy lifestyle and periodic checkups.")

        # ---------------- SAVE HISTORY ----------------
        st.session_state.history.append({
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi,
            "Risk %": f"{probability:.2f}",
            "Result": "Diabetic" if prediction == 1 else "Non-Diabetic"
        })

    # ---------------- HISTORY & DOWNLOAD ----------------
    if st.session_state.history:
        st.subheader("🧾 Prediction History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            csv,
            "diabetes_prediction_history.csv",
            "text/csv"
        )

    # ---------------- FOOTER ----------------
    st.write("---")
    st.caption("Final Year Project | Diabetes Prediction using Machine Learning")
