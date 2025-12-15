import streamlit as st
import pickle
import numpy as np

# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# SESSION STATE SETUP
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------
# USER CREDENTIALS
# -------------------------
USER_CREDENTIALS = {
    "Ritika": "111",
    "tina": "xyz"
}

# -------------------------
# LOGOUT FUNCTION
# -------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully! Please login again.")

# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #306998;'>Final Year Project | YBI Foundation Internship</h4>", unsafe_allow_html=True)
    st.write("---")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Welcome {username}!")
        else:
            st.error("❌ Invalid username or password")

# -------------------------
# MAIN APP
# -------------------------
if st.session_state.logged_in:
    # Sidebar logout
    st.sidebar.button("Logout", on_click=logout)

    # Load the trained model
    model = pickle.load(open("model.pkl", "rb"))

    # Welcome message
    st.markdown(f"<h2 style='color: #FF4B4B;'>Welcome {st.session_state.username}!</h2>", unsafe_allow_html=True)
    st.write("Enter patient details in the sidebar to predict diabetes risk.")
    st.write("---")

    # -------------------------
    # SIDEBAR INPUTS
    # -------------------------
    st.sidebar.header("Patient Information")
    preg = st.sidebar.slider("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 1, 120, 30)

    # -------------------------
    # PREDICTION BUTTON
    # -------------------------
    if st.button("Predict"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = model.predict(data)

        if result[0] == 1:
            st.error("⚠️ Person is Diabetic")
        else:
            st.success("✅ Person is NOT Diabetic")



