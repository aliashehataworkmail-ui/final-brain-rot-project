import joblib
model = joblib.load("model.joblib")
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64



# Page configuration
st.set_page_config(
page_title="Test Yourself: Are you brain rotted?", page_icon="🧠", layout="centered"
)

# Custom CSS for styling, colors, and fonts
st.markdown("""
<style>
@import url('https://fonts.google.com/share?selection.family=Gelasio:ital,wght@0,400..700;1,400..700');

html, body, [class*="css"] {
font-family: 'Gelasio', serif;
}

/* Change main background color */
.stApp {
background-color: #081c15;
color: #f8f5f0;
}

#/* Style buttons to look sleek and modern */
.stButton>button {
background: linear-gradient(45deg, #FF4B4B, #FF914D);
color: white;
border-radius: 12px;
padding: 0.5rem 1rem;
border: none;
font-weight: 600;
box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
}

.stButton>button:hover {
background: linear-gradient(45deg, #ff3333, #ff7b25);
color: white;
}

/* Style form question labels to be a creamy color */
.stNumberInput label, .stSelectbox label, .stSlider label {
color: #FDF6EC !important;
}

/* 2. Make error box text readable if it shows up */
.stError {
background-color: #FDF6EC !important;
color: #1a1a1a !important;
}


/* 3. Change all button text color to black */
.stButton>button, .stFormSubmitButton>button {
color: #000000 !important;
}

.stButton>button:hover, .stFormSubmitButton>button:hover {
color: #000000 !important;
}
</style>
""",
unsafe_allow_html=True,
)

# Initialize session state for page navigation
if "page" not in st.session_state:
st.session_state.page = 1


def go_to_page(page_num):
st.session_state.page = page_num


# --- PAGE 1: Welcome & About ---
if st.session_state.page == 1:
st.title("Test Yourself: Are you brain rotted?🧠💨")
st.image(
"image1.png", width=250
)

st.markdown("### About Section")
st.write(
"Welcome! This app uses machine learning to analyze your daily digital"
" habits, screen time, and lifestyle factors to predict your digital"
" dependency score."
)

st.write("")
if st.button("Click to start 🤯"):
go_to_page(2)


# --- PAGE 2: Questions Form ---
elif st.session_state.page == 2:
st.title("Test Yourself: Are you brain rotted???")
st.subheader("Please fill out your lifestyle details:")

with st.form("prediction_form"):
# Question 1: Age
age = st.number_input(
"Question 1: How old are you?", min_value=1, max_value=100, value=20
)

# Question 2: Gender
gender = st.selectbox(
"Question 2: What's your gender?",
options=["Male", "Female"],
)

# Question 3: Region
region = st.selectbox(
"Question 3: What's your region?",
options=["Europe", "Asia", "North America", "Africa", "South America", "Middle East"],
)

# Question 4 onwards: Digital & Lifestyle habits
screen_time = st.number_input(
"Question 4: What is your average daily screen time (in hours)?",
min_value=0.0,
max_value=24.0,
value=4.0,
)

unlocks = st.number_input(
"How many times do you unlock your phone daily? (check screen time settings)",
min_value=0,
max_value=500,
value=50,
)

notifications = st.number_input(
"How many notifications do you get daily?",
min_value=0,
max_value=2000,
value=100,
)

sleep_hours = st.number_input(
"How many hours do you sleep daily?",
min_value=0.0,
max_value=24.0,
value=7.0,
)

sleep_quality = st.slider(
"Rate your quality of sleep from 1 to 5", min_value=1, max_value=5, value=3
)

stress_level = st.slider(
"Rate your stress level from 1 to 10", min_value=1, max_value=10, value=5
)

focus_level = st.slider(
"From 0 to 100, what would you rate your focus level?",
min_value=0,
max_value=100,
value=50,
)

productivity = st.slider(
"What would you say is your level of productivity (1 to 100)?",
min_value=0,
max_value=100,
value=50,
)

submitted = st.form_submit_button("Predict my digital dependency score")

if submitted:
# Store inputs in session state to use on page 3
# Note: Ensure the features match what your trained model expects!
st.session_state.input_data = {
"age": age,
"gender": gender,
"region": region,
"device_hours_per_day": screen_time,
"phone_unlocks": unlocks,
"notifications_per_day": notifications,
"sleep_hours": sleep_hours,
"sleep_quality": sleep_quality,
"stress_level": stress_level,
"focus_score": focus_level,
"productivity_score": productivity,
}
go_to_page(3)



# --- PAGE 3: Results ---
elif st.session_state.page == 3:
st.title("Results")

st.markdown("Your digital dependency score is......")

# Load model and make prediction
try:
with open("model.joblib", "rb") as f:
model = joblib.load(f)
with open("scaler.joblib", "rb") as f:
scaler = joblib.load(f)
with open("model_columns.pkl", "rb") as f:
model_columns = pickle.load(f)

target_column = ["digital_dependence_score"]
model_columns = [col for col in model_columns if col not in target_column]

# 2. Get user input saved from Page 2
user_inputs = st.session_state.input_data

# 3. Build a raw dataframe using the exact keys saved in session state
input_data_raw = pd.DataFrame(
[
[
user_inputs["age"],
user_inputs["gender"],
user_inputs["region"],
user_inputs["device_hours_per_day"],
user_inputs["phone_unlocks"],
user_inputs["notifications_per_day"],
user_inputs["sleep_hours"],
user_inputs["sleep_quality"],
user_inputs["stress_level"],
user_inputs["focus_score"],
user_inputs["productivity_score"],
]
],
columns=[
"age",
"gender",
"region",
"device_hours_per_day",
"phone_unlocks",
"notifications_per_day",
"sleep_hours",
"sleep_quality",
"stress_level",
"focus_score",
"productivity_score",
],
)

# 4. Apply get_dummies to encode gender and region
input_encoded = pd.get_dummies(input_data_raw)

# 5. Reindex to match the exact training columns (fills missing dummy columns with 0)
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# CRITICAL: Scale the user input using the exact same scaler from training!
input_scaled = scaler.transform(input_final)

# Predict score
prediction = model.predict(input_scaled)[0]
score = round(prediction, 2)

except Exception as e:
st.error(f"Something went wrong: {e}")
score = 70

# Display Score Big
st.markdown(f"# **{score}**")

# Brain rot message logic based on score
if score > 50:
st.error("unfortunately you ARE brain rotted🧠 👻")
st.image(
"image3.png", width=250
)
else:
st.success("You are safe! Your brain is clean of major digital rot 🧘‍♂️")
st.image(
"image2.png", width=250
)

st.write("")
if st.button("🔄 Take Test Again"):
go_to_page(1)
