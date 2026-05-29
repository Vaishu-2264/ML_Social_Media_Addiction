import streamlit as st
import pandas as pd
import joblib
import time
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# Load Model & Encoders
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():

    artifacts = joblib.load("model_prod_file_SM_addiction.pkl")

    return (
        artifacts["model"],
        artifacts["cat_encod"],
        artifacts["num_encod"],
        artifacts["label_encod"]
    )

model, cat_encoder, scaler, label_encoder = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Column Lists
# ─────────────────────────────────────────────────────────────
NUM_COLS = [
    "age",
    "daily_usage_hours",
    "num_platforms_used",
    "avg_session_minutes",
    "night_usage",
    "mental_health_score",
    "screen_time_before_sleep"
]

CAT_COLS = [
    "gender",
    "country",
    "primary_platform",
    "purpose"
]

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
import base64

# Function to load image
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("social_media.jpg")

# Background CSS
page_bg = f"""
<style>

/* Background image */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Transparent header */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Main app container */
[data-testid="stVerticalBlock"] {{
    
    background: rgba(255, 255, 255, 0.25);

    backdrop-filter: blur(8px);

    padding: 20px;

    border-radius: 20px;
}}

/* Headings and labels */
h1, h2, h3, h4, h5, h6, label, p {{
    color: black !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background-color: rgba(255,255,255,0.50);
    color: black !important;
    border-radius: 10px;
}}

/* Slider container */
.stSlider {{
    background-color: rgba(255,255,255,0.15);
    padding: 10px;
    border-radius: 10px;
}}

/* Slider text */
.stSlider label {{
    color: white !important;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("# 📱 Social Media Addiction Predictor")

st.markdown("""
Fill in the details below to analyze the predicted level of social media addiction.
""")

st.divider()

# ─────────────────────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown("## 👤 User Information")

    c1, c2 = st.columns(2)

    with c1:

        age = st.slider(
            "Age", 
            1, 100, 25
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        country = st.selectbox(
            "Country",
            ["India", "USA", "UK", "Canada",
             "Australia", "Germany", "Brazil"]
        )

        primary_platform = st.selectbox(
            "Primary Platform",
            ["Instagram", "TikTok", "Twitter",
             "YouTube", "Snapchat"]
        )

    with c2:

        daily_usage_hours = st.slider(
            "Daily Usage Hours",
            0.0, 12.0, 4.0
        )

        num_platforms_used = st.slider(
            "Number of Platforms Used",
            1, 5, 3
        )

        avg_session_minutes = st.slider(
            "Average Session Minutes",
            1.0, 120.0, 30.0
        )

        screen_time_before_sleep = st.slider(
            "Screen Time Before Sleep (minutes)",
            0.0, 180.0, 30.0
        )

    st.markdown("## 🧠 Behavioral Information")

    c3, c4 = st.columns(2)

    with c3:

        purpose = st.selectbox(
            "Main Purpose of Usage",
            ["Education", "Entertainment",
             "Socializing", "News",
             "Content Creation"]
        )

        night_usage = st.selectbox(
            "Night Usage",
            [0, 1]
        )

    with c4:

        mental_health_score = st.slider(
            "Mental Health Score",
            1.0, 10.0, 5.0
        )

    st.markdown("")

    submitted = st.form_submit_button(
        "Predict Addiction Level",
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if submitted:

    with st.spinner("🔍 Analyzing user behavior..."):

        time.sleep(2)

    # ── Numerical Data ──────────────────────────────────────

    # Columns used during scaling
    SCALE_COLS = [
        "age",
        "daily_usage_hours",
        "avg_session_minutes",
        "mental_health_score",
        "screen_time_before_sleep"
    ]

    # Data to scale
    num_df = pd.DataFrame([[
        age,
        daily_usage_hours,
        avg_session_minutes,
        mental_health_score,
        screen_time_before_sleep
    ]], columns=SCALE_COLS)

    # Scale numerical columns
    num_scaled = pd.DataFrame(
        scaler.transform(num_df),
        columns=SCALE_COLS
    )

    # Numeric columns NOT scaled
    extra_num_df = pd.DataFrame([[
        num_platforms_used,
        night_usage
    ]], columns=[
        "num_platforms_used",
        "night_usage"
    ])

    # ── Categorical Data ───────────────────────────────────
    cat_df = pd.DataFrame([[
        gender,
        country,
        primary_platform,
        purpose
    ]], columns=CAT_COLS)

    cat_encoded = cat_encoder.transform(cat_df)

    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()

    cat_df_encoded = pd.DataFrame(
        cat_encoded,
        columns=cat_encoder.get_feature_names_out(CAT_COLS)
    )

    # ── Final Dataset ──────────────────────────────────────
    final_df = pd.concat(
    [num_scaled, extra_num_df, cat_df_encoded],
    axis=1
    )

    # ── Match Model Columns ────────────────────────────────
    expected_cols = model.feature_names_in_

    for col in expected_cols:

        if col not in final_df.columns:
            final_df[col] = 0

    final_df = final_df[expected_cols]

    # ── Prediction ─────────────────────────────────────────
    prediction = model.predict(final_df)

    prediction_label = label_encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(final_df)[0]

    # ───────────────────────────────────────────────────────
    # Results
    # ───────────────────────────────────────────────────────
    st.divider()

    if prediction_label == "Low":

        st.success("## ✅ Addiction Level: LOW")

        st.markdown("""
The user shows healthy and balanced social media usage behavior.
""")

    elif prediction_label == "Medium":

        st.warning("## ⚠️ Addiction Level: MEDIUM")

        st.markdown("""
The user shows moderate dependency on social media platforms.
""")

    else:

        st.error("## 🚨 Addiction Level: HIGH")

        st.markdown("""
The user may have strong addictive social media usage patterns.
""")

    # ── Probability Scores ────────────────────────────────
    st.markdown("## 📊 Prediction Confidence")

    class_names = label_encoder.inverse_transform(model.classes_)

    prob_df = pd.DataFrame({
        "Addiction Level": class_names,
        "Probability": probabilities
    })

    st.dataframe(prob_df, use_container_width=True)