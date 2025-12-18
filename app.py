import streamlit as st
import numpy as np
import joblib
import keras
from pathlib import Path

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Asthma Prediction", layout="centered")
st.title("🫁 Asthma Prediction")

# =========================
# Paths
# =========================
MODEL_PATH = Path("hybrid_model_fixed.h5")
SCALER_PATH = Path("scaler.pkl")
CONFIG_PATH = Path("model_config.pkl")

DEFAULT_THRESHOLD = 0.65

# =========================
# Loaders
# =========================
@st.cache_resource
def load_model():
    return keras.saving.load_model(str(MODEL_PATH), compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load(str(SCALER_PATH))

@st.cache_resource
def load_config():
    return joblib.load(str(CONFIG_PATH))

model = load_model()
scaler = load_scaler()
cfg = load_config()

FEATURE_ORDER = cfg["feature_names"]
BEST_THRESHOLD = float(cfg.get("best_threshold", DEFAULT_THRESHOLD))

# =========================
# Helpers (UI only)
# =========================
def pick(label, options_dict, default_key):
    keys = list(options_dict.keys())
    default_index = keys.index(default_key)
    choice = st.selectbox(label, keys, index=default_index)
    return options_dict[choice]

def yes_no_code(label, default="No"):
    options = ["No", "Yes"]
    idx = 0 if default == "No" else 1
    choice = st.selectbox(label, options, index=idx)
    return 1 if choice == "Yes" else 0

# =========================
# UI – Questionnaire
# =========================

st.subheader("Personal Information")

Age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)

Gender = pick("Gender", {"Male": 0, "Female": 1}, "Male")
Ethnicity = pick(
    "Ethnicity",
    {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3},
    "Caucasian",
)
EducationLevel = pick(
    "Education Level",
    {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3},
    "None",
)

st.divider()

st.subheader("Health & Lifestyle Information")

st.markdown("**Calculated BMI**")

height_cm = st.number_input("Height (cm)", 0.0, 250.0, 0.0, 1.0)
weight_kg = st.number_input("Weight (kg)", 0.0, 300.0, 0.0, 1.0)

if height_cm > 0:
    BMI = round(weight_kg / ((height_cm / 100.0) ** 2), 2)
else:
    BMI = 0.0

st.info(f"Calculated BMI: **{BMI}**")

Smoking = yes_no_code("Smoking")

PhysicalActivity = st.slider("Physical Activity (0–10)", 0, 10, 0, 1)
DietQuality = st.slider("Diet Quality (0–10)", 0, 10, 0, 1)
SleepQuality = st.slider("Sleep Quality (4–10)", 4, 10, 4, 1)

PollutionExposure = st.slider("Pollution Exposure (0–10)", 0, 10, 0, 1)
PollenExposure = st.slider("Pollen Exposure (0–10)", 0, 10, 0, 1)
DustExposure = st.slider("Dust Exposure (0–10)", 0, 10, 0, 1)

PetAllergy = yes_no_code("Pet Allergy")
FamilyHistoryAsthma = yes_no_code("Family History of Asthma")
HistoryOfAllergies = yes_no_code("History of Allergies")
Eczema = yes_no_code("Eczema")
HayFever = yes_no_code("Hay Fever")
GastroesophagealReflux = yes_no_code("Gastroesophageal Reflux")

Wheezing = yes_no_code("Wheezing")
ShortnessOfBreath = yes_no_code("Shortness of Breath")
ChestTightness = yes_no_code("Chest Tightness")
Coughing = yes_no_code("Coughing")
NighttimeSymptoms = yes_no_code("Nighttime Symptoms")
ExerciseInduced = yes_no_code("Exercise Induced Symptoms")

st.divider()

# =========================
# Fixed Values
# =========================
PatientID = 6200.0
LungFunctionFEV1 = 2.5
LungFunctionFVC = 3.5

# =========================
# Build Input Vector
# =========================
user_values = {
    "PatientID": PatientID,
    "Age": Age,
    "Gender": Gender,
    "Ethnicity": Ethnicity,
    "EducationLevel": EducationLevel,
    "BMI": BMI,
    "Smoking": Smoking,
    "PhysicalActivity": PhysicalActivity,
    "DietQuality": DietQuality,
    "SleepQuality": SleepQuality,
    "PollutionExposure": PollutionExposure,
    "PollenExposure": PollenExposure,
    "DustExposure": DustExposure,
    "PetAllergy": PetAllergy,
    "FamilyHistoryAsthma": FamilyHistoryAsthma,
    "HistoryOfAllergies": HistoryOfAllergies,
    "Eczema": Eczema,
    "HayFever": HayFever,
    "GastroesophagealReflux": GastroesophagealReflux,
    "LungFunctionFEV1": LungFunctionFEV1,
    "LungFunctionFVC": LungFunctionFVC,
    "Wheezing": Wheezing,
    "ShortnessOfBreath": ShortnessOfBreath,
    "ChestTightness": ChestTightness,
    "Coughing": Coughing,
    "NighttimeSymptoms": NighttimeSymptoms,
    "ExerciseInduced": ExerciseInduced,
}

X = np.array([[user_values[f] for f in FEATURE_ORDER]], dtype=np.float32)

# =========================
# Predict + Styled Advice Box
# =========================
if st.button("Predict", type="primary"):
    X_scaled = scaler.transform(X)
    X_model = X_scaled.reshape((1, X_scaled.shape[1], 1))

    prob = float(model.predict(X_model, verbose=0)[0][0])

    symptom_score = (
        Wheezing +
        ShortnessOfBreath +
        ChestTightness +
        NighttimeSymptoms +
        ExerciseInduced
    )

    CLINICAL_OVERRIDE = symptom_score >= 3
    final_pred = 1 if CLINICAL_OVERRIDE else int(prob >= BEST_THRESHOLD)

    st.subheader("Result")

    if final_pred == 1:
        st.error("High likelihood of Asthma")

        st.markdown(
            """
            <div style="
                background-color:#f8d7da;
                padding:18px;
                border-radius:12px;
                color:#7f1d1d;
                font-size:16px;
                line-height:1.6;
            ">
            Work closely with a healthcare provider to identify and avoid asthma triggers,
            follow a personalized Asthma Action Plan, and maintain a healthy lifestyle.
            Regular monitoring and early intervention can help reduce symptoms and prevent exacerbations.
            </div>
            """,
            unsafe_allow_html=True,
        )



    else:
        st.success("Low likelihood of Asthma")

        st.markdown(
            """
            <div style="
                background-color:#d8f3dc;
                padding:18px;
                border-radius:12px;
                color:#1b4332;
                font-size:16px;
                line-height:1.6;
            ">
            Focus on maintaining good respiratory health by staying physically active,
            avoiding environmental pollutants, and adopting a balanced, healthy lifestyle.
            Regular checkups and awareness of early symptoms can help minimize future risks.
            </div>
            """,
            unsafe_allow_html=True,
        )


