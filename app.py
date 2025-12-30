import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_feature_engineering
from src.explanation import generate_explanations

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "logreg_l1_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    "<h1 style='font-size:34px; margin-bottom:0.2rem;'>Employee Attrition Risk Predictor</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
Predict **employee attrition risk** using a machine learning model trained on structured HR data.  
Designed for **HR and People Analytics teams**.
"""
)

st.divider()

tab_predict, tab_explain, tab_model, tab_about = st.tabs(
    ["Risk Prediction", "Explanation", "Model Overview", "About"]
)


# ------------------------------------------------------------------
# Helper: raw input builder
# ------------------------------------------------------------------
def build_input_dataframe(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    defaults = {
        "DailyRate": 800,
        "HourlyRate": 65,
        "MonthlyRate": 14000,
        "PercentSalaryHike": 15,
        "NumCompaniesWorked": 2,
        "Education": 3,
        "EducationField": "Life Sciences",
        "JobLevel": 2,
        "StockOptionLevel": 1,
        "TrainingTimesLastYear": 2,
        "YearsAtCompany": min(int(user_input["TotalWorkingYears"]), 40),
        "YearsInCurrentRole": max(int(user_input["TotalWorkingYears"]) - 1, 0),
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": max(int(user_input["TotalWorkingYears"]) - 1, 0),
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
    }

    for col, val in defaults.items():
        df[col] = val

    return df


# ------------------------------------------------------------------
# Risk Prediction Tab
# ------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.25, 1])

    with left:
        with st.form("attrition_form"):
            age = st.number_input("Age", 18, 65, 30)
            distance = st.slider("Commute Distance (1 = close, 29 = far)", 1, 29, 5)
            income = st.number_input("Monthly Income (USD)", 500, 50000, 5000, step=500)
            experience = st.number_input("Total Working Years", 0, 40, 5)

            business_travel = st.selectbox(
                "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
            )
            overtime = st.selectbox("Overtime Required", ["No", "Yes"])
            department = st.selectbox(
                "Department", ["Research & Development", "Sales", "Human Resources"]
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
            )
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced"]
            )

            job_involvement = st.slider("Job Involvement (1–4)", 1, 4, 3)
            work_life = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
            job_sat = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
            env_sat = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)

            submitted = st.form_submit_button("Predict Attrition Risk")

    with right:
        if submitted:
            user_input = {
                "Age": age,
                "DistanceFromHome": distance,
                "MonthlyIncome": income,
                "TotalWorkingYears": experience,
                "BusinessTravel": business_travel,
                "OverTime": overtime,
                "Department": department,
                "Gender": gender,
                "JobRole": job_role,
                "MaritalStatus": marital_status,
                "JobInvolvement": job_involvement,
                "WorkLifeBalance": work_life,
                "JobSatisfaction": job_sat,
                "EnvironmentSatisfaction": env_sat,
            }

            input_raw = build_input_dataframe(user_input)
            input_fe = add_feature_engineering(input_raw, mode="inference")

            risk_score = float(model.predict_proba(input_fe)[0, 1])
            risk_pct = int(round(risk_score * 100))

            # Save state
            st.session_state["input_fe"] = input_fe
            st.session_state["risk_score"] = risk_score

            preprocessor = model.named_steps["preprocessing"]
            clf = model.named_steps["model"]

            feature_names = preprocessor.get_feature_names_out()
            coefs = clf.coef_[0]
            X_trans = preprocessor.transform(input_fe)

            coef_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coef": coefs,
                    "contribution": X_trans[0] * coefs,
                }
            )

            st.session_state["coef_df"] = coef_df

            # ---------------- CLEAN ENTERPRISE GAUGE ----------------
            needle_color = (
                "#000000"
                if risk_score < 0.30
                else "#D5DF17" if risk_score < 0.60 else "#610A00"
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={
                        "suffix": "%",
                        "font": {"size": 44, "color": "#111"},
                    },
                    title={
                        "text": "Estimated Attrition Risk",
                        "font": {"size": 18, "color": "#ffffff"},
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {
                            "color": needle_color,
                            "thickness": 0.3,
                        },
                        "steps": [
                            {"range": [0, 30], "color": "#057F0F"},
                            {"range": [30, 60], "color": "#D5DF17"},
                            {"range": [60, 100], "color": "#610A00"},
                        ],
                    },
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(t=50, b=10, l=10, r=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            if risk_score < 0.30:
                st.success("Low attrition risk detected.")
            elif risk_score < 0.60:
                st.warning("Moderate attrition risk detected.")
            else:
                st.error("High attrition risk detected.")

            st.caption(
                "Risk score represents the estimated probability of employee attrition based on historical HR patterns."
            )

        else:
            st.info("Fill the form and click Predict.")

# ------------------------------------------------------------------
# Explanation Tab
# ------------------------------------------------------------------
with tab_explain:
    if "coef_df" not in st.session_state:
        st.info("Generate a prediction to see the explanation.")
    else:
        st.subheader("Key Drivers Behind the Prediction")

        explanations = generate_explanations(
            st.session_state["coef_df"][["feature", "contribution"]]
        )

        for item in explanations:
            st.markdown(f"- **{item['text']}**")

# ------------------------------------------------------------------
# Model Overview
# ------------------------------------------------------------------
with tab_model:
    st.markdown(
        """
**Model:** Logistic Regression (L1)  
**Strengths:** interpretability, stability, feature selection  
**Use case:** HR decision support (not causal inference)
"""
    )

# ------------------------------------------------------------------
# About
# ------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
End-to-end ML project:
- Data cleaning
- Feature engineering
- Model training
- Explainable inference
- Streamlit deployment
"""
    )
