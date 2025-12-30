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
# Translation Dictionary (UI Static Texts)
# ------------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "Employee Attrition Risk Predictor",
        "intro": """
        This application demonstrates a **practical machine learning approach**
        to estimating employee attrition risk using structured HR data.

        It was built to translate analytical modeling into **clear, explainable insights**
        that can support **HR and People Analytics decision-making**.
        """,
        "tabs": ["Risk Prediction", "Explanation", "Model Overview", "About"],
        # Inputs
        "lbl_age": "Age",
        "lbl_dist": "Commute Distance (1 = close, 29 = far)",
        "lbl_income": "Monthly Income (USD)",
        "lbl_exp": "Total Working Years",
        "lbl_travel": "Business Travel",
        "lbl_overtime": "Overtime Required",
        "lbl_dept": "Department",
        "lbl_gender": "Gender",
        "lbl_role": "Job Role",
        "lbl_marital": "Marital Status",
        "lbl_involv": "Job Involvement (1–4)",
        "lbl_worklife": "Work-Life Balance (1–4)",
        "lbl_jobsat": "Job Satisfaction (1–4)",
        "lbl_envsat": "Environment Satisfaction (1–4)",
        "btn_predict": "Predict Attrition Risk",
        "info_fill": "Fill the form and click Predict.",
        # Results
        "res_low": "Low attrition risk detected.",
        "res_mod": "Moderate attrition risk detected.",
        "res_high": "High attrition risk detected.",
        "res_caption": "Risk score represents the estimated probability of employee attrition based on historical HR patterns.",
        "gauge_title": "Attrition Risk",
        # Explanations
        "exp_wait": "Generate a prediction to see the explanation.",
        "exp_title": "Key Drivers Behind the Prediction",
        "exp_card_title": "Attrition driver",
        "exp_inc": "Increases risk",
        "exp_red": "Reduces risk",
        "exp_caption": "Drivers are derived from model feature contributions for this specific employee. They indicate statistical association, not causality.",
        # Model Overview
        "mo_title": "Model Overview",
        "mo_intro": "This application uses a **regularized Logistic Regression (L1)** model to estimate **attrition risk**. The output is a **probability score** (0–100%) intended for **decision support**, not for automated decisions.",
        "mo_q1": "What model is used and why?",
        "mo_a1": """
        **Model type:** Logistic Regression with **L1 regularization** (feature selection)

        **Why this model:**
        - **Interpretable:** we can explain which factors push risk up or down for a specific employee.
        - **Stable baseline:** strong performance without overfitting in structured HR datasets.
        - **Feature selection:** L1 regularization reduces noisy signals by shrinking some coefficients to zero.
        """,
        "mo_q2": "How was the model evaluated?",
        "mo_a2": """
        **Evaluation approach**
        - **Train/Test split:** 80/20 with stratification (attrition rate preserved across splits)
        - **Cross-validation on training set:** used to estimate expected generalization performance
        - **Class imbalance handling:** model trained with `class_weight="balanced"` to avoid ignoring the minority attrition class
        """,
        "mo_q3": "Performance summary (interpreted for HR)",
        "mo_a3_intro": "Because attrition is relatively rare, **accuracy alone can be misleading**. We focus on metrics that reflect how well the model identifies attrition cases.",
        "mo_metrics": [
            "ROC-AUC (Test)",
            "Recall for Attrition (Test)",
            "Precision for Attrition (Test)",
        ],
        "mo_a3_desc": """
        **What these mean in plain terms**
        - **ROC-AUC (0.83):** the model is generally good at ranking higher-risk employees above lower-risk employees.
        - **Recall (0.68):** out of employees who actually left, the model correctly flags about **68%** as higher risk.
        - **Precision (0.39):** among employees flagged as higher risk, about **39%** actually left (the rest are false alarms).

        In People Analytics terms, this is a typical trade-off:
        - Higher **recall** helps avoid missing true attrition risk,
        - Lower **precision** means more employees may be flagged than will actually leave.
        """,
        "mo_q4": "Concrete example on the test set (counts)",
        "mo_a4": """
        On the held-out test set (n=294):
        - **True Attrition:** 47 employees 
        - **Predicted Attrition correctly (TP):** 32 
        - **Missed Attrition (FN):** 15 
        - **False Alarms (FP):** 51 
        - **Correct Non-Attrition (TN):** 196 

        These numbers depend on the default threshold (0.50).
        Different HR use cases may prefer different thresholds (e.g., more recall vs fewer false alarms).
        """,
        "mo_disc": "Important: The model captures statistical association from historical data. It does not prove causality and should not be used as the sole basis for HR actions.",
        # About
        "about_title": "About Me",
        "about_text": """
        ### Project purpose

        This project was developed as a practical application of the concepts learned during the  
        **IBM – Data Analysis with Python** program.

        The main objective was to go beyond isolated notebook exercises and build a  
        **complete, end-to-end machine learning solution**, including:

        - Data preparation and feature engineering  
        - Model training and evaluation  
        - Explainable, business-oriented predictions  
        - Interactive deployment using Streamlit  

        The project also serves as a **portfolio piece**, demonstrating applied data science
        and machine learning skills in a real-world HR analytics context.

        ---

        ### What this project demonstrates

        - Structured data analysis and feature engineering  
        - Handling class imbalance in binary classification problems  
        - Interpretable machine learning using L1-regularized Logistic Regression  
        - Translating model outputs into clear, HR-friendly explanations  
        - Building a production-style inference application  

        This application focuses on **clarity, interpretability, and usability** rather than
        black-box modeling approaches.

        ---

        ### Links & profiles

        - **LinkedIn:** https://www.linkedin.com/in/kaganmart9/

        - **GitHub:** https://github.com/kaganmart9

        - **Medium:** https://medium.com/@kaganmart9

        - **Portfolio website:** https://kaganmart9.github.io/
        """,
        "about_caption": "This application is intended for educational and portfolio purposes. Model outputs reflect statistical associations based on historical data and should not be interpreted as causal conclusions.",
    },
    "Türkçe": {
        "title": "Çalışan Kaybı (Attrition) Risk Tahminleyicisi",
        "intro": """
        Bu uygulama, yapılandırılmış İK verilerini kullanarak çalışan kaybı riskini tahmin etmek için 
        **pratik bir makine öğrenimi yaklaşımı** sunmaktadır.

        Analitik modellemeyi, **İK ve İnsan Analitiği karar süreçlerini** destekleyebilecek 
        **net ve açıklanabilir içgörülere** dönüştürmek amacıyla oluşturulmuştur.
        """,
        "tabs": ["Risk Tahmini", "Açıklama", "Model Genel Bakış", "Hakkında"],
        # Inputs
        "lbl_age": "Yaş",
        "lbl_dist": "İşe Gidiş Mesafesi (1 = yakın, 29 = uzak)",
        "lbl_income": "Aylık Gelir (USD)",
        "lbl_exp": "Toplam Çalışma Yılı",
        "lbl_travel": "İş Seyahati",
        "lbl_overtime": "Fazla Mesai",
        "lbl_dept": "Departman",
        "lbl_gender": "Cinsiyet",
        "lbl_role": "İş Rolü/Pozisyon",
        "lbl_marital": "Medeni Durum",
        "lbl_involv": "İşe Katılım (1–4)",
        "lbl_worklife": "İş-Yaşam Dengesi (1–4)",
        "lbl_jobsat": "İş Tatmini (1–4)",
        "lbl_envsat": "Ortam Tatmini (1–4)",
        "btn_predict": "Ayrılma Riskini Tahmin Et",
        "info_fill": "Formu doldurun ve Tahmin Et butonuna tıklayın.",
        # Results
        "res_low": "Düşük ayrılma riski tespit edildi.",
        "res_mod": "Orta düzey ayrılma riski tespit edildi.",
        "res_high": "Yüksek ayrılma riski tespit edildi.",
        "res_caption": "Risk skoru, geçmiş İK verilerine dayanarak çalışanın işten ayrılma olasılığını temsil eder.",
        "gauge_title": "Ayrılma Riski",
        # Explanations
        "exp_wait": "Açıklamayı görmek için bir tahmin oluşturun.",
        "exp_title": "Tahminin Arkasındaki Temel Etkenler",
        "exp_card_title": "Ayrılma Etkeni",
        "exp_inc": "Riski artırır",
        "exp_red": "Riski azaltır",
        "exp_caption": "Etkenler, bu çalışana özel model katkı değerlerinden türetilmiştir. İstatistiksel ilişkiyi gösterir, nedenselliği değil.",
        # Model Overview
        "mo_title": "Model Genel Bakış",
        "mo_intro": "Bu uygulama, **ayrılma riskini** tahmin etmek için **L1 düzenlileştirmeli Lojistik Regresyon** modeli kullanır. Çıktı, otomatik kararlar için değil, **karar desteği** amaçlı bir **olasılık skorudur** (%0–100).",
        "mo_q1": "Hangi model kullanıldı ve neden?",
        "mo_a1": """
        **Model tipi:** L1 düzenlileştirmeli (özellik seçimi) Lojistik Regresyon

        **Neden bu model:**
        - **Yorumlanabilir:** Belirli bir çalışan için hangi faktörlerin riski artırıp azalttığını açıklayabiliriz.
        - **Kararlı temel:** Yapılandırılmış İK veri setlerinde aşırı öğrenme (overfitting) olmadan güçlü performans.
        - **Özellik seçimi:** L1 düzenlileştirme, bazı katsayıları sıfıra indirerek gürültülü sinyalleri azaltır.
        """,
        "mo_q2": "Model nasıl değerlendirildi?",
        "mo_a2": """
        **Değerlendirme yaklaşımı**
        - **Eğitim/Test ayrımı:** 80/20 tabakalı (ayrılma oranı her iki grupta korunarak)
        - **Eğitim setinde çapraz doğrulama:** Beklenen genelleme performansını tahmin etmek için kullanıldı.
        - **Sınıf dengesizliği yönetimi:** Azınlıktaki ayrılma sınıfını göz ardı etmemek için model `class_weight="balanced"` ile eğitildi.
        """,
        "mo_q3": "Performans özeti (İK için yorumlanmış)",
        "mo_a3_intro": "Ayrılma (attrition) nispeten nadir olduğundan, **yalnızca doğruluk (accuracy) yanıltıcı olabilir**. Modelin ayrılma durumlarını ne kadar iyi tespit ettiğini yansıtan metriklere odaklanıyoruz.",
        "mo_metrics": [
            "ROC-AUC (Test)",
            "Ayrılma için Duyarlılık (Recall)",
            "Ayrılma için Kesinlik (Precision)",
        ],
        "mo_a3_desc": """
        **Basit ifadelerle anlamları:**
        - **ROC-AUC (0.83):** Model, yüksek riskli çalışanları düşük riskli olanların önüne sıralamada genel olarak başarılıdır.
        - **Duyarlılık/Recall (0.68):** Gerçekten ayrılan çalışanlar arasından model yaklaşık **%68**'ini doğru bir şekilde yüksek riskli olarak işaretledi.
        - **Kesinlik/Precision (0.39):** Yüksek riskli olarak işaretlenen çalışanlar arasında yaklaşık **%39**'u gerçekten ayrıldı (gerisi yanlış alarm).

        İnsan Analitiği terimleriyle bu tipik bir takastır:
        - Yüksek **duyarlılık**, gerçek ayrılma risklerini kaçırmamayı sağlar,
        - Düşük **kesinlik**, gerçekten ayrılacak olandan daha fazla çalışanın işaretlenebileceği anlamına gelir.
        """,
        "mo_q4": "Test setinden somut örnek (sayılar)",
        "mo_a4": """
        Ayrılmış test setinde (n=294):
        - **Gerçek Ayrılma:** 47 çalışan 
        - **Doğru Tahmin Edilen Ayrılma (TP):** 32 
        - **Kaçırılan Ayrılma (FN):** 15 
        - **Yanlış Alarmlar (FP):** 51 
        - **Doğru Tespit Edilen Kalıcılar (TN):** 196 

        Bu sayılar varsayılan eşik değerine (0.50) bağlıdır.
        Farklı İK kullanım senaryoları farklı eşik değerlerini tercih edebilir.
        """,
        "mo_disc": "Önemli: Model, geçmiş verilerden istatistiksel ilişkileri yakalar. Nedenselliği kanıtlamaz ve İK aksiyonları için tek temel olarak kullanılmamalıdır.",
        # About
        "about_title": "Hakkımda",
        "about_text": """
        ### Proje amacı

        Bu proje, **IBM – Data Analysis with Python** programı sırasında öğrenilen kavramların pratik bir 
        uygulaması olarak geliştirilmiştir.

        Temel amaç, izole edilmiş not defteri alıştırmalarının ötesine geçmek ve aşağıdakileri içeren 
        **uçtan uca eksiksiz bir makine öğrenimi çözümü** oluşturmaktı:

        - Veri hazırlama ve özellik mühendisliği  
        - Model eğitimi ve değerlendirme  
        - Açıklanabilir, iş odaklı tahminler  
        - Streamlit kullanarak interaktif dağıtım  

        Proje aynı zamanda, gerçek dünya İK analitiği bağlamında uygulamalı veri bilimi 
        ve makine öğrenimi becerilerini gösteren bir **portfolyo çalışması** işlevi görmektedir.

        ---

        ### Bu proje neyi gösteriyor?

        - Yapılandırılmış veri analizi ve özellik mühendisliği  
        - İkili sınıflandırma problemlerinde sınıf dengesizliğinin yönetimi  
        - L1-düzenlileştirmeli Lojistik Regresyon kullanarak yorumlanabilir makine öğrenimi  
        - Model çıktılarını net, İK dostu açıklamalara dönüştürme  
        - Üretim tarzı bir tahmin uygulaması oluşturma  

        Bu uygulama, kara kutu modelleme yaklaşımlarından ziyade **netlik, yorumlanabilirlik ve kullanılabilirliğe** odaklanmaktadır.

        ---

        ### Bağlantılar & profiller

        - **LinkedIn:** https://www.linkedin.com/in/kaganmart9/

        - **GitHub:** https://github.com/kaganmart9

        - **Medium:** https://medium.com/@kaganmart9

        - **Portfolio web sitesi:** https://kaganmart9.github.io/
        """,
        "about_caption": "Bu uygulama eğitim ve portfolyo amaçlıdır. Model çıktıları geçmiş verilere dayalı istatistiksel ilişkileri yansıtır ve nedensel sonuçlar olarak yorumlanmamalıdır.",
    },
}

# ------------------------------------------------------------------
# Option Mappings (UI Display -> Model Value)
# ------------------------------------------------------------------
# The model expects specific English strings. We map Turkish UI selections back to them.
OPTION_MAP = {
    # Business Travel
    "Non-Travel": "Non-Travel",
    "Travel_Rarely": "Travel_Rarely",
    "Travel_Frequently": "Travel_Frequently",
    "Seyahat Yok": "Non-Travel",
    "Nadir Seyahat": "Travel_Rarely",
    "Sık Seyahat": "Travel_Frequently",
    # Overtime
    "No": "No",
    "Yes": "Yes",
    "Hayır": "No",
    "Evet": "Yes",
    # Department
    "Research & Development": "Research & Development",
    "Sales": "Sales",
    "Human Resources": "Human Resources",
    "Araştırma & Geliştirme": "Research & Development",
    "Satış": "Sales",
    "İnsan Kaynakları": "Human Resources",
    # Gender
    "Female": "Female",
    "Male": "Male",
    "Kadın": "Female",
    "Erkek": "Male",
    # Marital Status
    "Single": "Single",
    "Married": "Married",
    "Divorced": "Divorced",
    "Bekar": "Single",
    "Evli": "Married",
    "Boşanmış": "Divorced",
}

# Lists for UI Display
OPTIONS_UI = {
    "English": {
        "travel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        "overtime": ["No", "Yes"],
        "dept": ["Research & Development", "Sales", "Human Resources"],
        "gender": ["Female", "Male"],
        "role": [
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
        "marital": ["Single", "Married", "Divorced"],
    },
    "Türkçe": {
        "travel": ["Seyahat Yok", "Nadir Seyahat", "Sık Seyahat"],
        "overtime": ["Hayır", "Evet"],
        "dept": ["Araştırma & Geliştirme", "Satış", "İnsan Kaynakları"],
        "gender": ["Kadın", "Erkek"],
        "role": [
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
        "marital": ["Bekar", "Evli", "Boşanmış"],
    },
}

# ------------------------------------------------------------------
# Simple UI CSS (cards + spacing)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
/* Card layout */
.driver-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
@media (min-width: 900px) { .driver-grid { grid-template-columns: 1fr 1fr; } }

.driver-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.02);
}

.driver-title {
  font-size: 12px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.60);
  margin-bottom: 6px;
}

.driver-text {
  font-size: 15px;
  line-height: 1.35;
}

.badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  margin-left: 8px;
  border: 1px solid rgba(0,0,0,0.08);
}

.badge-risk { background: rgba(0,0,0,0.06); }

</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "logreg_l1_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Language Selector & Header
# ------------------------------------------------------------------
col_lang_1, col_lang_2 = st.columns([6, 1])
with col_lang_2:
    selected_lang = st.selectbox(
        "Language / Dil", ["English", "Türkçe"], label_visibility="collapsed"
    )

texts = TRANSLATIONS[selected_lang]

st.markdown(
    f"<h1 style='font-size:34px;margin-top: 0.2rem; margin-bottom:0.2rem;'>{texts['title']}</h1>",
    unsafe_allow_html=True,
)

st.markdown(texts["intro"])

st.divider()

tab_predict, tab_explain, tab_model, tab_about = st.tabs(texts["tabs"])


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
            age = st.number_input(texts["lbl_age"], 18, 65, 30)
            distance = st.slider(texts["lbl_dist"], 1, 29, 5)
            income = st.number_input(texts["lbl_income"], 500, 50000, 5000, step=500)
            experience = st.number_input(texts["lbl_exp"], 0, 40, 5)

            # Selectboxes with translated options
            business_travel = st.selectbox(
                texts["lbl_travel"], OPTIONS_UI[selected_lang]["travel"]
            )
            overtime = st.selectbox(
                texts["lbl_overtime"], OPTIONS_UI[selected_lang]["overtime"]
            )
            department = st.selectbox(
                texts["lbl_dept"], OPTIONS_UI[selected_lang]["dept"]
            )
            gender = st.selectbox(
                texts["lbl_gender"], OPTIONS_UI[selected_lang]["gender"]
            )
            job_role = st.selectbox(
                texts["lbl_role"], OPTIONS_UI[selected_lang]["role"]
            )
            marital_status = st.selectbox(
                texts["lbl_marital"], OPTIONS_UI[selected_lang]["marital"]
            )

            job_involvement = st.slider(texts["lbl_involv"], 1, 4, 3)
            work_life = st.slider(texts["lbl_worklife"], 1, 4, 3)
            job_sat = st.slider(texts["lbl_jobsat"], 1, 4, 3)
            env_sat = st.slider(texts["lbl_envsat"], 1, 4, 3)

            submitted = st.form_submit_button(texts["btn_predict"])

    with right:
        if submitted:
            # Map translated UI selections back to model-compatible English strings
            user_input = {
                "Age": age,
                "DistanceFromHome": distance,
                "MonthlyIncome": income,
                "TotalWorkingYears": experience,
                "BusinessTravel": OPTION_MAP.get(business_travel, business_travel),
                "OverTime": OPTION_MAP.get(overtime, overtime),
                "Department": OPTION_MAP.get(department, department),
                "Gender": OPTION_MAP.get(gender, gender),
                "JobRole": job_role,  # Job roles assumed standard/English in list
                "MaritalStatus": OPTION_MAP.get(marital_status, marital_status),
                "JobInvolvement": job_involvement,
                "WorkLifeBalance": work_life,
                "JobSatisfaction": job_sat,
                "EnvironmentSatisfaction": env_sat,
            }

            input_raw = build_input_dataframe(user_input)
            input_fe = add_feature_engineering(input_raw, mode="inference")

            # ---- Model inference ----
            risk_score = float(model.predict_proba(input_fe)[0, 1])
            risk_pct = int(round(risk_score * 100))

            # ---- Persist state for Explanation tab ----
            st.session_state["input_fe"] = input_fe
            st.session_state["risk_score"] = risk_score

            # ---- Build per-feature contribution table for explanations ----
            preprocessor = model.named_steps["preprocessing"]
            clf = model.named_steps["model"]

            feature_names = preprocessor.get_feature_names_out()
            coefs = clf.coef_[0]
            X_trans = preprocessor.transform(input_fe)

            coef_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coef": coefs,
                    "contribution": (X_trans[0] * coefs),
                }
            ).sort_values("contribution", ascending=False)

            st.session_state["coef_df"] = coef_df

            # -------- Color logic (for gauge + center text) --------
            if risk_score < 0.30:
                main_color = "#057F0F"
            elif risk_score < 0.60:
                main_color = "#D5DF17"
            else:
                main_color = "#610A00"

            # -------- Gauge (center % text) --------
            fig = go.Figure(
                go.Indicator(
                    mode="gauge",
                    value=risk_pct,
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": main_color, "thickness": 0.28},
                        "steps": [
                            {"range": [0, 30], "color": "#057F0F"},
                            {"range": [30, 60], "color": "#D5DF17"},
                            {"range": [60, 100], "color": "#610A00"},
                        ],
                    },
                )
            )

            fig.add_annotation(
                x=0.5,
                y=0.47,
                text=f"<b>{risk_pct}%</b>",
                font=dict(size=46, color=main_color),
                showarrow=False,
            )
            fig.add_annotation(
                x=0.5,
                y=0.32,
                text=texts["gauge_title"],
                font=dict(size=14, color="rgba(255,255,255,0.55)"),
                showarrow=False,
            )

            fig.update_layout(
                height=320,
                margin=dict(t=20, b=10, l=10, r=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            if risk_score < 0.30:
                st.success(texts["res_low"])
            elif risk_score < 0.60:
                st.warning(texts["res_mod"])
            else:
                st.error(texts["res_high"])

            st.caption(texts["res_caption"])

        else:
            st.info(texts["info_fill"])

# ------------------------------------------------------------------
# Explanation Tab
# ------------------------------------------------------------------
with tab_explain:
    if "coef_df" not in st.session_state or "risk_score" not in st.session_state:
        st.info(texts["exp_wait"])
    else:
        st.markdown(
            f"<h2 style='margin-bottom:1rem;'>{texts['exp_title']}</h2>",
            unsafe_allow_html=True,
        )

        # PASSING THE LANGUAGE HERE
        explanations = generate_explanations(
            st.session_state["coef_df"][["feature", "contribution"]], lang=selected_lang
        )

        # --- Card container ---
        st.markdown(
            """
            <style>
            .driver-card {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
                padding: 16px 18px;
                margin-bottom: 14px;
            }
            .driver-title {
                font-weight: 600;
                font-size: 15px;
                margin-bottom: 6px;
            }
            .badge-risk {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 999px;
                font-size: 12px;
                margin-left: 8px;
            }
            .risk-up {
                background-color: #7f1d1d;
                color: #fecaca;
            }
            .risk-down {
                background-color: #064e3b;
                color: #a7f3d0;
            }
            .driver-text {
                font-size: 14px;
                color: #e5e7eb;
                line-height: 1.5;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for item in explanations:
            badge_class = "risk-up" if item["direction"] == "increases" else "risk-down"
            badge_text = (
                texts["exp_inc"]
                if item["direction"] == "increases"
                else texts["exp_red"]
            )

            st.markdown(
                f"""
                <div class="driver-card">
                    <div class="driver-title">
                        {texts['exp_card_title']}
                        <span class="badge-risk {badge_class}">{badge_text}</span>
                    </div>
                    <div class="driver-text">
                        {item["text"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(texts["exp_caption"])

# ------------------------------------------------------------------
# Model Overview
# ------------------------------------------------------------------
with tab_model:
    st.subheader(texts["mo_title"])

    # --- High-level summary for HR ---
    st.markdown(texts["mo_intro"])

    # --- What model and why ---
    with st.expander(texts["mo_q1"], expanded=True):
        st.markdown(texts["mo_a1"])

    # --- Evaluation setup ---
    with st.expander(texts["mo_q2"], expanded=True):
        st.markdown(texts["mo_a2"])

    # --- Key metrics (HR-friendly) ---
    with st.expander(texts["mo_q3"], expanded=True):
        st.markdown(texts["mo_a3_intro"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(texts["mo_metrics"][0], "0.83")
        with col_b:
            st.metric(texts["mo_metrics"][1], "0.68")
        with col_c:
            st.metric(texts["mo_metrics"][2], "0.39")

        st.markdown(texts["mo_a3_desc"])

    # --- Confusion matrix summary (concrete counts) ---
    with st.expander(texts["mo_q4"], expanded=False):
        st.markdown(texts["mo_a4"])

    # --- Important disclaimers ---
    st.caption(texts["mo_disc"])


# ------------------------------------------------------------------
# About
# ------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"<h2 style='font-size:26px;'>{texts['about_title']}</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(texts["about_text"])

    st.markdown("---")

    st.caption(texts["about_caption"])
