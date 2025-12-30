import pandas as pd

# ------------------------------------------------------------------
# Translation Maps for Feature Names (Raw Feature -> Readable)
# ------------------------------------------------------------------
FEATURE_NAME_MAP = {
    "English": {
        "num__Age": "Age",
        "num__DistanceFromHome": "Commute Distance",
        "num__MonthlyIncome": "Monthly Income",
        "num__TotalWorkingYears": "Total Working Years",
        "num__YearsAtCompany": "Years at Company",
        "num__YearsInCurrentRole": "Years in Current Role",
        "num__YearsSinceLastPromotion": "Years Since Last Promotion",
        "num__YearsWithCurrManager": "Years With Current Manager",
        "num__NumCompaniesWorked": "Number of Companies Worked",
        "num__PercentSalaryHike": "Percent Salary Hike",
        "num__TrainingTimesLastYear": "Training Times Last Year",
        "num__JobInvolvement": "Job Involvement",
        "num__PerformanceRating": "Performance Rating",
        "cat__OverTime_Yes": "Working Overtime",
        "cat__OverTime_No": "No Overtime",
        "cat__BusinessTravel_Travel_Frequently": "Frequent Business Travel",
        "cat__BusinessTravel_Non-Travel": "No Business Travel",
        "cat__MaritalStatus_Single": "Being Single",
    },
    "Türkçe": {
        "num__Age": "Yaş",
        "num__DistanceFromHome": "Ev-İş Mesafesi",
        "num__MonthlyIncome": "Aylık Gelir",
        "num__TotalWorkingYears": "Toplam Çalışma Yılı",
        "num__YearsAtCompany": "Şirketteki Yıl Sayısı",
        "num__YearsInCurrentRole": "Mevcut Pozisyondaki Yıl",
        "num__YearsSinceLastPromotion": "Son Terfiden Beri Geçen Yıl",
        "num__YearsWithCurrManager": "Mevcut Yöneticiyle Çalışma Yılı",
        "num__NumCompaniesWorked": "Çalışılan Şirket Sayısı",
        "num__PercentSalaryHike": "Maaş Artış Yüzdesi",
        "num__TrainingTimesLastYear": "Geçen Yıl Eğitim Sayısı",
        "num__JobInvolvement": "İşe Katılım Düzeyi",
        "num__PerformanceRating": "Performans Puanı",
        "cat__OverTime_Yes": "Fazla Mesai Yapmak",
        "cat__OverTime_No": "Fazla Mesai Yapmamak",
        "cat__BusinessTravel_Travel_Frequently": "Sık İş Seyahati",
        "cat__BusinessTravel_Non-Travel": "Seyahat Olmaması",
        "cat__MaritalStatus_Single": "Bekar Olmak",
    },
}

# ------------------------------------------------------------------
# Suffix Templates (Sentence endings)
# ------------------------------------------------------------------
TEMPLATES = {
    "English": {
        "increase": "increases the attrition risk for this employee.",
        "reduce": "reduces the attrition risk for this employee.",
    },
    "Türkçe": {
        "increase": "bu çalışan için ayrılma riskini artırır.",
        "reduce": "bu çalışan için ayrılma riskini azaltır.",
    },
}


def generate_explanations(df, top_n=5, lang="English"):
    """
    Generate human-readable explanations from feature contributions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['feature', 'contribution']
    top_n : int
        Number of strongest drivers to return
    lang : str
        Language for the explanation ("English" or "Türkçe")

    Returns
    -------
    List[dict]
        Each dict contains:
        - text
        - direction: 'increases' | 'reduces'
    """

    explanations = []

    # Ensure supported language, fallback to English
    if lang not in ["English", "Türkçe"]:
        lang = "English"

    # Sort by absolute contribution magnitude
    df_sorted = (
        df.assign(abs_contribution=df["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
    )

    for _, row in df_sorted.iterrows():
        feature_raw = row["feature"]
        contribution = row["contribution"]

        # 1. Determine Direction
        if contribution > 0:
            direction_key = "increases"  # internal key
            suffix = TEMPLATES[lang]["increase"]
        else:
            direction_key = "reduces"  # internal key
            suffix = TEMPLATES[lang]["reduce"]

        # 2. Determine Feature Name
        # Look up in our map first
        if feature_raw in FEATURE_NAME_MAP[lang]:
            clean_feature = FEATURE_NAME_MAP[lang][feature_raw]
        else:
            # Fallback cleanup logic if not in map
            clean_feature = (
                feature_raw.replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
                .capitalize()
            )

        # 3. Construct Sentence
        text = f"{clean_feature} {suffix}"

        explanations.append(
            {
                "feature": feature_raw,
                "direction": direction_key,
                "text": text,
            }
        )

    return explanations
