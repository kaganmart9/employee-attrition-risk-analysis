import pandas as pd

# Human-readable mappings
FEATURE_TEXT_MAP = {
    "num__is_single_flag": "Being single increases attrition risk.",
    "num__overtime_flag": "Working overtime increases attrition risk.",
    "num__engagement_index": "High employee engagement reduces attrition risk.",
    "num__travel_intensity": "Frequent business travel increases attrition risk.",
    "num__low_income_flag": "Lower income levels increase attrition risk.",
    "num__early_tenure_flag": "Employees early in their tenure are more likely to leave.",
}


def generate_explanations(
    contribution_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Generate instance-level explanations based on feature contributions.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        DataFrame with columns: ['feature', 'contribution']
    top_n : int
        Number of drivers to show.

    Returns
    -------
    explanations : list of dict
        Human-readable explanations.
    """

    df = contribution_df.copy()

    # Sort by absolute contribution (instance-level impact)
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).head(top_n)

    explanations = []

    for _, row in df.iterrows():
        feature = row["feature"]
        contrib = row["contribution"]

        base_text = FEATURE_TEXT_MAP.get(
            feature,
            feature.replace("num__", "")
            .replace("cat__", "")
            .replace("_", " ")
            .capitalize(),
        )

        if contrib > 0:
            explanations.append(
                {
                    "feature": feature,
                    "direction": "increase",
                    "text": f"{base_text} This factor increases the attrition risk for this employee.",
                }
            )
        else:
            explanations.append(
                {
                    "feature": feature,
                    "direction": "decrease",
                    "text": f"{base_text} This factor reduces the attrition risk for this employee.",
                }
            )

    return explanations
