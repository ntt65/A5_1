"""Fixed business rules for predicting overdue status from raw features."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from preprocess import FEATURE_COLUMNS


RULE_COUNT = 5
SCORE_DESCRIPTION = "binary 0/1 rule predictions used as AUC scores"
RULE_DESCRIPTIONS = (
    "overdue_count_6m >= 3",
    "debt_ratio > 0.7",
    "annual_income < 3000",
    "credit_card_count > 7",
    "overdue_count_6m >= 1 and debt_ratio > 0.5",
)


def first_matching_rule(row: Mapping[str, float]) -> int:
    """Return the first matching rule number, or zero for the normal default."""
    if row["overdue_count_6m"] >= 3:
        return 1
    elif row["debt_ratio"] > 0.7:
        return 2
    elif row["annual_income"] < 3000:
        return 3
    elif row["credit_card_count"] > 7:
        return 4
    elif row["overdue_count_6m"] >= 1 and row["debt_ratio"] > 0.5:
        return 5
    return 0


def rule_based_predict(row: Mapping[str, float]) -> int:
    """Return an overdue label using five predefined risk conditions."""
    return int(first_matching_rule(row) > 0)


def predict_rule_numbers(features: pd.DataFrame) -> np.ndarray:
    """Attribute each prediction to one rule in the original if/elif order."""
    inputs = features.loc[:, list(FEATURE_COLUMNS)]
    return np.fromiter(
        (first_matching_rule(row) for row in inputs.to_dict(orient="records")),
        dtype=np.int64,
        count=len(inputs),
    )


def predict_rules(features: pd.DataFrame) -> np.ndarray:
    """Predict labels in row order using only the approved input columns."""
    return (predict_rule_numbers(features) > 0).astype(np.int64)


def summarize_rule_contributions(features: pd.DataFrame, target: pd.Series) -> dict:
    """Count each positive prediction once under its first matching rule."""
    if len(features) != len(target) or not features.index.equals(target.index):
        raise ValueError("Rule contribution targets must align with feature rows.")
    if not target.isin([0, 1]).all():
        raise ValueError("Rule contribution targets must be binary.")
    rule_numbers = predict_rule_numbers(features)
    positive = target.to_numpy() == 1
    contributions = []
    for number, description in enumerate(RULE_DESCRIPTIONS, start=1):
        applied = rule_numbers == number
        contributions.append({
            "rule_number": number,
            "condition": description,
            "applied": int(applied.sum()),
            "tp": int((applied & positive).sum()),
            "fp": int((applied & ~positive).sum()),
        })
    return {
        "attribution": "first matching if/elif condition",
        "rules": contributions,
        "default_normal_count": int((rule_numbers == 0).sum()),
        "positive_predictions": int((rule_numbers > 0).sum()),
        "tp": sum(rule["tp"] for rule in contributions),
        "fp": sum(rule["fp"] for rule in contributions),
    }
