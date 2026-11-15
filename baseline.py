"""Fixed business rules for predicting overdue status from raw features."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from preprocess import FEATURE_COLUMNS


RULE_COUNT = 5
SCORE_DESCRIPTION = "binary 0/1 rule predictions used as AUC scores"


def rule_based_predict(row: Mapping[str, float]) -> int:
    """Return an overdue label using five predefined risk conditions."""
    if row["overdue_count_6m"] >= 3:
        return 1
    elif row["debt_ratio"] > 0.7:
        return 1
    elif row["annual_income"] < 3000:
        return 1
    elif row["credit_card_count"] > 7:
        return 1
    elif row["overdue_count_6m"] >= 1 and row["debt_ratio"] > 0.5:
        return 1
    return 0


def predict_rules(features: pd.DataFrame) -> np.ndarray:
    """Predict labels in row order using only the approved input columns."""
    inputs = features.loc[:, list(FEATURE_COLUMNS)]
    return np.fromiter(
        (rule_based_predict(row) for row in inputs.to_dict(orient="records")),
        dtype=np.int64,
        count=len(inputs),
    )
