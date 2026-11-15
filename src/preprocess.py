"""Shared splits and train-fitted preprocessing for both prediction tasks."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = (
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "credit_card_count",
    "overdue_count_6m",
)
TARGET_COLUMNS = ("credit_score", "is_overdue")
RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "finance_data.csv"


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_reg_train: pd.Series
    y_reg_test: pd.Series
    y_cls_train: pd.Series
    y_cls_test: pd.Series


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.shape != (10_000, 8) or list(frame) != list(FEATURE_COLUMNS + TARGET_COLUMNS):
        raise ValueError("Expected 10,000 samples, six features, and two targets.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("The mission dataset must contain numeric columns.")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("The source dataset contains missing or nonfinite values.")
    if not frame["credit_score"].between(0, 1000).all():
        raise ValueError("Credit scores must be within 0 to 1000.")
    if set(frame["is_overdue"].unique()) != {0, 1}:
        raise ValueError("The classification target must contain both binary classes.")
    return frame


def split_data(frame: pd.DataFrame) -> DataSplit:
    X_train, X_test, targets_train, targets_test = train_test_split(
        frame.loc[:, list(FEATURE_COLUMNS)],
        frame.loc[:, list(TARGET_COLUMNS)],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["is_overdue"],
    )
    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_reg_train=targets_train["credit_score"],
        y_reg_test=targets_test["credit_score"],
        y_cls_train=targets_train["is_overdue"],
        y_cls_test=targets_test["is_overdue"],
    )


def make_preprocessor(
    numeric_columns: Sequence[str] = FEATURE_COLUMNS,
    categorical_columns: Sequence[str] = (),
) -> ColumnTransformer:
    """Return an unfitted transformer suitable for use inside a CV pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_columns)),
            ("categorical", categorical, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_pipeline(path: str | Path = DATA_PATH) -> dict:
    frame = load_data(path)
    split = split_data(frame)
    repeated = split_data(frame)
    assert split.X_train.shape == (8000, 6), "Unexpected training dimensions."
    assert split.X_test.shape == (2000, 6), "Unexpected test dimensions."
    assert split.X_train.index.intersection(split.X_test.index).empty, "Split overlap."
    assert split.X_train.index.union(split.X_test.index).equals(frame.index), "Split coverage."
    assert split.X_train.index.equals(repeated.X_train.index), "Unstable training split."
    assert split.X_test.index.equals(repeated.X_test.index), "Unstable test split."
    assert split.X_train.index.equals(split.y_reg_train.index), "Regression train alignment."
    assert split.X_test.index.equals(split.y_reg_test.index), "Regression test alignment."
    assert split.X_train.index.equals(split.y_cls_train.index), "Classification train alignment."
    assert split.X_test.index.equals(split.y_cls_test.index), "Classification test alignment."
    assert not set(TARGET_COLUMNS).intersection(split.X_train.columns), "Target leakage."
    overall_rate = float(frame["is_overdue"].mean())
    assert 0 < overall_rate < 0.5, "Expected minority overdue class."
    assert abs(split.y_cls_train.mean() - overall_rate) <= 1 / len(split.X_train), "Train stratification."
    assert abs(split.y_cls_test.mean() - overall_rate) <= 1 / len(split.X_test), "Test stratification."

    preprocessor = make_preprocessor()
    train_values = preprocessor.fit_transform(split.X_train)
    numeric = preprocessor.named_transformers_["numeric"]
    train_medians = split.X_train.median().to_numpy()
    expected_means = split.X_train.fillna(split.X_train.median()).mean().to_numpy()
    assert np.allclose(numeric["imputer"].statistics_, train_medians), "Train median fit."
    assert np.allclose(numeric["scaler"].mean_, expected_means), "Train scaler fit."
    means_before_test = numeric["scaler"].mean_.copy()
    test_values = preprocessor.transform(split.X_test)
    assert train_values.shape == (8000, 6), "Unexpected transformed train dimensions."
    assert test_values.shape == (2000, 6), "Unexpected transformed test dimensions."
    assert np.isfinite(train_values).all() and np.isfinite(test_values).all(), "Nonfinite transform."
    assert np.array_equal(numeric["scaler"].mean_, means_before_test), "Test refitted scaler."
    assert np.array_equal(numeric["imputer"].statistics_, train_medians), "Test refitted imputer."
    assert list(preprocessor.get_feature_names_out()) == list(FEATURE_COLUMNS), "Feature mapping."

    # Exercise missing values and an unseen category without modifying the dataset.
    fixture_train = pd.DataFrame(
        {
            "age": [20.0, np.nan, 40.0],
            "annual_income": [1500.0, 3000.0, np.nan],
            "employment": ["salaried", np.nan, "self_employed"],
        }
    )
    fixture_test = pd.DataFrame(
        {"age": [100.0], "annual_income": [np.nan], "employment": ["unseen"]}
    )
    fixture_preprocessor = make_preprocessor(("age", "annual_income"), ("employment",))
    fixture_train_values = fixture_preprocessor.fit_transform(fixture_train)
    fixture_numeric = fixture_preprocessor.named_transformers_["numeric"]
    fixture_encoder = fixture_preprocessor.named_transformers_["categorical"]["encoder"]
    fixture_means = fixture_numeric["scaler"].mean_.copy()
    fixture_categories = fixture_encoder.categories_[0].copy()
    fixture_test_values = fixture_preprocessor.transform(fixture_test)
    assert np.isfinite(fixture_train_values).all(), "Missing-value train transform."
    assert np.isfinite(fixture_test_values).all(), "Missing-value test transform."
    assert np.array_equal(fixture_numeric["imputer"].statistics_, [30.0, 2250.0]), "Fixture imputation."
    assert np.array_equal(fixture_numeric["scaler"].mean_, fixture_means), "Fixture test refit."
    assert np.array_equal(fixture_encoder.categories_[0], fixture_categories), "Category test refit."
    assert (fixture_test_values[:, 2:] == 0).all(), "Unknown category encoding."

    return {
        "samples": len(frame),
        "features": len(FEATURE_COLUMNS),
        "train_samples": len(split.X_train),
        "test_samples": len(split.X_test),
        "overdue_rate": overall_rate,
        "train_overdue_rate": float(split.y_cls_train.mean()),
        "test_overdue_rate": float(split.y_cls_test.mean()),
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_pipeline(), ensure_ascii=False))


if __name__ == "__main__":
    main()
