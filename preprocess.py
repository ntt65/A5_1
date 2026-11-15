"""Expose the preprocessing API and CLI specified in plans/A5_1.md."""

from src.preprocess import (
    DATA_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMNS,
    DataSplit,
    TrainingMedianImputer,
    load_data,
    main,
    make_preprocessor,
    split_data,
    validate_pipeline,
)

__all__ = [
    "DATA_PATH",
    "FEATURE_COLUMNS",
    "RANDOM_STATE",
    "TARGET_COLUMNS",
    "DataSplit",
    "TrainingMedianImputer",
    "load_data",
    "make_preprocessor",
    "split_data",
    "validate_pipeline",
]


if __name__ == "__main__":
    main()
