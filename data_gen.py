"""Generate the mission dataset without replacing existing source data."""

from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 10_000
RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent / "finance_data.csv"


def generate_data() -> pd.DataFrame:
    """Preserve the mission's legacy NumPy RNG sequence and formulas."""
    rng = np.random.RandomState(RANDOM_STATE)
    data = {
        "age": rng.randint(20, 70, N_SAMPLES),
        "annual_income": rng.normal(5000, 2000, N_SAMPLES).round(0),
        "spending_score": rng.randint(1, 100, N_SAMPLES),
        "debt_ratio": rng.uniform(0, 1, N_SAMPLES).round(2),
        "credit_card_count": rng.randint(1, 10, N_SAMPLES),
        "overdue_count_6m": rng.poisson(0.5, N_SAMPLES),
    }
    frame = pd.DataFrame(data)
    frame["annual_income"] = frame["annual_income"].apply(
        lambda value: max(value, 1500)
    )
    frame["credit_score"] = (
        300
        + (frame["annual_income"] / 100) * 3
        - frame["overdue_count_6m"] * 50
        - frame["debt_ratio"] * 100
        + rng.normal(0, 30, N_SAMPLES)
    )
    frame["credit_score"] = frame["credit_score"].clip(0, 1000).round(0)
    threshold = frame["credit_score"].quantile(0.15)
    frame["is_overdue"] = np.where(
        (frame["credit_score"] < threshold) & (rng.rand(N_SAMPLES) > 0.2),
        1,
        0,
    )
    return frame.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    expected = generate_data()
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        if existing.shape != expected.shape or list(existing) != list(expected):
            raise ValueError("Existing dataset schema differs from the mission.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in existing.dtypes):
            raise ValueError("Existing dataset contains nonnumeric columns.")
        if not np.allclose(existing.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12):
            raise ValueError("Existing dataset differs from the mission generator.")
        print("Existing finance_data.csv verified and reused.")
    else:
        expected.to_csv(DATA_PATH, index=False, mode="x")
        print("Generated finance_data.csv.")
    print(f"Samples={len(expected)}, overdue_rate={expected['is_overdue'].mean():.4%}")


if __name__ == "__main__":
    main()
