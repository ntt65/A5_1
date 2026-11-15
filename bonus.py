"""Validate the custom transformer and plot a Train-only learning curve."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve

from modeling import make_model, save_figure
from preprocess import RANDOM_STATE, load_data, split_data, validate_pipeline

ROOT = Path(__file__).resolve().parent
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
CURVE_PATH = "artifacts/learning_curve.png"
METRICS_PATH = "artifacts/bonus_metrics.json"


def run_bonus() -> dict:
    preprocessing = validate_pipeline()
    split = split_data(load_data())
    estimator = make_model(LogisticRegression(
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000
    ))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(split.X_train, split.y_cls_train))
    expected_sizes = (np.asarray(TRAIN_FRACTIONS) * len(folds[0][0])).astype(int)
    random = np.random.RandomState(RANDOM_STATE)
    held_out_rows = []
    for training_rows, validation_rows in folds:
        assert not np.intersect1d(training_rows, validation_rows).size, "CV fold overlap."
        for rows in (training_rows, validation_rows):
            assert split.X_train.iloc[rows].index.intersection(split.X_test.index).empty, "Test leakage."
        shuffled_rows = random.permutation(training_rows)
        for size in expected_sizes:
            assert set(split.y_cls_train.iloc[shuffled_rows[:size]].unique()) == {0, 1}, "Single-class subset."
        held_out_rows.extend(validation_rows.tolist())
    assert sorted(held_out_rows) == list(range(len(split.X_train))), "CV validation coverage."

    sizes, train_scores, validation_scores = learning_curve(
        estimator, split.X_train, split.y_cls_train,
        train_sizes=TRAIN_FRACTIONS, cv=folds, scoring="f1",
        shuffle=True, random_state=RANDOM_STATE,
        n_jobs=1, verbose=0, error_score="raise",
    )
    np.testing.assert_array_equal(sizes, expected_sizes)
    for scores in (train_scores, validation_scores):
        assert scores.shape == (5, 5), "Unexpected learning-curve dimensions."
        assert np.isfinite(scores).all(), "Nonfinite learning-curve score."
        assert ((scores >= 0) & (scores <= 1)).all(), "F1 outside [0, 1]."
    assert not hasattr(estimator["preprocess"], "transformers_"), "CV mutated the unfitted template."
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    (ROOT / "artifacts").mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sizes, train_mean, "o-", label="Train F1")
    axis.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    axis.plot(sizes, validation_mean, "o-", label="Validation F1")
    axis.fill_between(sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.15)
    axis.set(xlabel="Training samples per fold", ylabel="F1 score", title="Logistic Regression learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, CURVE_PATH)
    image = mpimg.imread(ROOT / CURVE_PATH)
    assert image.ndim == 3 and min(image.shape[:2]) > 0 and np.isfinite(image).all(), "Invalid curve PNG."

    results = {
        "custom_transformer": {
            "class": "TrainingMedianImputer",
            "strategy": "Train medians, zero for entirely missing columns",
            "validation": preprocessing["custom_transformer"],
        },
        "learning_curve": {
            "estimator": "Logistic Regression",
            "class_weight": "balanced",
            "metric": "f1",
            "training_pool_samples": len(split.X_train),
            "test_samples_used": 0,
            "cv": {"splitter": "StratifiedKFold", "folds": 5, "shuffle": True, "random_state": RANDOM_STATE},
            "train_fractions": TRAIN_FRACTIONS,
            "train_sizes": sizes.tolist(),
            "train_scores": train_scores.tolist(),
            "validation_scores": validation_scores.tolist(),
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "validation_mean": validation_mean.tolist(),
            "validation_std": validation_std.tolist(),
            "train_validation_gap": (train_mean - validation_mean).tolist(),
            "plot": CURVE_PATH,
        },
        "validation": {
            "status": "passed", "train_only": True, "folds_disjoint": True,
            "both_classes_in_subsets": True, "finite_scores": True, "png_decoded": True,
        },
    }
    output = ROOT / METRICS_PATH
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == results, "Saved bonus metrics differ from calculated results."
    curve = saved["learning_curve"]
    for prefix in ("train", "validation"):
        scores = np.asarray(curve[f"{prefix}_scores"])
        np.testing.assert_allclose(curve[f"{prefix}_mean"], scores.mean(axis=1))
        np.testing.assert_allclose(curve[f"{prefix}_std"], scores.std(axis=1))
    np.testing.assert_allclose(curve["train_validation_gap"], train_mean - validation_mean)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.parse_args()
    results = run_bonus()
    curve = results["learning_curve"]
    print(json.dumps({
        "custom_transformer": results["custom_transformer"]["validation"],
        "train_sizes": curve["train_sizes"],
        "train_f1": curve["train_mean"], "validation_f1": curve["validation_mean"],
        "validation": results["validation"]["status"], "plot": CURVE_PATH, "metrics": METRICS_PATH,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
