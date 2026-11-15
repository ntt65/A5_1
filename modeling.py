"""Train, evaluate, and validate the models specified in plans/A5_1.md."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from baseline import RULE_COUNT, SCORE_DESCRIPTION, predict_rules
from preprocess import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMNS,
    DataSplit,
    load_data,
    make_preprocessor,
    split_data,
)

ROOT = Path(__file__).resolve().parent
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLD = 0.5
RF_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8],
    "model__min_samples_leaf": [1, 3, 5],
}
PLOT_PATHS = {
    "ridge": "artifacts/ridge_coefficients.png",
    "lasso": "artifacts/lasso_coefficients.png",
    "confusion_matrix": "artifacts/confusion_matrix.png",
    "roc_curves": "artifacts/roc_curves.png",
    "feature_importance": "artifacts/feature_importance.png",
}


class CreditScoreRegressor(RegressorMixin, BaseEstimator):
    """Use the same bounded predictions for CV, evaluation, and callers."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return np.clip(self.estimator_.predict(X), 0.0, 1000.0)


def make_model(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


def predict_credit_score(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return credit scores in [0, 1000] using the fitted regression pipeline."""
    return np.asarray(model.predict(features.loc[:, list(FEATURE_COLUMNS)]), dtype=float)


def predict_overdue_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the probability of the positive overdue class."""
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features.loc[:, list(FEATURE_COLUMNS)])
    return np.asarray(probabilities[:, positive_column], dtype=float)


def summarize_cv(search: GridSearchCV) -> dict:
    results = search.cv_results_
    candidates = []
    for index, parameters in enumerate(results["params"]):
        candidates.append(
            {
                "parameters": parameters,
                "mean_score": float(results["mean_test_score"][index]),
                "std_score": float(results["std_test_score"][index]),
                "fold_scores": [
                    float(results[f"split{fold}_test_score"][index])
                    for fold in range(search.n_splits_)
                ],
                "rank": int(results["rank_test_score"][index]),
            }
        )
    return {
        "splitter": type(search.cv).__name__,
        "n_splits": search.n_splits_,
        "shuffle": search.cv.shuffle,
        "random_state": search.cv.random_state,
        "scoring": search.scoring,
        "verbose": search.verbose,
        "candidate_count": len(candidates),
        "training_samples": int(
            search.best_estimator_["preprocess"].named_transformers_["numeric"]["scaler"].n_samples_seen_
        ),
        "preprocessing_inside_pipeline": isinstance(search.estimator, Pipeline),
        "best_parameters": search.best_params_,
        "best_score": float(search.best_score_),
        "candidates": candidates,
    }


def fit_regression(estimator: BaseEstimator, split: DataSplit) -> tuple:
    search = GridSearchCV(
        make_model(CreditScoreRegressor(estimator)),
        {"model__estimator__alpha": ALPHAS},
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        verbose=0,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(split.X_train, split.y_reg_train)
    coefficients = []
    for alpha in ALPHAS:
        model = search.best_estimator_
        if alpha != search.best_params_["model__estimator__alpha"]:
            model = clone(search.estimator).set_params(model__estimator__alpha=alpha)
            model.fit(split.X_train, split.y_reg_train)
        coefficients.append(model["model"].estimator_.coef_.copy())
    return search, np.asarray(coefficients)


def evaluate_classification(target: pd.Series, scores: np.ndarray) -> tuple:
    predictions = (scores >= THRESHOLD).astype(int)
    record = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "auc": float(roc_auc_score(target, scores)),
        "threshold": THRESHOLD,
        "zero_division": 0,
    }
    fpr, tpr, _ = roc_curve(target, scores, pos_label=1)
    record["roc_auc"] = float(auc(fpr, tpr))
    record["roc_point_count"] = len(fpr)
    return record, (fpr, tpr), predictions


def save_figure(figure, relative_path: str) -> None:
    figure.savefig(ROOT / relative_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_coefficient_plot(name: str, coefficients: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, feature in enumerate(FEATURE_COLUMNS):
        axis.plot(ALPHAS, coefficients[:, index], marker="o", label=feature)
    axis.set_xscale("log")
    axis.set_xlabel("Alpha")
    axis.set_ylabel("Coefficient (standardized features)")
    axis.set_title(f"{name.title()} coefficients fitted on Train")
    axis.axhline(0, color="gray", linewidth=0.7)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, PLOT_PATHS[name])


def save_classification_plots(metrics: dict, curves: dict, matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Normal (0)", "Overdue (1)"]).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d"
    )
    axis.set_title("Random Forest confusion matrix (threshold 0.5)")
    figure.tight_layout()
    save_figure(figure, PLOT_PATHS["confusion_matrix"])

    figure, axis = plt.subplots(figsize=(7, 6))
    labels = {
        "rule_based": "Rule-based (binary score)",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
    for name, (fpr, tpr) in curves.items():
        axis.plot(fpr, tpr, label=f"{labels[name]} (AUC={metrics['classification'][name]['auc']:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, PLOT_PATHS["roc_curves"])

    importance = sorted(metrics["feature_importance"], key=lambda entry: entry["importance"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh([entry["feature"] for entry in importance], [entry["importance"] for entry in importance])
    axis.set(xlabel="Mean decrease in impurity", title="Random Forest feature importance")
    figure.tight_layout()
    save_figure(figure, PLOT_PATHS["feature_importance"])


def validate_preprocessor(model: Pipeline, training: pd.DataFrame, test: pd.DataFrame) -> None:
    preprocessor = model["preprocess"]
    assert list(preprocessor.feature_names_in_) == list(FEATURE_COLUMNS), "Fitted feature schema."
    assert list(preprocessor.get_feature_names_out()) == list(FEATURE_COLUMNS), "Feature name mapping."
    numeric = preprocessor.named_transformers_["numeric"]
    medians = training.median()
    np.testing.assert_allclose(numeric["imputer"].statistics_, medians.to_numpy())
    np.testing.assert_allclose(numeric["scaler"].mean_, training.fillna(medians).mean().to_numpy())
    assert int(numeric["scaler"].n_samples_seen_) == len(training), "Non-Train preprocessing fit."
    means_before = numeric["scaler"].mean_.copy()
    transformed = preprocessor.transform(test)
    assert transformed.shape == (len(test), 6), "Transformed feature dimensions."
    assert np.isfinite(transformed).all(), "Nonfinite transformed Test."
    np.testing.assert_array_equal(numeric["scaler"].mean_, means_before)


def validate_search(search: GridSearchCV, split: DataSplit, regression: bool) -> dict:
    assert isinstance(search.estimator, Pipeline), "CV must receive the full Pipeline."
    assert not hasattr(search.estimator["preprocess"], "transformers_"), "CV template was prefitted."
    assert search.verbose == 0 and search.error_score == "raise", "CV execution settings."
    assert search.n_splits_ == 5 and search.cv.n_splits == 5, "CV fold count."
    assert search.cv.shuffle and search.cv.random_state == RANDOM_STATE, "CV reproducibility."
    candidates = list(ParameterGrid(search.param_grid))
    assert len(candidates) <= 50 and all(len(values) <= 5 for values in search.param_grid.values())
    assert search.cv_results_["params"] == candidates, "CV candidate coverage."
    if regression:
        assert type(search.cv) is KFold and search.scoring == "neg_root_mean_squared_error"
        assert search.param_grid == {"model__estimator__alpha": ALPHAS}
    else:
        assert type(search.cv) is StratifiedKFold and search.scoring == "f1"
        assert search.param_grid == RF_GRID and len(candidates) == 12
    target = split.y_reg_train if regression else split.y_cls_train
    held_out_indices = []
    fold_sizes = []
    for training_rows, validation_rows in search.cv.split(split.X_train, target):
        assert not np.intersect1d(training_rows, validation_rows).size, "CV fold overlap."
        fold_training = split.X_train.iloc[training_rows]
        fold_validation = split.X_train.iloc[validation_rows]
        assert fold_training.index.intersection(split.X_test.index).empty, "CV training Test leakage."
        assert fold_validation.index.intersection(split.X_test.index).empty, "CV validation Test leakage."
        if not regression:
            assert set(target.iloc[training_rows].unique()) == {0, 1}
            assert set(target.iloc[validation_rows].unique()) == {0, 1}
        held_out_indices.extend(validation_rows.tolist())
        fold_sizes.append({"train": len(training_rows), "validation": len(validation_rows)})
    assert sorted(held_out_indices) == list(range(len(split.X_train))), "CV fold coverage."
    assert np.isfinite(search.cv_results_["mean_test_score"]).all(), "Nonfinite CV means."
    assert np.isfinite(search.cv_results_["std_test_score"]).all(), "Nonfinite CV deviations."
    for fold in range(5):
        assert np.isfinite(search.cv_results_[f"split{fold}_test_score"]).all(), "Nonfinite fold score."
    return {"passed": True, "candidate_count": len(candidates), "fold_sizes": fold_sizes}


def validate_results(metrics: dict, split: DataSplit, models: dict, searches: dict,
                     predictions: dict, curves: dict, matrix: np.ndarray, coefficients: dict) -> dict:
    assert split.X_train.shape == (8000, 6) and split.X_test.shape == (2000, 6), "Split dimensions."
    assert list(split.X_train) == list(FEATURE_COLUMNS) == list(split.X_test), "Input feature schema."
    assert not set(TARGET_COLUMNS).intersection(split.X_train), "Target feature leakage."
    assert split.X_train.index.intersection(split.X_test.index).empty, "Train/Test overlap."
    assert len(split.X_train.index.union(split.X_test.index)) == 10_000, "Split coverage."
    for features, targets in (
        (split.X_train, split.y_reg_train), (split.X_train, split.y_cls_train),
        (split.X_test, split.y_reg_test), (split.X_test, split.y_cls_test),
    ):
        assert features.index.equals(targets.index), "Target alignment."
    repeated = split_data(load_data())
    assert repeated.X_train.index.equals(split.X_train.index), "Train split reproducibility."
    assert repeated.X_test.index.equals(split.X_test.index), "Test split reproducibility."
    assert RULE_COUNT >= 5, "Insufficient baseline rules."
    assert models["logistic_regression"]["model"].class_weight == "balanced"
    assert models["random_forest"]["model"].class_weight == "balanced"
    for model in models.values():
        validate_preprocessor(model, split.X_train, split.X_test)
    cv_checks = {
        name: validate_search(search, split, name in ("ridge", "lasso"))
        for name, search in searches.items()
    }
    ranges = {}
    for name, values in predictions.items():
        upper = 1000.0 if name in ("ridge", "lasso") else 1.0
        assert values.shape == (2000,), "Prediction length."
        assert np.isfinite(values).all(), "Nonfinite prediction."
        assert ((values >= 0) & (values <= upper)).all(), "Prediction range."
        ranges[name] = {"count": len(values), "finite": True, "min": float(values.min()),
                        "max": float(values.max()), "allowed_range": [0.0, upper], "passed": True}
    assert set(np.unique(predictions["rule_based"])) <= {0, 1}, "Nonbinary rule scores."
    for name in ("ridge", "lasso"):
        record = metrics["regression"][name]
        values = predictions[name]
        expected = [root_mean_squared_error(split.y_reg_test, values),
                    mean_absolute_error(split.y_reg_test, values), r2_score(split.y_reg_test, values)]
        np.testing.assert_allclose([record[field] for field in ("rmse", "mae", "r2")], expected)
        assert np.isfinite([record["alpha"], *expected]).all(), "Nonfinite regression metrics."
        assert record["alpha"] == searches[name].best_params_["model__estimator__alpha"]
        assert coefficients[name].shape == (5, 6) and np.isfinite(coefficients[name]).all()
        wrapper = models[name]["model"]
        np.testing.assert_array_equal(wrapper.predict(np.zeros((1, 6))),
                                      np.clip(wrapper.estimator_.predict(np.zeros((1, 6))), 0, 1000))
    for name, (fpr, tpr) in curves.items():
        record = metrics["classification"][name]
        expected, _, _ = evaluate_classification(split.y_cls_test, predictions[name])
        for field in ("accuracy", "precision", "recall", "f1", "auc"):
            assert np.isfinite(record[field]) and 0 <= record[field] <= 1
            assert np.isclose(record[field], expected[field]), "Classification metric mismatch."
        assert np.isfinite(fpr).all() and np.isfinite(tpr).all()
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0), "Invalid ROC monotonicity."
        assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1, "ROC endpoints."
        assert np.isclose(auc(fpr, tpr), record["auc"]), "ROC/AUC mismatch."
    assert matrix.shape == (2, 2) and int(matrix.sum()) == 2000, "Confusion matrix dimensions/count."
    np.testing.assert_array_equal(matrix, confusion_matrix(
        split.y_cls_test, predictions["random_forest"] >= THRESHOLD, labels=[0, 1]))
    importance = metrics["feature_importance"]
    assert [entry["feature"] for entry in importance] == list(FEATURE_COLUMNS)
    weights = np.asarray([entry["importance"] for entry in importance])
    assert weights.shape == (6,) and np.isfinite(weights).all() and (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0), "Feature importance normalization."
    np.testing.assert_array_equal(weights, models["random_forest"]["model"].feature_importances_)
    png_checks = {}
    for relative_path in metrics["plots"]:
        path = ROOT / relative_path
        assert path.is_file() and path.stat().st_size > 0, "Missing/empty PNG."
        with path.open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature."
        image = mpimg.imread(path)
        assert image.ndim == 3 and min(image.shape[:2]) > 0 and np.isfinite(image).all()
        png_checks[relative_path] = {"passed": True, "bytes": path.stat().st_size}
    return {"status": "passed", "features": 6, "shared_test_samples": 2000,
            "train_only_preprocessing": True, "cv_pipeline_and_folds": cv_checks,
            "prediction_ranges": ranges, "metrics_recomputed": True, "roc_auc_matches": True,
            "confusion_matrix_count": int(matrix.sum()), "png_files": png_checks,
            "feature_importance": "passed"}


def validate_json_contract(metrics: dict) -> None:
    assert {"regression", "classification", "data", "plots", "feature_importance", "validation"} <= metrics.keys()
    assert set(metrics["regression"]) == {"ridge", "lasso"}
    assert set(metrics["classification"]) == {"rule_based", "logistic_regression", "random_forest"}
    assert metrics["plots"] == list(PLOT_PATHS.values()) and len(metrics["plots"]) == 5
    for name in ("ridge", "lasso"):
        record = metrics["regression"][name]
        assert {"alpha", "rmse", "mae", "r2", "cv", "prediction_method"} <= record.keys()
        assert all(np.isfinite(record[field]) for field in ("alpha", "rmse", "mae", "r2"))
        assert record["cv"]["candidate_count"] == 5
    for record in metrics["classification"].values():
        assert all(np.isfinite(record[field]) for field in ("accuracy", "precision", "recall", "f1", "auc"))
        assert record["threshold"] == THRESHOLD and record["zero_division"] == 0
    forest = metrics["classification"]["random_forest"]
    assert forest["cv"]["candidate_count"] == 12 and forest["cv"]["scoring"] == "f1"
    assert forest["best_parameters"] == forest["cv"]["best_parameters"]
    assert forest["cv_score"] == forest["cv"]["best_score"]
    assert np.asarray(forest["confusion_matrix"]).shape == (2, 2)
    assert sum(map(sum, forest["confusion_matrix"])) == 2000
    assert metrics["classification"]["rule_based"]["score_method"] == SCORE_DESCRIPTION
    assert metrics["data"]["features"] == list(FEATURE_COLUMNS)
    assert metrics["data"]["train_samples"] == 8000 and metrics["data"]["test_samples"] == 2000
    assert metrics["data"]["random_state"] == RANDOM_STATE
    assert metrics["validation"]["status"] == "passed"
    json.dumps(metrics, allow_nan=False)


def run_modeling() -> dict:
    split = split_data(load_data())
    (ROOT / "artifacts").mkdir(exist_ok=True)
    searches, models, coefficients = {}, {}, {}
    metrics = {
        "regression": {}, "classification": {},
        "data": {"samples": 10000, "train_samples": len(split.X_train), "test_samples": len(split.X_test),
                 "random_state": RANDOM_STATE, "stratify": "is_overdue", "features": list(FEATURE_COLUMNS),
                 "excluded_targets": list(TARGET_COLUMNS), "train_overdue_rate": float(split.y_cls_train.mean()),
                 "test_overdue_rate": float(split.y_cls_test.mean())},
        "plots": list(PLOT_PATHS.values()),
    }
    predictions = {}
    for name, estimator in (("ridge", Ridge(random_state=RANDOM_STATE)),
                            ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=10000))):
        searches[name], coefficients[name] = fit_regression(estimator, split)
        models[name] = searches[name].best_estimator_
        predictions[name] = predict_credit_score(models[name], split.X_test)
        metrics["regression"][name] = {
            "alpha": float(searches[name].best_params_["model__estimator__alpha"]),
            "rmse": float(root_mean_squared_error(split.y_reg_test, predictions[name])),
            "mae": float(mean_absolute_error(split.y_reg_test, predictions[name])),
            "r2": float(r2_score(split.y_reg_test, predictions[name])),
            "prediction_method": "actual predict_credit_score return values clipped to [0, 1000]",
            "cv": summarize_cv(searches[name]),
            "coefficient_path": {"alphas": ALPHAS, "features": list(FEATURE_COLUMNS),
                                 "coefficients": coefficients[name].tolist(), "fit_samples": 8000},
        }
        save_coefficient_plot(name, coefficients[name])

    models["logistic_regression"] = make_model(LogisticRegression(
        class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000))
    models["logistic_regression"].fit(split.X_train, split.y_cls_train)
    searches["random_forest"] = GridSearchCV(
        make_model(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        RF_GRID, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1", verbose=0, n_jobs=1, error_score="raise",
    )
    searches["random_forest"].fit(split.X_train, split.y_cls_train)
    models["random_forest"] = searches["random_forest"].best_estimator_
    predictions["rule_based"] = predict_rules(split.X_test)
    curves = {}
    labels = {}
    for name in ("rule_based", "logistic_regression", "random_forest"):
        if name != "rule_based":
            predictions[name] = predict_overdue_probability(models[name], split.X_test)
        record, curves[name], labels[name] = evaluate_classification(split.y_cls_test, predictions[name])
        record["score_method"] = SCORE_DESCRIPTION if name == "rule_based" else "positive class 1 predict_proba"
        metrics["classification"][name] = record
    metrics["classification"]["rule_based"]["rule_count"] = RULE_COUNT
    metrics["classification"]["logistic_regression"]["class_weight"] = "balanced"
    forest = metrics["classification"]["random_forest"]
    forest.update({"class_weight": "balanced", "best_parameters": searches["random_forest"].best_params_,
                   "cv_score": float(searches["random_forest"].best_score_),
                   "cv": summarize_cv(searches["random_forest"])})
    matrix = confusion_matrix(split.y_cls_test, labels["random_forest"], labels=[0, 1])
    forest["confusion_matrix"] = matrix.tolist()
    forest["confusion_matrix_labels"] = [0, 1]
    metrics["feature_importance"] = [
        {"feature": name, "importance": float(weight)}
        for name, weight in zip(models["random_forest"]["preprocess"].get_feature_names_out(),
                                models["random_forest"]["model"].feature_importances_)
    ]
    save_classification_plots(metrics, curves, matrix)
    metrics["validation"] = validate_results(metrics, split, models, searches, predictions, curves, matrix, coefficients)
    validate_json_contract(metrics)
    metrics["validation"]["json_contract"] = "passed"
    output = ROOT / "artifacts/metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    validate_json_contract(saved)
    assert saved == metrics, "Saved JSON differs from evaluated results."
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.parse_args()
    metrics = run_modeling()
    print(json.dumps({
        "classification": {name: {field: record[field] for field in ("accuracy", "precision", "recall", "f1", "auc")}
                           for name, record in metrics["classification"].items()},
        "regression": {name: {field: record[field] for field in ("alpha", "rmse", "mae", "r2")}
                       for name, record in metrics["regression"].items()},
        "random_forest_best_parameters": metrics["classification"]["random_forest"]["best_parameters"],
        "random_forest_cv_f1": metrics["classification"]["random_forest"]["cv_score"],
        "validation": metrics["validation"]["status"], "plots": metrics["plots"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
