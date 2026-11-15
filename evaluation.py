"""Compare imbalance handling, ensemble stability, latency, and decision costs."""

import argparse
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path
from time import perf_counter
import tracemalloc

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ResamplingPipeline
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from modeling import make_model, predict_overdue_probability
from preprocess import RANDOM_STATE, load_data, make_preprocessor, split_data


ROOT = Path(__file__).resolve().parent
METRICS_PATH = "artifacts/evaluation_metrics.json"
FP_COST = 1
FN_COST = 10
SENSITIVITY_FN_COSTS = (5, 10, 20)
SINGLE_REQUEST_LIMIT_MS = 100.0
SERIALIZED_MODEL_LIMIT_BYTES = 10 * 1024 * 1024
THRESHOLDS = np.append(np.linspace(0.0, 1.0, 101), np.nextafter(1.0, np.inf))


def classification_cost(
    target, probabilities, threshold: float, fp_cost: int = FP_COST, fn_cost: int = FN_COST,
) -> dict:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "auc": float(roc_auc_score(target, probabilities)),
        "cost": int(fp * fp_cost + fn * fn_cost),
    }


def evaluate_folds(templates: dict, split, folds: list) -> tuple[dict, dict]:
    """Fit each pipeline only on fold-training rows and retain held-out scores."""
    fold_results = {}
    oof_probabilities = {}
    for name, template in templates.items():
        records = []
        oof = np.full(len(split.X_train), np.nan)
        coverage = np.zeros(len(split.X_train), dtype=int)
        for training_rows, validation_rows in folds:
            training = split.X_train.iloc[training_rows]
            validation = split.X_train.iloc[validation_rows]
            target_train = split.y_cls_train.iloc[training_rows]
            target_validation = split.y_cls_train.iloc[validation_rows]
            assert training.index.intersection(validation.index).empty
            assert training.index.intersection(split.X_test.index).empty
            assert validation.index.intersection(split.X_test.index).empty
            model = clone(template)
            started = perf_counter()
            model.fit(training, target_train)
            fit_seconds = perf_counter() - started
            numeric = model["preprocess"].named_transformers_["numeric"]
            assert int(numeric["scaler"].n_samples_seen_) == len(training)
            np.testing.assert_allclose(numeric["imputer"].statistics_, training.median())
            train_probabilities = predict_overdue_probability(model, training)
            validation_probabilities = predict_overdue_probability(model, validation)
            assert validation_probabilities.shape == (len(validation),)
            assert np.isfinite(validation_probabilities).all()
            assert ((validation_probabilities >= 0) & (validation_probabilities <= 1)).all()
            oof[validation_rows] = validation_probabilities
            coverage[validation_rows] += 1
            record = classification_cost(target_validation, validation_probabilities, 0.5)
            record.update({
                "train_f1": float(f1_score(target_train, train_probabilities >= 0.5, zero_division=0)),
                "fit_seconds": float(fit_seconds),
                "training_samples": len(training),
                "validation_samples": len(validation),
                "smote_generated_samples": 0,
            })
            if name == "smote":
                sampler = model["smote"]
                minority_count = int(target_train.value_counts().min())
                assert sampler.nn_k_.n_samples_fit_ == minority_count
                record["smote_generated_samples"] = int(sum(sampler.sampling_strategy_.values()))
                assert record["smote_generated_samples"] == int(
                    target_train.value_counts().max() - minority_count
                )
            records.append(record)
        np.testing.assert_array_equal(coverage, np.ones(len(split.X_train), dtype=int))
        assert np.isfinite(oof).all()
        fold_results[name] = records
        oof_probabilities[name] = oof
        print(f"CV completed: {name}, folds={len(records)}", flush=True)
    return fold_results, oof_probabilities


def summarize_folds(records: list) -> dict:
    fields = ("precision", "recall", "f1", "auc")
    validation_mean = {field: float(np.mean([row[field] for row in records])) for field in fields}
    validation_std = {field: float(np.std([row[field] for row in records])) for field in fields}
    train_scores = [row["train_f1"] for row in records]
    return {
        "validation_mean": validation_mean,
        "validation_std": validation_std,
        "train_f1_mean": float(np.mean(train_scores)),
        "train_f1_std": float(np.std(train_scores)),
        "validation_f1_mean": validation_mean["f1"],
        "validation_f1_std": validation_std["f1"],
        "generalization_gap": float(np.mean(train_scores) - validation_mean["f1"]),
        "fit_seconds_mean": float(np.mean([row["fit_seconds"] for row in records])),
        "folds": records,
    }


def benchmark_model(model, training) -> dict:
    """Measure fitted-pipeline inference using Train inputs on the same host."""
    requests = [training.iloc[[index]] for index in range(100)]
    batch = training.iloc[:256]
    for _ in range(5):
        predict_overdue_probability(model, requests[0])
        predict_overdue_probability(model, batch)
    single_seconds = []
    for request in requests:
        started = perf_counter()
        predict_overdue_probability(model, request)
        single_seconds.append(perf_counter() - started)
    batch_seconds = []
    for _ in range(10):
        started = perf_counter()
        predict_overdue_probability(model, batch)
        batch_seconds.append(perf_counter() - started)
    tracemalloc.start()
    for _ in range(3):
        predict_overdue_probability(model, batch)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    serialized_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    p95_ms = float(np.quantile(single_seconds, 0.95) * 1000.0)
    batch_median_seconds = float(np.median(batch_seconds))
    assert p95_ms > 0 and batch_median_seconds > 0 and serialized_bytes > 0
    return {
        "preprocessing_included": True,
        "benchmark_input_source": "Train features only",
        "warmup_iterations": 5,
        "single_request_repetitions": len(requests),
        "single_request_p95_ms": p95_ms,
        "single_request_median_ms": float(np.median(single_seconds) * 1000.0),
        "batch_samples": len(batch),
        "batch_repetitions": len(batch_seconds),
        "batch_median_ms": batch_median_seconds * 1000.0,
        "batch_samples_per_second": float(len(batch) / batch_median_seconds),
        "serialized_model_bytes": serialized_bytes,
        "python_traced_inference_peak_bytes": int(peak_bytes),
        "meets_assumed_limits": bool(
            p95_ms <= SINGLE_REQUEST_LIMIT_MS
            and serialized_bytes <= SERIALIZED_MODEL_LIMIT_BYTES
        ),
    }


def choose_threshold(target, probabilities, fp_cost: int = FP_COST, fn_cost: int = FN_COST) -> dict:
    candidates = [
        classification_cost(target, probabilities, value, fp_cost, fn_cost)
        for value in THRESHOLDS
    ]
    selected = min(candidates, key=lambda row: (row["cost"], row["fn"], row["fp"], row["threshold"]))
    return {
        "selected_threshold": selected["threshold"],
        "oof_selected": selected,
        "oof_default": classification_cost(target, probabilities, 0.5, fp_cost, fn_cost),
        "threshold_candidates": candidates,
    }


def validate_record(
    record: dict, target, probabilities, fp_cost: int = FP_COST, fn_cost: int = FN_COST,
) -> None:
    """Check metric arithmetic independently against the confusion counts."""
    tn, fp, fn, tp = confusion_matrix(
        target, probabilities >= record["threshold"], labels=[0, 1]
    ).ravel()
    assert [record[field] for field in ("tn", "fp", "fn", "tp")] == [tn, fp, fn, tp]
    assert tn + fp + fn + tp == len(target)
    assert record["cost"] == int(fp * fp_cost + fn * fn_cost)
    expected_precision = tp / (tp + fp) if tp + fp else 0.0
    expected_recall = tp / (tp + fn) if tp + fn else 0.0
    expected_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    np.testing.assert_allclose(
        [record["precision"], record["recall"], record["f1"]],
        [expected_precision, expected_recall, expected_f1],
    )
    assert np.isclose(record["auc"], roc_auc_score(target, probabilities))


def run_evaluation() -> dict:
    split = split_data(load_data())
    assert split.X_train.index.intersection(split.X_test.index).empty
    original = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
    forest_parameters = original["classification"]["random_forest"]["best_parameters"]
    forest = make_model(RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
    )).set_params(**forest_parameters)
    logistic = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    templates = {
        "unweighted": make_model(clone(logistic)),
        "balanced": make_model(clone(logistic).set_params(class_weight="balanced")),
        "smote": ResamplingPipeline([
            ("preprocess", make_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", clone(logistic)),
        ]),
        "decision_tree": make_model(DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
            max_depth=forest["model"].max_depth,
            min_samples_leaf=forest["model"].min_samples_leaf,
            max_features=forest["model"].max_features,
        )),
        "random_forest": forest,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(split.X_train, split.y_cls_train))
    fold_results, oof_probabilities = evaluate_folds(templates, split, folds)
    summaries = {name: summarize_folds(records) for name, records in fold_results.items()}
    fitted_models = {
        "logistic_regression": clone(templates["balanced"]).fit(split.X_train, split.y_cls_train),
        "random_forest": clone(forest).fit(split.X_train, split.y_cls_train),
    }
    benchmarks = {name: benchmark_model(model, split.X_train) for name, model in fitted_models.items()}
    source_names = {"logistic_regression": "balanced", "random_forest": "random_forest"}
    sensitivity_scenarios = [
        {
            "fp_cost": FP_COST,
            "fn_cost": fn_cost,
            "models": {
                name: choose_threshold(
                    split.y_cls_train, oof_probabilities[source_names[name]], FP_COST, fn_cost
                )
                for name in fitted_models
            },
        }
        for fn_cost in SENSITIVITY_FN_COSTS
    ]
    threshold_selection = next(
        scenario["models"] for scenario in sensitivity_scenarios if scenario["fn_cost"] == FN_COST
    )
    eligible = [name for name in fitted_models if benchmarks[name]["meets_assumed_limits"]]
    decision = {
        "status": "no_candidate_meets_assumed_limits",
        "model": None, "threshold": None,
        "selection_basis": "minimum Train OOF cost among candidates meeting assumed operating limits",
        "eligible_models": eligible,
        "test_used_for_selection": False,
    }
    if eligible:
        selected_name = min(eligible, key=lambda name: (
            threshold_selection[name]["oof_selected"]["cost"],
            threshold_selection[name]["oof_selected"]["fn"],
            benchmarks[name]["single_request_p95_ms"], name,
        ))
        decision.update({
            "status": "selected_under_assumed_limits",
            "model": selected_name,
            "threshold": threshold_selection[selected_name]["selected_threshold"],
        })

    # Model and threshold selection are complete before reading Test outcomes.
    for name, model in fitted_models.items():
        probabilities = predict_overdue_probability(model, split.X_test)
        assert probabilities.shape == (len(split.X_test),) and np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        for scenario in sensitivity_scenarios:
            selected = scenario["models"][name]
            fp_cost, fn_cost = scenario["fp_cost"], scenario["fn_cost"]
            selected["test_default"] = classification_cost(
                split.y_cls_test, probabilities, 0.5, fp_cost, fn_cost
            )
            selected["test_selected"] = classification_cost(
                split.y_cls_test, probabilities, selected["selected_threshold"], fp_cost, fn_cost
            )
            for field in ("test_default", "test_selected"):
                validate_record(selected[field], split.y_cls_test, probabilities, fp_cost, fn_cost)
            oof = oof_probabilities[source_names[name]]
            for field in ("oof_default", "oof_selected"):
                validate_record(selected[field], split.y_cls_train, oof, fp_cost, fn_cost)
            candidates = selected["threshold_candidates"]
            np.testing.assert_array_equal([row["threshold"] for row in candidates], THRESHOLDS)
            for record in candidates:
                validate_record(record, split.y_cls_train, oof, fp_cost, fn_cost)
            assert selected["oof_selected"] == min(
                candidates,
                key=lambda row: (row["cost"], row["fn"], row["fp"], row["threshold"]),
            )
            assert selected["selected_threshold"] == selected["oof_selected"]["threshold"]
            assert selected["oof_default"]["threshold"] == selected["test_default"]["threshold"] == 0.5
        selected = threshold_selection[name]
        for metric in ("precision", "recall", "f1", "auc"):
            assert np.isclose(selected["test_default"][metric], original["classification"][name][metric])

    results = {
        "data": {"train_samples": len(split.X_train), "test_samples": len(split.X_test)},
        "cv": {
            "splitter": "StratifiedKFold", "folds": len(folds),
            "shuffle": True, "random_state": RANDOM_STATE, "test_samples_used": 0,
            "preprocessing_and_resampling_inside_training_fold": True,
            "validation_predictions_per_train_sample": 1,
            "forest_parameters": forest_parameters,
            "forest_parameter_source": "previous Train-only model-selection CV",
            "interpretation": "conditional comparison at fixed parameters; OOF threshold scores are tuning scores",
        },
        "imbalance_comparison": {name: summaries[name] for name in ("unweighted", "balanced", "smote")},
        "ensemble_comparison": {name: summaries[name] for name in ("decision_tree", "random_forest")},
        "operating_assumptions": {
            "purpose": "illustrative loan prescreening with missed-overdue cost prioritized",
            "cost_units": "hypothetical relative units", "fp_cost": FP_COST, "fn_cost": FN_COST,
            "single_request_p95_limit_ms": SINGLE_REQUEST_LIMIT_MS,
            "serialized_model_limit_bytes": SERIALIZED_MODEL_LIMIT_BYTES,
            "memory_measurement": "pickle size and Python-tracked inference allocations; native memory and process RSS require separate measurement",
            "host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
            "versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "imbalanced-learn", "numpy")},
        },
        "benchmarks": benchmarks,
        "threshold_selection": threshold_selection,
        "cost_sensitivity": {
            "selection_basis": "minimum Train OOF cost on the same probabilities and threshold candidates",
            "cost_units": "hypothetical relative units",
            "tie_break": ["cost", "fn", "fp", "threshold"],
            "test_used_for_selection": False,
            "scenarios": sensitivity_scenarios,
        },
        "operating_decision": decision,
        "validation": {
            "status": "passed", "train_test_disjoint": True,
            "oof_coverage_once": True, "fold_only_preprocessing_and_smote": True,
            "validation_samples_unchanged": True, "cost_and_metrics_recomputed": True,
            "thresholds_selected_without_test": True, "test_default_matches_existing_metrics": True,
            "cost_sensitivity_same_oof_and_threshold_candidates": True,
            "cost_sensitivity_arithmetic_and_optima_recomputed": True,
        },
    }
    output = ROOT / METRICS_PATH
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == results, "Saved evaluation results differ from calculated metrics."
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.parse_args()
    results = run_evaluation()
    print(json.dumps({
        "imbalance_validation_f1": {
            name: record["validation_mean"]["f1"] for name, record in results["imbalance_comparison"].items()
        },
        "operating_decision": results["operating_decision"],
        "selected_thresholds": {
            name: record["selected_threshold"] for name, record in results["threshold_selection"].items()
        },
        "validation": results["validation"]["status"], "metrics": METRICS_PATH,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
