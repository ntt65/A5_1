"""Validate documented results against the saved, locally evaluated artifacts."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent


def measured_rows(evaluation: dict) -> list[list[str]]:
    """Build the rows whose timing measurements change on each execution."""
    rows = []
    names = {"unweighted": "무보정", "balanced": "클래스 가중치", "smote": "SMOTE"}
    for name, label in names.items():
        record = evaluation["imbalance_comparison"][name]
        rows.append([label] + [
            f"{record['validation_mean'][field]:.4f}" for field in ("precision", "recall", "f1", "auc")
        ] + [f"{record['fit_seconds_mean']:.4f}"])
    for name, label in (("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")):
        benchmark = evaluation["benchmarks"][name]
        rows.append([
            label, f"{benchmark['single_request_p95_ms']:.4f}",
            f"{benchmark['batch_samples_per_second']:.4f}",
            str(benchmark["serialized_model_bytes"]), str(benchmark["python_traced_inference_peak_bytes"]),
        ])
    return rows


def refresh_readme() -> int:
    """Refresh measured rows while preserving the surrounding report text."""
    evaluation = json.loads((ROOT / "artifacts/evaluation_metrics.json").read_text(encoding="utf-8"))
    path = ROOT / "README.md"
    original = path.read_bytes()
    lines = original.decode("utf-8-sig").splitlines(keepends=True)
    rows = measured_rows(evaluation)
    for cells in rows:
        positions = []
        for index, line in enumerate(lines):
            existing = [cell.strip() for cell in line.strip().split("|")[1:-1]]
            if len(existing) == len(cells) and existing[0] == cells[0]:
                positions.append(index)
        assert len(positions) == 1, f"Expected one measurement row for {cells[0]}."
        position = positions[0]
        ending = lines[position][len(lines[position].rstrip("\r\n")):]
        lines[position] = "| " + " | ".join(cells) + " |" + ending
    encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
    path.write_bytes("".join(lines).encode(encoding))
    return len(rows)


def require_row(document: str, cells: list[str]) -> None:
    expected = "| " + " | ".join(cells) + " |"
    assert expected in document.splitlines(), f"README row differs from saved metrics: {expected}"


def validate_links(path: Path) -> int:
    document = path.read_text(encoding="utf-8-sig")
    count = 0
    for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document):
        destination = destination.split("#", 1)[0]
        if not destination or "://" in destination or destination.startswith("mailto:"):
            continue
        target = path.parent / destination
        assert target.exists(), f"Missing document link in {path.name}: {destination}"
        count += 1
    return count


def validate_report() -> dict:
    assert sys.version_info[:2] == (3, 11), "Python 3.11 is required."
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
    bonus = json.loads((ROOT / "artifacts/bonus_metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "artifacts/evaluation_metrics.json").read_text(encoding="utf-8"))
    for results in (metrics, bonus, evaluation):
        assert results["validation"]["status"] == "passed", "Artifact validation is incomplete."

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8-sig")
    classification_names = {
        "rule_based": "Rule-based",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
    for name, label in classification_names.items():
        record = metrics["classification"][name]
        require_row(readme, [label] + [
            f"{record[field]:.4f}" for field in ("accuracy", "precision", "recall", "f1", "auc")
        ])
    for name in ("ridge", "lasso"):
        record = metrics["regression"][name]
        require_row(readme, [name.title()] + [
            f"{record[field]:.4f}" for field in ("alpha", "rmse", "mae", "r2")
        ])

    ridge_candidates = metrics["regression"]["ridge"]["cv"]["candidates"]
    lasso_candidates = metrics["regression"]["lasso"]["cv"]["candidates"]
    assert len(ridge_candidates) == len(lasso_candidates) == 5, "Missing Alpha CV candidates."
    for ridge, lasso in zip(ridge_candidates, lasso_candidates):
        alpha = ridge["parameters"]["model__estimator__alpha"]
        assert alpha == lasso["parameters"]["model__estimator__alpha"]
        require_row(readme, [
            f"{alpha:g}",
            f"{-ridge['mean_score']:.4f} ± {ridge['std_score']:.4f}",
            f"{-lasso['mean_score']:.4f} ± {lasso['std_score']:.4f}",
        ])

    contributions = metrics["classification"]["rule_based"]["rule_contributions"]
    for rule in contributions["rules"]:
        require_row(readme, [
            str(rule["rule_number"]), f"`{rule['condition']}`",
            str(rule["applied"]), str(rule["tp"]), str(rule["fp"]),
        ])

    for cells in measured_rows(evaluation):
        require_row(readme, cells)
    for name, label in (("decision_tree", "Decision Tree"), ("random_forest", "Random Forest")):
        record = evaluation["ensemble_comparison"][name]
        require_row(readme, [
            label,
            f"{record['train_f1_mean']:.4f} ± {record['train_f1_std']:.4f}",
            f"{record['validation_f1_mean']:.4f} ± {record['validation_f1_std']:.4f}",
            f"{record['generalization_gap']:.4f}",
        ])
    for name in ("logistic_regression", "random_forest"):
        label = classification_names[name]
        selection = evaluation["threshold_selection"][name]
        require_row(readme, [
            label, f"{selection['selected_threshold']:.2f}", str(selection["oof_selected"]["cost"]),
            str(selection["test_default"]["cost"]), str(selection["test_selected"]["cost"]),
            str(selection["test_selected"]["fp"]), str(selection["test_selected"]["fn"]),
        ])

    for name, comparison in metrics["f1_comparison"]["models"].items():
        assert f"{comparison['absolute_gain']:.4f}" in readme, f"Missing {name} F1 gain."
        assert f"{comparison['relative_gain_percent']:.2f}%" in readme, f"Missing {name} relative F1 gain."

    assessment_path = ROOT / "docs/A5_1_mission_naito_eval.md"
    assessment = assessment_path.read_text(encoding="utf-8-sig")
    assert len(re.findall(r"^\* 대책:", assessment, re.MULTILINE)) == 16
    assert len(re.findall(r"^\* 조치 결과:", assessment, re.MULTILINE)) == 16
    for number in (13, 15, 16):
        assert f"평가 항목 #{number} [ X ]" in assessment, "Original evaluation record was changed."

    commands = (
        "uv pip check --python .venv",
        "data_gen.py", "preprocess.py --validate", "modeling.py --validate",
        "bonus.py --validate", "evaluation.py --validate", "validate_report.py --refresh --validate",
    )
    assert any(all(command in block for command in commands)
               for block in re.findall(r"```powershell\s*\n(.*?)```", readme, re.DOTALL)), \
        "The single PowerShell validation block is incomplete."

    dependencies = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        requirement = line.strip()
        if requirement and not requirement.startswith("#"):
            package, version = requirement.split("==")
            assert importlib.metadata.version(package) == version, f"Version mismatch for {package}."
            dependencies[package] = version
    assert dependencies["imbalanced-learn"] == "0.14.0"

    documents = (readme_path, assessment_path, ROOT / "plans/A5_1.md")
    return {
        "status": "passed",
        "classification_rows": 3,
        "regression_rows": 2,
        "alpha_cv_rows": len(ridge_candidates),
        "rule_contribution_rows": len(contributions["rules"]),
        "imbalance_rows": 3,
        "ensemble_rows": 2,
        "benchmark_rows": 2,
        "threshold_rows": 2,
        "countermeasures": 16,
        "local_document_links": sum(validate_links(path) for path in documents),
        "dependencies": dependencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", required=True)
    parser.add_argument("--refresh", action="store_true", help="Refresh measured README rows before validation.")
    arguments = parser.parse_args()
    refreshed = 0
    if arguments.refresh:
        refreshed = refresh_readme()
    results = validate_report()
    results["refreshed_measurement_rows"] = refreshed
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
