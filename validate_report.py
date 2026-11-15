"""Validate documented results against the saved, locally evaluated artifacts."""

import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


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


def supplement_sections(metrics: dict, evaluation: dict) -> dict[str, str]:
    """Build the follow-up report from the saved CV and operating results."""
    sections = {}
    ridge = metrics["regression"]["ridge"]["cv"]["candidates"]
    lasso = metrics["regression"]["lasso"]["cv"]["candidates"]
    lines = [
        "| Alpha | Ridge CV MAE 평균 ± 표준편차 | Lasso CV MAE 평균 ± 표준편차 |",
        "| --- | --- | --- |",
    ]
    for ridge_candidate, lasso_candidate in zip(ridge, lasso):
        alpha = ridge_candidate["parameters"]["model__estimator__alpha"]
        lines.append(
            f"| {alpha:g} | {ridge_candidate['mean_mae']:.4f} ± {ridge_candidate['std_mae']:.4f} "
            f"| {lasso_candidate['mean_mae']:.4f} ± {lasso_candidate['std_mae']:.4f} |"
        )
    lines.extend([
        "",
        f"동일한 Train 5-fold에서 RMSE와 MAE를 함께 측정했다. Ridge는 Alpha 0.01→10에서 "
        f"MAE가 {ridge[0]['mean_mae']:.4f}→{ridge[3]['mean_mae']:.4f}로 변했다. "
        f"Lasso는 Alpha 0.1→10→100에서 MAE가 {lasso[1]['mean_mae']:.4f}→"
        f"{lasso[3]['mean_mae']:.4f}→{lasso[4]['mean_mae']:.4f}로 변해 강한 규제의 영향을 보여준다. "
        "두 오차의 단위는 신용 점수이며, Alpha는 CV RMSE로 선택하고 선택 모델의 Test RMSE·MAE는 위 표에 보고했다. "
        "[Alpha별 MAE 원점수·집계](artifacts/metrics.json)",
    ])
    sections["alpha-mae"] = "\n".join(lines)

    tree = evaluation["ensemble_comparison"]["decision_tree"]
    forest = evaluation["ensemble_comparison"]["random_forest"]
    change = forest["validation_f1_std"] - tree["validation_f1_std"]
    direction = "증가" if change >= 0 else "감소"
    sections["fold-dispersion"] = (
        f"같은 5개 검증 폴드에서 F1 표준편차는 단일 트리 {tree['validation_f1_std']:.4f}에서 "
        f"RF {forest['validation_f1_std']:.4f}로 {abs(change):.4f} {direction}했다. "
        "표준편차는 np.std(ddof=0)으로 계산했다. RF의 검증 평균과 Train−Validation 격차는 개선되었고, "
        "폴드 점수 변동은 위 수치로 관찰된다. RF 파라미터는 기존 Train CV 선택값을 고정한 조건부 비교이며, "
        "이론적 편향·분산의 직접 분해에는 별도 반복 표본 실험이 필요하다. "
        "[동일 폴드 비교 원점수](artifacts/evaluation_metrics.json)"
    )

    names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
    eligible = evaluation["operating_decision"]["eligible_models"]
    fastest = min(eligible, key=lambda name: (
        evaluation["benchmarks"][name]["single_request_p95_ms"],
        evaluation["benchmarks"][name]["serialized_model_bytes"], name,
    ), default=None)
    lines = [
        "| 업무 사례 | 후보 모델 | 임계값 | Train OOF 비용 | Train OOF Recall | p95(ms) | 모델 크기(bytes) | 선택 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    cases = (("연체 미탐 비용 우선 사전심사", evaluation["operating_decision"]["model"]),
             ("응답시간 우선 대량 요청", fastest))
    for case, selected in cases:
        for name, label in names.items():
            record = evaluation["threshold_selection"][name]
            benchmark = evaluation["benchmarks"][name]
            choice = "채택" if name == selected else "비교 후보"
            lines.append(
                f"| {case} | {label} | {record['selected_threshold']:.2f} "
                f"| {record['oof_selected']['cost']} | {record['oof_selected']['recall']:.4f} "
                f"| {benchmark['single_request_p95_ms']:.4f} | {benchmark['serialized_model_bytes']} | {choice} |"
            )
    lines.extend([
        "",
        "두 사례 모두 예시 한도인 p95 100ms·직렬화 모델 10MiB를 적용한다. 사전심사는 한도를 만족하는 후보의 "
        "Train OOF 비용을 우선하고, 대량 요청은 적격 후보 중 p95·모델 크기 순서로 선택한다. "
        "비용은 FP 1·FN 10의 가정이며 임계값은 Train OOF에서 확정했다. 측정 환경과 반복 횟수는 위 벤치마크 설명을 따른다. "
        "[운영 한도·실측값·선택 근거](artifacts/evaluation_metrics.json)",
    ])
    sections["operating-cases"] = "\n".join(lines)

    lines = [
        "| FP:FN | 모델 | 기준 | 임계값 | Train OOF 비용 | Test FP | Test FN | Test Precision | Test Recall | Test 비용 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in evaluation["cost_sensitivity"]["scenarios"]:
        ratio = f"{scenario['fp_cost']}:{scenario['fn_cost']}"
        for name, label in names.items():
            selection = scenario["models"][name]
            for suffix, method in (("default", "기본"), ("selected", "비용 선택")):
                record = selection[f"test_{suffix}"]
                oof = selection[f"oof_{suffix}"]
                lines.append(
                    f"| {ratio} | {label} | {method} | {record['threshold']:.2f} | {oof['cost']} "
                    f"| {record['fp']} | {record['fn']} | {record['precision']:.4f} "
                    f"| {record['recall']:.4f} | {record['cost']} |"
                )
    sections["cost-sensitivity"] = "\n".join(lines)
    return sections


def refresh_readme() -> int:
    """Refresh measured rows while preserving the surrounding report text."""
    evaluation = json.loads((ROOT / "artifacts/evaluation_metrics.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
    path = ROOT / "README.md"
    original = path.read_bytes()
    lines = original.decode("utf-8-sig").splitlines(keepends=True)
    rows = measured_rows(evaluation)
    table_headers = {
        6: ["처리", "CV Precision", "CV Recall", "CV F1", "CV AUC", "평균 학습시간(초)"],
        5: ["모델", "단일 요청 p95(ms)", "일괄 처리량(건/초)",
            "직렬화 모델 크기(bytes)", "Python 추적 추론 최대 할당(bytes)"],
    }
    for cells in rows:
        positions = []
        current_header = []
        for index, line in enumerate(lines):
            if not line.strip().startswith("|"):
                current_header = []
                continue
            existing = [cell.strip() for cell in line.strip().split("|")[1:-1]]
            if not current_header:
                current_header = existing
            if (current_header == table_headers[len(cells)]
                    and len(existing) == len(cells) and existing[0] == cells[0]):
                positions.append(index)
        assert len(positions) == 1, f"Expected one measurement row for {cells[0]}."
        position = positions[0]
        ending = lines[position][len(lines[position].rstrip("\r\n")):]
        lines[position] = "| " + " | ".join(cells) + " |" + ending
    encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
    document = "".join(lines)
    newline = "\r\n" if b"\r\n" in original else "\n"
    for name, content in supplement_sections(metrics, evaluation).items():
        start = f"<!-- {name}:start -->"
        end = f"<!-- {name}:end -->"
        assert document.count(start) == document.count(end) == 1, f"Missing report section {name}."
        before, rest = document.split(start)
        _, after = rest.split(end)
        document = before + start + newline + content.replace("\n", newline) + newline + end + after
    path.write_bytes(document.encode(encoding))
    return len(rows)


def require_row(document: str, cells: list[str]) -> None:
    expected = "| " + " | ".join(cells) + " |"
    assert expected in document.splitlines(), f"README row differs from saved metrics: {expected}"


def validate_links(path: Path) -> int:
    document = path.read_text(encoding="utf-8-sig")
    count = 0
    for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document):
        destination = unquote(destination.strip("<>").split("#", 1)[0])
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
    for name, content in supplement_sections(metrics, evaluation).items():
        assert content in readme, f"Follow-up section differs from saved results: {name}."

    api_examples = [block for block in re.findall(r"```python\s*\n(.*?)```", readme, re.DOTALL)
                    if "preprocessor = make_preprocessor()" in block]
    assert len(api_examples) == 1, "Expected one executable preprocessing/model API example."
    exec(compile(api_examples[0], "README.md API example", "exec"), {"__name__": "__report_example__"})
    for filename, functions in (
        ("modeling.py", ("save_coefficient_plot", "save_classification_plots")),
        ("bonus.py", ("run_bonus",)),
    ):
        source = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
        definitions = {node.name: node.lineno for node in source.body if isinstance(node, ast.FunctionDef)}
        for function in functions:
            assert f"({filename}#L{definitions[function]})" in readme, f"Stale plot source link: {function}."
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

    followup_path = ROOT / "docs/A5_1 mission_naito_eval2.md"
    followup = followup_path.read_text(encoding="utf-8-sig")
    assert len(re.findall(r"^\* 대책:", followup, re.MULTILINE)) == 16
    assert len(re.findall(r"^\* 조치 결과:", followup, re.MULTILINE)) == 16
    for number in range(1, 17):
        assert f"평가 항목 #{number} [ ✓ ]" in followup, "Original follow-up evaluation was changed."
    assert len(re.findall(r"^\* 대책:[^\n]*\n\* 조치 결과:", followup, re.MULTILINE)) == 16

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

    documents = (readme_path, assessment_path, followup_path, ROOT / "plans/A5_1.md")
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
        "followup_countermeasures": 16,
        "alpha_mae_rows": len(ridge_candidates),
        "cost_sensitivity_rows": len(evaluation["cost_sensitivity"]["scenarios"]) * 4,
        "api_examples_executed": len(api_examples),
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
