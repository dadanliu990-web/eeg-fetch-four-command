"""Choose A07's no-action confidence threshold using A07T runs only.

Runs 1-4 fit a temporary model; runs 5-6 choose one fixed threshold. A07E is
never opened here. The threshold is later applied to the saved A07E predictions.
"""

import contextlib
import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import select_bciciv2a_four_commands as experiment

from select_bciciv2a_four_commands import (
    CLASSES,
    fit_model,
    load_session,
    predict,
    slice_banks,
)


ROOT = Path(__file__).resolve().parent
SUBJECT = "A07"
CANDIDATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975)
TARGET_ACCURACY = 0.90
MIN_COVERAGE = 0.50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=experiment.DATA_ROOT,
                        help="存放 A07T.gdf 和 A07T.mat 的目录")
    args = parser.parse_args()
    experiment.DATA_ROOT = args.data_root
    mne.set_log_level("ERROR")
    banks, labels, rows, _ = load_session(SUBJECT, "T")
    runs = np.array([int(row["run"]) for row in rows])
    train = runs <= 4
    validate = runs >= 5
    with contextlib.redirect_stdout(io.StringIO()):
        model = fit_model(slice_banks(banks, train), labels[train])
        predicted, probabilities = predict(model, slice_banks(banks, validate))
    confidence = probabilities.max(axis=1)
    correct = predicted == labels[validate]

    comparisons = []
    for threshold in CANDIDATES:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        comparisons.append({
            "threshold": threshold,
            "accepted": count,
            "total": len(correct),
            "coverage": count / len(correct),
            "accepted_accuracy": float(correct[accepted].mean()) if count else None,
            "accepted_errors": int((~correct & accepted).sum()),
        })

    eligible = [
        item for item in comparisons
        if item["accepted_accuracy"] is not None
        and item["accepted_accuracy"] >= TARGET_ACCURACY
        and item["coverage"] >= MIN_COVERAGE
    ]
    if not eligible:
        raise RuntimeError("验证集没有达到预设正确率和覆盖率的阈值；未生成策略。")
    chosen = min(eligible, key=lambda item: item["threshold"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_path = ROOT / f"bciciv2a_{SUBJECT}_abstain_policy_{stamp}.json"
    validation_path = ROOT / f"bciciv2a_{SUBJECT}_abstain_validation_{stamp}.csv"
    policy = {
        "subject": SUBJECT,
        "threshold": chosen["threshold"],
        "rule": "max(class_probability) >= threshold => command; otherwise NO_ACTION",
        "selected_using": "A07T runs 1-4 fit, runs 5-6 validate",
        "selection_rule": "lowest threshold among fixed candidates with accepted accuracy >= 0.90 and coverage >= 0.50",
        "target_accuracy": TARGET_ACCURACY,
        "minimum_coverage": MIN_COVERAGE,
        "chosen_validation_result": chosen,
        "all_validation_results": comparisons,
        "note": "A07E was not used to choose the threshold. Classifier probabilities are not calibrated guarantees.",
    }
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    with validation_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "session", "run", "trial", "truth", "prediction", "confidence", "accepted", "correct"
        ])
        writer.writeheader()
        for row, label, conf, is_correct in zip(np.array(rows, dtype=object)[validate], predicted, confidence, correct):
            writer.writerow({
                **row,
                "prediction": CLASSES[label],
                "confidence": f"{conf:.6f}",
                "accepted": int(conf >= chosen["threshold"]),
                "correct": int(is_correct),
            })
    print(f"A07T 验证试次：{len(correct)}；未筛选正确：{int(correct.sum())}/{len(correct)}")
    for result in comparisons:
        accuracy = result["accepted_accuracy"]
        print(f"阈值 {result['threshold']:.3f} | 保留 {result['accepted']:2d}/{len(correct)} | "
              f"保留指令正确率 {accuracy:.3f}" if accuracy is not None else
              f"阈值 {result['threshold']:.3f} | 没有保留指令")
    print(f"已选阈值：{chosen['threshold']:.3f}")
    print(f"策略：{policy_path}")
    print(f"验证记录：{validation_path}")
    print("只读取 A07T；尚未以 A07E 调整阈值。")


if __name__ == "__main__":
    main()
