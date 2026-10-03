"""Select one BCI IV 2a subject for four cued motor-imagery commands.

Within AxxT, task runs 1-4 train and 5-6 validate. Select a subject using
validation balanced accuracy only, then refit on all six AxxT runs and assess
the selected subject's independent AxxE session. No idle class is trained.
"""

import csv
import argparse
from datetime import datetime
from pathlib import Path
import time

import joblib
import mne
from mne.decoding import CSP
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FILTER_BANKS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
                (20.0, 24.0), (24.0, 30.0))
DATA_ROOT = Path(__file__).resolve().parent / "data" / "BCICIV_2a"
OUTPUT_DIR = Path(__file__).resolve().parent
CLASSES = ("LEFT_HAND", "RIGHT_HAND", "FEET", "TONGUE")
COMMANDS = ("LEFT_SHIFT", "RIGHT_SHIFT", "GRASP", "RELEASE")
CUE_OFFSET_SECONDS = 0.5
WINDOW_SECONDS = 2.0
RUN_TRIALS = 48
N_TRAIN_RUNS = 4


def causal_filter_at_run_boundaries(signals, sfreq, annotations):
    """Forward filter each run separately, matching the original experiment."""
    starts = [0]
    starts.extend(
        int(round(onset * sfreq))
        for onset, event in zip(annotations.onset, annotations.description)
        if event == "32766"
    )
    boundaries = sorted({x for x in starts if 0 <= x < signals.shape[1]})
    boundaries.append(signals.shape[1])
    banks = []
    for low, high in FILTER_BANKS:
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        result = np.empty_like(signals)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            result[:, start:stop] = sosfilt(sos, signals[:, start:stop], axis=1)
        banks.append(result)
    return banks


def load_session(subject, kind):
    stem = f"{subject}{kind}"
    raw = mne.io.read_raw_gdf(DATA_ROOT / f"{stem}.gdf", preload=True, verbose="ERROR")
    if len(raw.ch_names) < 25 or not all(name.startswith("EEG") for name in raw.ch_names[:22]):
        raise ValueError(f"{stem}: 通道排列不是预期的前 22 个 EEG")
    sfreq = float(raw.info["sfreq"])
    signals = raw.get_data(picks=raw.ch_names[:22])
    banks = causal_filter_at_run_boundaries(signals, sfreq, raw.annotations)
    trials = sorted(float(onset) for onset, code in zip(
        raw.annotations.onset, raw.annotations.description
    ) if code == "768")
    cues = sorted((float(onset), code) for onset, code in zip(
        raw.annotations.onset, raw.annotations.description
    ) if code in {"769", "770", "771", "772", "783"})
    rejected = np.asarray([float(onset) for onset, code in zip(
        raw.annotations.onset, raw.annotations.description
    ) if code == "1023"])
    labels = np.asarray(loadmat(DATA_ROOT / f"{stem}.mat")["classlabel"]).ravel()
    if len(trials) != 288 or len(cues) != 288 or len(labels) != 288:
        raise ValueError(f"{stem}: 预期 288 试次，实际开始 {len(trials)}、提示 {len(cues)}、标签 {len(labels)}")
    width = int(round(WINDOW_SECONDS * sfreq))
    windows = [[] for _ in FILTER_BANKS]
    y, rows, rejected_count = [], [], 0
    for number, (trial_start, (cue_start, cue), label) in enumerate(
        zip(trials, cues, labels), start=1
    ):
        if not 1.0 <= cue_start - trial_start <= 4.0:
            raise ValueError(f"{stem} 第 {number} 次试次的提示与开始未对齐")
        label = int(label)
        if label not in (1, 2, 3, 4):
            raise ValueError(f"{stem} 第 {number} 次标签不是 1–4")
        if cue != "783" and int(cue) - 768 != label:
            raise ValueError(f"{stem} 第 {number} 次 GDF 和 MAT 标签不一致")
        if np.any(np.abs(rejected - trial_start) < 0.05):
            rejected_count += 1
            continue
        start = int(round((cue_start + CUE_OFFSET_SECONDS) * sfreq))
        stop = start + width
        if stop > signals.shape[1]:
            raise ValueError(f"{stem} 第 {number} 次窗口越界")
        for index, bank in enumerate(banks):
            chunk = bank[:, start:stop]
            if chunk.shape != (22, width):
                raise ValueError(f"{stem} 第 {number} 次窗口不完整")
            windows[index].append(chunk.copy())
        y.append(label - 1)
        rows.append({"session": stem, "run": (number - 1) // RUN_TRIALS + 1,
                     "trial": number, "truth": CLASSES[label - 1]})
    raw.close()
    return ([np.asarray(chunks, dtype=np.float64) for chunks in windows],
            np.asarray(y, dtype=int), rows, rejected_count)


def slice_banks(x, mask):
    return [band[mask] for band in x]


def fit_model(x, y):
    csp_models, features = [], []
    for band in x:
        csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
        features.append(csp.fit_transform(band, y))
        csp_models.append(csp)
    joined = np.concatenate(features, axis=1)
    scaler = StandardScaler().fit(joined)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(scaler.transform(joined), y)
    return csp_models, scaler, classifier


def predict(model, x):
    csp_models, scaler, classifier = model
    features = np.concatenate([csp.transform(band) for csp, band in zip(csp_models, x)],
                              axis=1)
    probabilities = classifier.predict_proba(scaler.transform(features))
    return np.argmax(probabilities, axis=1), probabilities


def metrics(y, prediction):
    matrix = confusion_matrix(y, prediction, labels=range(4))
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return float(balanced_accuracy_score(y, prediction)), recalls, matrix


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="存放 A01T.gdf 等文件及同名 .mat 标签的目录")
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    mne.set_log_level("ERROR")
    started = time.time()
    rankings = []
    best = None
    for number in range(1, 10):
        subject = f"A{number:02d}"
        x, y, rows, rejected_count = load_session(subject, "T")
        run_numbers = np.asarray([row["run"] for row in rows])
        fit_mask = run_numbers <= N_TRAIN_RUNS
        val_mask = ~fit_mask
        if set(y[fit_mask]) != set(range(4)) or set(y[val_mask]) != set(range(4)):
            raise ValueError(f"{subject}: 训练或验证缺少某一类")
        model = fit_model(slice_banks(x, fit_mask), y[fit_mask])
        prediction, _ = predict(model, slice_banks(x, val_mask))
        score, recalls, _ = metrics(y[val_mask], prediction)
        rankings.append({
            "subject": subject, "train_trials": int(fit_mask.sum()),
            "validation_trials": int(val_mask.sum()), "rejected_trials": rejected_count,
            "validation_balanced_accuracy": f"{score:.6f}",
            **{f"recall_{name.lower()}": f"{recall:.6f}"
               for name, recall in zip(CLASSES, recalls)},
        })
        print(f"{subject}: 验证平衡准确率 {score:.3f}；" +
              "，".join(f"{name} {recall:.1%}" for name, recall in zip(CLASSES, recalls)),
              flush=True)
        rank = (score, min(recalls), -number)
        if best is None or rank > best[0]:
            best = (rank, subject)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_path = OUTPUT_DIR / f"bciciv2a_four_class_validation_{stamp}.csv"
    with ranking_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rankings[0]))
        writer.writeheader()
        writer.writerows(rankings)
    subject = best[1]
    print(f"\n仅根据 AxxT 运行段 5–6 选中 {subject}；现在使用其全部训练运行重新拟合。")
    x_train, y_train, _, train_rejected = load_session(subject, "T")
    final_model = fit_model(x_train, y_train)
    print(f"现在才读取 {subject}E 独立评估会话。")
    x_test, y_test, test_rows, test_rejected = load_session(subject, "E")
    prediction, probabilities = predict(final_model, x_test)
    score, recalls, matrix = metrics(y_test, prediction)
    print(f"测试有效试次 {len(y_test)}，跳过伪迹 {test_rejected}；"
          f"四类平衡准确率 {score:.3f}")
    print("测试各类召回：" + "，".join(
        f"{name} {recall:.1%}" for name, recall in zip(CLASSES, recalls)
    ))
    print("混淆矩阵行=真实、列=预测，顺序 " + ", ".join(CLASSES))
    print(matrix)
    model_path = OUTPUT_DIR / f"bciciv2a_{subject}_four_class_{stamp}.joblib"
    result_path = OUTPUT_DIR / f"bciciv2a_{subject}_four_class_test_{stamp}.csv"
    joblib.dump({
        "subject": subject, "model": final_model, "classes": list(CLASSES),
        "command_mapping": dict(zip(CLASSES, COMMANDS)),
        "train_session": f"{subject}T", "test_session": f"{subject}E",
        "selection": "AxxT task runs 1-4 train, runs 5-6 validation; chosen by balanced accuracy",
        "channels": "first 22 EEG channels; last 3 EOG channels excluded",
        "filter_bands_hz": [list(band) for band in FILTER_BANKS],
        "filter": "causal Butterworth order 4, reset at run boundaries",
        "window_seconds": WINDOW_SECONDS, "cue_offset_seconds": CUE_OFFSET_SECONDS,
        "note": "Offline cued four-class MI; no no-action class or continuous real-time control.",
    }, model_path)
    with result_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "session", "run", "trial", "truth", "prediction", "command"
        ] + [f"p_{name.lower()}" for name in CLASSES])
        writer.writeheader()
        for row, label, prob in zip(test_rows, prediction, probabilities):
            writer.writerow({**row, "prediction": CLASSES[label],
                             "command": COMMANDS[label],
                             **{f"p_{name.lower()}": f"{value:.6f}"
                                for name, value in zip(CLASSES, prob)}})
    print(f"验证比较：{ranking_path}")
    print(f"选中模型：{model_path}")
    print(f"测试逐试次：{result_path}")
    print(f"总耗时：{time.time() - started:.1f} 秒")
    print("这只是提示式四类运动想象；不含静息期、实时意图识别或实物抓取验证。")


if __name__ == "__main__":
    main()
