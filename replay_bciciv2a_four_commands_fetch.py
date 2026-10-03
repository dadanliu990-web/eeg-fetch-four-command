"""Replay one person's offline four-class EEG predictions in FetchPickAndPlace.

This is a cue-by-cue simulation. The EEG classifier chooses a high-level command;
the robot's motion within GRASP and RELEASE is scripted. It is not live EEG control.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np


gym.register_envs(gymnasium_robotics)

ROOT = Path(__file__).resolve().parent
CLASS_TO_COMMAND = {
    "LEFT_HAND": "LEFT_SHIFT",
    "RIGHT_HAND": "RIGHT_SHIFT",
    "FEET": "GRASP",
    "TONGUE": "RELEASE",
}
GAIN = 5.0
POSITION_TOLERANCE = 0.025
SUCCESS_TOLERANCE = 0.05
OPEN = 1.0
CLOSED = -1.0


def read_predictions(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"预测文件为空：{path}")
    for row in rows:
        if row["prediction"] not in CLASS_TO_COMMAND:
            raise ValueError(f"未知预测类别：{row['prediction']}")
        if row["truth"] not in CLASS_TO_COMMAND:
            raise ValueError(f"未知真实类别：{row['truth']}")
    return rows


def one_step(env, observation, target, grip):
    action = np.zeros(4, dtype=np.float32)
    if target is not None:
        action[:3] = np.clip(
            (target - observation["observation"][:3]) * GAIN, -1.0, 1.0
        )
    action[3] = grip
    next_observation, _, terminated, truncated, _ = env.step(action)
    return next_observation, terminated or truncated


def move_to(env, observation, target, grip, max_steps=60, stop_when=None, tolerance=POSITION_TOLERANCE):
    for step in range(max_steps + 1):
        if stop_when is not None and stop_when(observation):
            return observation, step, True
        if np.linalg.norm(target - observation["observation"][:3]) < tolerance:
            return observation, step, True
        if step == max_steps:
            break
        observation, ended = one_step(env, observation, target, grip)
        if ended:
            return observation, step + 1, False
    return observation, max_steps, False


def hold_gripper(env, observation, grip, steps=20):
    for step in range(steps):
        observation, ended = one_step(env, observation, None, grip)
        if ended:
            return observation, step + 1, False
    return observation, steps, True


def grasp(env, observation):
    block = observation["observation"][3:6].copy()
    total = 0
    for target, grip, phase in (
        (block + np.array([0.0, 0.0, 0.12]), OPEN, "approach"),
        (block, OPEN, "descend"),
    ):
        observation, steps, reached = move_to(env, observation, target, grip)
        total += steps
        if not reached:
            return observation, total, f"failed_{phase}"

    observation, steps, finished = hold_gripper(env, observation, CLOSED)
    total += steps
    if not finished:
        return observation, total, "failed_close"

    lifted = lambda obs: obs["observation"][5] >= block[2] + 0.04
    observation, steps, _ = move_to(
        env,
        observation,
        block + np.array([0.0, 0.0, 0.20]),
        CLOSED,
        max_steps=55,
        stop_when=lifted,
    )
    total += steps
    if lifted(observation):
        return observation, total, "grasped"
    return observation, total, "failed_no_lift"


def release(env, observation):
    goal = observation["desired_goal"].copy()
    total = 0
    for target, phase in (
        (goal + np.array([0.0, 0.0, 0.16]), "transport"),
        (goal + np.array([0.0, 0.0, 0.03]), "place"),
    ):
        observation, steps, reached = move_to(env, observation, target, CLOSED)
        total += steps
        if not reached:
            return observation, total, f"failed_{phase}", True

    observation, steps, finished = hold_gripper(env, observation, OPEN)
    total += steps
    if not finished:
        return observation, total, "failed_open", False
    error = np.linalg.norm(observation["achieved_goal"] - goal)
    status = "placed" if error < SUCCESS_TOLERANCE else "released_outside_goal"
    return observation, total, status, False


def shift(env, observation, direction, holding):
    before = observation["observation"][:3].copy()
    target = before + np.array([0.0, direction * 0.04, 0.0])
    observation, steps, reached = move_to(
        env, observation, target, CLOSED if holding else OPEN, max_steps=30, tolerance=0.005
    )
    delta_y = float(observation["observation"][1] - before[1])
    status = "moved" if reached and direction * delta_y >= 0.025 else "failed_shift"
    return observation, steps, status, delta_y


def latest_prediction_file():
    files = sorted(ROOT.glob("bciciv2a_A??_four_class_test_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("未找到四分类预测 CSV；请先运行四分类训练程序。")
    return files[-1]


def load_policy(path, subject):
    if path is None:
        files = sorted(
            ROOT.glob(f"bciciv2a_{subject}_abstain_policy_*.json"),
            key=lambda item: item.stat().st_mtime,
        )
        if not files:
            raise FileNotFoundError(
                f"未找到 {subject} 低置信度策略；请先运行 calibrate_bciciv2a_A07_abstain.py，"
                "或使用 --no-abstain 回放旧策略。"
            )
        path = files[-1]
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["subject"] != subject:
        raise ValueError(f"策略受试者 {policy['subject']} 与预测受试者 {subject} 不一致")
    threshold = float(policy["threshold"])
    if not 0.25 <= threshold <= 1.0:
        raise ValueError(f"无效置信度阈值：{threshold}")
    return path, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="四分类逐试次预测 CSV，默认读取最新文件")
    parser.add_argument("--run", type=int, choices=range(1, 7), help="只回放指定 A07E 轮次")
    parser.add_argument("--limit", type=int, default=12, help="最多回放多少条；0 表示全部（默认 12）")
    parser.add_argument("--render", action="store_true", help="显示 MuJoCo 仿真窗口")
    parser.add_argument("--seed", type=int, default=700, help="Fetch 场景随机种子基数")
    parser.add_argument("--policy", type=Path, help="低置信度不动作策略 JSON；默认读取同一受试者最新策略")
    parser.add_argument("--no-abstain", action="store_true", help="禁用低置信度不动作规则，以便与原方案对照")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 不能为负数")

    source = args.csv or latest_prediction_file()
    rows = read_predictions(source)
    if args.run is not None:
        rows = [row for row in rows if int(row["run"]) == args.run]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        parser.error("筛选后没有可回放的试次")

    subjects = {row["session"][:3] for row in rows}
    if len(subjects) != 1:
        parser.error("一份回放 CSV 只能包含一位受试者")
    subject = subjects.pop()
    if args.no_abstain:
        policy_path, threshold = None, 0.0
    else:
        policy_path, threshold = load_policy(args.policy, subject)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / f"bciciv2a_four_command_fetch_replay_{timestamp}.csv"
    print(f"读取离线预测：{source}")
    print("左手→左移；右手→右移；双脚→抓取；舌头→放置并松开。")
    print("每行 EEG 预测只触发一次指令；无持续静息检测。")
    print(f"低置信度规则：{'关闭' if args.no_abstain else f'最高类别概率 < {threshold:.3f} 时不动作'}")
    if policy_path:
        print(f"策略文件：{policy_path}")

    env = gym.make(
        "FetchPickAndPlace-v4",
        render_mode="human" if args.render else None,
        max_episode_steps=100000,
    )
    results = []
    observation = None
    current_run = None
    holding = False
    reset_next = False
    try:
        for index, row in enumerate(rows, start=1):
            run = int(row["run"])
            if observation is None or run != current_run or reset_next:
                observation, _ = env.reset(seed=args.seed + run * 100 + index)
                # The existing Fetch demo places the target at table height.
                env.unwrapped.goal[2] = 0.42
                observation["desired_goal"] = env.unwrapped.goal.copy()
                holding = False
                current_run = run
                reset_next = False

            predicted_command = CLASS_TO_COMMAND[row["prediction"]]
            confidence = max(float(row[f"p_{name.lower()}"]) for name in CLASS_TO_COMMAND)
            command = predicted_command if confidence >= threshold else "NO_ACTION"
            holding_before = holding
            steps = 0
            delta_y = 0.0
            if command == "NO_ACTION":
                status = "no_action_low_confidence"
            elif command in ("LEFT_SHIFT", "RIGHT_SHIFT"):
                direction = 1 if command == "LEFT_SHIFT" else -1
                observation, steps, status, delta_y = shift(
                    env, observation, direction, holding
                )
            elif command == "GRASP":
                if holding:
                    status = "ignored_already_holding"
                else:
                    observation, steps, status = grasp(env, observation)
                    holding = status == "grasped"
            else:
                if not holding:
                    status = "ignored_not_holding"
                else:
                    observation, steps, status, holding = release(env, observation)
                    if status in ("placed", "released_outside_goal"):
                        reset_next = True

            error = float(
                np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
            )
            result = {
                "session": row["session"],
                "run": run,
                "trial": row["trial"],
                "truth": row["truth"],
                "prediction": row["prediction"],
                "eeg_correct": int(row["truth"] == row["prediction"]),
                "confidence": f"{confidence:.6f}",
                "predicted_command": predicted_command,
                "command": command,
                "robot_status": status,
                "robot_steps": steps,
                "delta_y_m": f"{delta_y:.4f}",
                "block_goal_distance_m": f"{error:.4f}",
                "holding_before": int(holding_before),
                "holding_after": int(holding),
            }
            results.append(result)
            if args.render or index <= 12 or index % 25 == 0 or index == len(rows):
                print(
                    f"{index:3d}/{len(rows)} R{run} 试次 {int(row['trial']):2d} | "
                    f"真实 {row['truth']:10s} | 预测 {row['prediction']:10s} | "
                    f"{command:11s} | 置信度 {confidence:.3f} | {status:23s} | {steps:3d} 步"
                )
    finally:
        env.close()

    with log_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["robot_status"] for row in results)
    correct = sum(row["eeg_correct"] for row in results)
    accepted = [row for row in results if row["command"] != "NO_ACTION"]
    accepted_correct = sum(row["eeg_correct"] for row in accepted)
    suppressed_errors = sum(
        not row["eeg_correct"] for row in results if row["command"] == "NO_ACTION"
    )
    print(f"\nEEG 预测正确：{correct}/{len(results)}")
    print(f"已发出指令：{len(accepted)}/{len(results)}；其中预测正确 {accepted_correct}/{len(accepted)}"
          if accepted else f"已发出指令：0/{len(results)}")
    print(f"拦下错误预测：{suppressed_errors}；同时拦下正确预测："
          f"{len(results) - len(accepted) - suppressed_errors}")
    print("机器人动作结果：" + "；".join(f"{key} {value}" for key, value in sorted(counts.items())))
    print(f"逐试次回放日志：{log_path}")
    print("说明：这是公开数据的提示式离线回放；机器人抓放轨迹由手工规则控制。")


if __name__ == "__main__":
    main()
