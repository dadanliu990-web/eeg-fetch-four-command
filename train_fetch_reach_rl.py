"""Train and inspect a first reinforcement-learning policy for FetchReach.

This learns robot motion from simulator reward. It does not train an EEG decoder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReachDense-v4"
MODEL_PATH = Path(__file__).with_name("fetch_reach_sac.zip")
REPORT_PATH = Path(__file__).with_name("fetch_reach_rl_report.json")
SUCCESS_DISTANCE_M = 0.05


class TrainingProgress(BaseCallback):
    def __init__(self, total_steps: int, interval: int) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.interval = max(1, interval)
        self.next_print = self.interval
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_print or self.num_timesteps >= self.total_steps:
            elapsed = time.monotonic() - self.started
            print(
                f"训练进度：{self.num_timesteps}/{self.total_steps} 步；"
                f"耗时 {elapsed:.0f} 秒",
                flush=True,
            )
            self.next_print += self.interval
        return True


def evaluate(
    model: SAC | None,
    *,
    episodes: int,
    first_seed: int,
    render: bool = False,
    pause: float = 0.0,
) -> dict:
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    rows = []
    try:
        for episode in range(episodes):
            seed = first_seed + episode
            observation, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            success = False
            steps = 0
            distance = float(
                np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
            )
            for steps in range(1, env.spec.max_episode_steps + 1):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                distance = float(
                    np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
                )
                success = bool(info.get("is_success", distance < SUCCESS_DISTANCE_M))
                if pause:
                    time.sleep(pause)
                if success or terminated or truncated:
                    break
            rows.append(
                {
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "final_distance_m": round(distance, 4),
                }
            )
            if render:
                label = "成功" if success else "未到达"
                print(f"第 {episode + 1}/{episodes} 次：{label}，{steps} 步，末端误差 {distance:.3f} m")
                time.sleep(0.8)
    finally:
        env.close()

    successful_steps = [row["steps"] for row in rows if row["success"]]
    return {
        "episodes": episodes,
        "successes": len(successful_steps),
        "success_rate": len(successful_steps) / episodes,
        "avg_success_steps": round(float(np.mean(successful_steps)), 1)
        if successful_steps
        else None,
        "details": rows,
    }


def train(args: argparse.Namespace) -> None:
    print(f"仿真任务：{ENV_ID}")
    print("算法：SAC；奖励：每步按末端与目标的距离计算。")
    print("先以相同随机场景测量随机动作，再训练机器人动作策略。", flush=True)
    baseline = evaluate(None, episodes=args.eval_episodes, first_seed=args.eval_seed)
    print(f"训练前随机动作到达率：{baseline['successes']}/{args.eval_episodes}", flush=True)

    env = gym.make(ENV_ID)
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=0,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.95,
            policy_kwargs={"net_arch": [128, 128]},
        )
        print(f"开始训练 {args.steps} 步；计算设备：{model.device}", flush=True)
        started = time.monotonic()
        model.learn(
            total_timesteps=args.steps,
            callback=TrainingProgress(args.steps, args.progress_every),
        )
        elapsed = time.monotonic() - started
        model.save(MODEL_PATH)
    finally:
        env.close()

    learned = evaluate(model, episodes=args.eval_episodes, first_seed=args.eval_seed)
    print(f"训练后模型到达率：{learned['successes']}/{args.eval_episodes}")
    if learned["avg_success_steps"] is not None:
        print(f"成功回合平均步数：{learned['avg_success_steps']}")
    print(f"模型已保存：{MODEL_PATH}")

    report = {
        "environment": ENV_ID,
        "algorithm": "SAC",
        "training_steps": args.steps,
        "training_seconds": round(elapsed, 1),
        "train_seed": args.seed,
        "eval_first_seed": args.eval_seed,
        "random_policy": baseline,
        "trained_policy": learned,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"逐回合结果：{REPORT_PATH}")
    print("观看训练后的动作：用同一个脚本执行 watch --episodes 5")


def watch(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"找不到模型：{MODEL_PATH}；请先执行 train。")
    model = SAC.load(MODEL_PATH, device=args.device)
    print("打开 Fetch 动画窗口，按 Ctrl+C 可结束。", flush=True)
    result = evaluate(
        model,
        episodes=args.episodes,
        first_seed=args.seed,
        render=True,
        pause=args.pause,
    )
    print(f"观看回合到达率：{result['successes']}/{args.episodes}")


def evaluate_saved(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"找不到模型：{MODEL_PATH}；请先执行 train。")
    model = SAC.load(MODEL_PATH, device=args.device)
    random_result = evaluate(None, episodes=args.episodes, first_seed=args.seed)
    learned_result = evaluate(model, episodes=args.episodes, first_seed=args.seed)
    print(f"相同 {args.episodes} 个新场景：")
    print(f"随机动作到达率：{random_result['successes']}/{args.episodes}")
    print(f"训练模型到达率：{learned_result['successes']}/{args.episodes}")
    if learned_result["avg_success_steps"] is not None:
        print(f"模型成功回合平均步数：{learned_result['avg_success_steps']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FetchReach 强化学习入门实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并评估 SAC 策略")
    train_parser.add_argument("--steps", type=int, default=20_000)
    train_parser.add_argument("--eval-episodes", type=int, default=20)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--eval-seed", type=int, default=10_000)
    train_parser.add_argument("--progress-every", type=int, default=2_000)
    train_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    watch_parser = subparsers.add_parser("watch", help="观看已保存策略的动作")
    watch_parser.add_argument("--episodes", type=int, default=10)
    watch_parser.add_argument("--seed", type=int, default=20_000)
    watch_parser.add_argument("--pause", type=float, default=0.35)
    watch_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    eval_parser = subparsers.add_parser("eval", help="用新场景比较随机动作和已保存模型")
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=30_000)
    eval_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()
    if args.command == "train":
        if args.steps <= 0 or args.eval_episodes <= 0:
            parser.error("--steps 和 --eval-episodes 必须为正整数")
        train(args)
    elif args.command == "watch":
        if args.episodes <= 0 or args.pause < 0:
            parser.error("--episodes 必须为正整数，--pause 不能为负数")
        watch(args)
    else:
        if args.episodes <= 0:
            parser.error("--episodes 必须为正整数")
        evaluate_saved(args)


if __name__ == "__main__":
    main()
