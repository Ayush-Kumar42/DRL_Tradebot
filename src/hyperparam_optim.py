"""
hyperparam_search.py
--------------------
Grid search over PPO × hyperparameter grids × closing strategies.
Results are saved to results/search_results.json and a summary CSV.

- Fixed 20k training steps per config
- Max 100 configs sampled randomly from the grid
- Evaluation via mean reward over n_eval_episodes
- Each run gets its own log file under results/logs/

Usage
-----
    python hyperparam_search.py [--n-eval-episodes 3] [--max-rows 0]

Requirements
------------
    pip install stable-baselines3 gymnasium
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm.auto import tqdm

from indicators import CONTINUOUS_FEATURES, load_and_compute
from tradingenv import TradingEnv

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "BTCUSDT_1m.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Search config
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TIMESTEPS = 20_000
MAX_CONFIGS     = 100

# ─────────────────────────────────────────────────────────────────────────────
#  Indicator lists
# ─────────────────────────────────────────────────────────────────────────────

T_INDICATORS  = ["MACD", "MA", "OBV", "HA"]
MR_INDICATORS = ["RSI", "BBANDS", "STOCH", "CCI"]

# ─────────────────────────────────────────────────────────────────────────────
#  Closing strategies
# ─────────────────────────────────────────────────────────────────────────────

CLOSE_STRATEGIES = [
    "fifo",
    "max_profit",
    "least_loss",
    "age_weighted",
    "most_risk",
    "pnl_balanced",
]

# ─────────────────────────────────────────────────────────────────────────────
#  PPO hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS: dict[str, list] = {
    "learning_rate": [3e-4, 1e-4, 3e-5],
    "n_steps":       [512, 1024, 2048],
    "batch_size":    [64, 128],
    "gamma":         [0.99, 0.95],
    "ent_coef":      [0.0, 0.01],
}

# ─────────────────────────────────────────────────────────────────────────────
#  Per-run logger
# ─────────────────────────────────────────────────────────────────────────────

def make_run_logger(run_id: int, strategy: str, params: dict) -> logging.Logger:
    """Creates a file logger unique to this run. Returns it for use in run_single."""
    param_slug = "_".join(f"{k}{v}" for k, v in params.items())
    # Keep filenames filesystem-safe
    param_slug = param_slug.replace("e-", "e-").replace(".", "p")
    log_name   = f"run{run_id:03d}_{strategy}_{param_slug}"[:120]
    log_path   = os.path.join(LOGS_DIR, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    return logger, log_path

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    print(f"[search] Loading data from {csv_path} ...")
    df = load_and_compute(csv_path)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    print(f"[search] Data shape: {df.shape}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(df: pd.DataFrame, strategy: str) -> TradingEnv:
    return TradingEnv(
        df=df,
        T_indicators=T_INDICATORS,
        MR_indicators=MR_INDICATORS,
        continuous_features=CONTINUOUS_FEATURES,
        initial_balance=10_000.0,
        close_strategy=strategy,
        trail_pct=0.05,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Training progress callback
# ─────────────────────────────────────────────────────────────────────────────

class TqdmCallback(BaseCallback):
    def __init__(self, total: int, run_logger: logging.Logger):
        super().__init__()
        self.total      = total
        self.run_logger = run_logger
        self.pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total, desc="  Training", unit="step", leave=False)

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        # Log every 1000 steps to file
        if self.num_timesteps % 1_000 == 0:
            self.run_logger.debug(f"step={self.num_timesteps}")
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
        self.run_logger.info(f"Training complete. Total steps: {self.num_timesteps}")

# ─────────────────────────────────────────────────────────────────────────────
#  Step logger (diagnostic rollout)
# ─────────────────────────────────────────────────────────────────────────────

def _log_step(step: int, action, info: dict, trade_events: int, logger: logging.Logger) -> None:
    action_str = "[" + ", ".join(f"{a:.3f}" for a in np.asarray(action).flatten()) + "]"
    portfolio  = info.get("portfolio_value", float("nan"))
    price      = info.get("current_price",   float("nan"))
    changes    = info.get("position_changes") or {}
    n_open     = len(changes.get("opened", []))
    n_close    = len(changes.get("closed", []))
    msg = (
        f"step={step:>5d} | trade_events={trade_events:>2d}"
        f" | opened={n_open} closed={n_close}"
        f" | action={action_str}"
        f" | price={price:>10.2f}"
        f" | portfolio={portfolio:>10.2f}"
    )
    print(f"  [PPO] {msg}")
    logger.info(msg)

# ─────────────────────────────────────────────────────────────────────────────
#  Single-run evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    run_id: int,
    params: dict[str, Any],
    strategy: str,
    df: pd.DataFrame,
    n_eval_episodes: int,
) -> dict:
    logger, log_path = make_run_logger(run_id, strategy, params)

    logger.info(f"=== Run {run_id} ===")
    logger.info(f"Strategy : {strategy}")
    logger.info(f"Params   : {params}")

    result = {
        "run_id":        run_id,
        "algo":          "PPO",
        "strategy":      strategy,
        "params":        params,
        "mean_reward":   None,
        "std_reward":    None,
        "train_time_s":  None,
        "log_file":      log_path,
        "error":         None,
    }

    try:
        env      = make_env(df, strategy)
        eval_env = make_env(df, strategy)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            **params,
        )

        # ── Train for fixed 20k steps ────────────────────────────────────
        t0 = time.time()
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            callback=TqdmCallback(total=TRAIN_TIMESTEPS, run_logger=logger),
            progress_bar=False,
        )
        result["train_time_s"] = round(time.time() - t0, 2)
        logger.info(f"Train time: {result['train_time_s']}s")

        # ── Mean reward evaluation ───────────────────────────────────────
        with tqdm(total=n_eval_episodes, desc="  Evaluating", unit="ep", leave=False) as pbar:
            def _on_eval_episode_end(*args, **kwargs):
                pbar.update(1)

            mean_r, std_r = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                callback=_on_eval_episode_end,
            )

        result["mean_reward"] = round(float(mean_r), 4)
        result["std_reward"]  = round(float(std_r),  4)
        logger.info(f"mean_reward={result['mean_reward']}  std_reward={result['std_reward']}")

        # ── 5-trade-event diagnostic rollout ────────────────────────────
        logger.info("--- 5-trade diagnostic rollout ---")
        diag_env = make_env(df, strategy)
        obs, info = diag_env.reset()

        step         = 0
        trade_events = 0
        done         = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = diag_env.step(action)
            step += 1

            changes = info.get("position_changes")
            if changes:
                n_opened = len(changes.get("opened", []))
                n_closed = len(changes.get("closed", []))
                trade_events += n_opened + n_closed
                _log_step(step, action, info, trade_events, logger)

            if trade_events >= 5 or terminated or truncated:
                if diag_env.positions:
                    _, diag_env.positions = diag_env.bl.force_close_all(diag_env.positions)
                done = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(result["error"])
        print(f"  [ERROR] {result['error'][:200]}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  Grid expansion + random sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_configs(max_configs: int) -> list[tuple[dict, str]]:
    """
    Expand full grid × strategies, shuffle, return up to max_configs combos.
    """
    keys   = list(PARAM_GRIDS.keys())
    combos = list(itertools.product(*PARAM_GRIDS.values()))
    all_combos = [
        (dict(zip(keys, c)), strategy)
        for c in combos
        for strategy in CLOSE_STRATEGIES
    ]
    random.shuffle(all_combos)
    return all_combos[:max_configs]

# ─────────────────────────────────────────────────────────────────────────────
#  Main search loop
# ─────────────────────────────────────────────────────────────────────────────

def run_search(
    n_eval_episodes: int = 3,
    max_rows: int | None = None,
):
    df      = load_data(DATA_PATH, max_rows=max_rows)
    combos  = sample_configs(MAX_CONFIGS)
    total   = len(combos)

    print(f"\n[search] Total runs       : {total}  (max {MAX_CONFIGS})")
    print(f"  Algorithm             : PPO")
    print(f"  Train steps per run   : {TRAIN_TIMESTEPS:,}")
    print(f"  Strategies            : {CLOSE_STRATEGIES}")
    print(f"  Eval episodes         : {n_eval_episodes}")
    print(f"  Logs                  : {LOGS_DIR}\n")

    results: list[dict] = []

    for i, (params, strategy) in enumerate(combos, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())

        print(
            f"\n{'=' * 80}\n"
            f"Run {i}/{total}\n"
            f"Strategy  : {strategy}\n"
            f"Params    : {param_str}\n"
            f"{'=' * 80}"
        )

        res = run_single(i, params, strategy, df, n_eval_episodes)
        results.append(res)

        print(
            f"\nResult:"
            f"\n  Mean Reward  : {res['mean_reward']}"
            f"\n  Std Reward   : {res['std_reward']}"
            f"\n  Train Time   : {res['train_time_s']}s"
            f"\n  Log          : {res['log_file']}"
        )

        if res["error"]:
            print(f"\nERROR:\n{res['error']}")

        _save_results(results)

    # Print top 10 by mean reward at the end
    sorted_results = sorted(
        [r for r in results if r["mean_reward"] is not None],
        key=lambda r: r["mean_reward"],
        reverse=True,
    )
    print(f"\n{'=' * 80}")
    print("Top 10 configs by mean reward:")
    for rank, r in enumerate(sorted_results[:10], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {rank:>2}. [{r['strategy']:>12}] reward={r['mean_reward']:>9.4f}  {param_str}")

    print(f"\n[search] Done. Results saved to {RESULTS_DIR}/")
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(results: list[dict]):
    json_path = os.path.join(RESULTS_DIR, "search_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    rows = []
    for r in results:
        row = {
            "run_id":        r["run_id"],
            "algo":          r["algo"],
            "strategy":      r["strategy"],
            "mean_reward":   r["mean_reward"],
            "std_reward":    r["std_reward"],
            "train_time_s":  r["train_time_s"],
            "log_file":      r["log_file"],
            "error":         r["error"],
        }
        row.update(r.get("params", {}))
        rows.append(row)

    csv_path = os.path.join(RESULTS_DIR, "search_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows loaded from the dataset. 0 = full dataset.",
    )
    args = parser.parse_args()

    run_search(
        n_eval_episodes=args.n_eval_episodes,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )