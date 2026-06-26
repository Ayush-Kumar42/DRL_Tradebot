"""
optuna_search.py
----------------
Optuna-based hyperparameter optimisation for PPO × closing strategies.

Three-stage pipeline
---------------------
Stage 1  –  23 k steps  –  up to N_TRIALS trials  (Optuna exploration / pruning)
Stage 2  – 100 k steps  –  specific configs from stage 1 (refinement)
Stage 3  – 200 k steps  –  positive performers from stage 2 (deep refinement,
                            randomised-window evaluation)

Results are persisted after every trial:
  results/optuna/
    study.db                  – Optuna SQLite journal (resume-safe)
    best_model/               – SB3 model checkpoint of all-time best
    best_metrics.json         – detailed metrics for that model
    stage{1,2,3}_results.csv  – per-trial summaries per stage
    logs/                     – per-trial log files
    position_logs/            – per-run position-change CSVs (stages 2 & 3)

Usage
-----
    # Stage 1 (exploration)
    python optuna_search.py --stage 1 [--n-trials 60] [--max-rows 0]

    # Stage 2 (refinement, reads stage-1 results automatically)
    python optuna_search.py --stage 2 [--top-k 10]

    # Stage 3 (deep refinement, reads stage-2 results automatically)
    python optuna_search.py --stage 3

    # Run all stages sequentially
    python optuna_search.py --stage all [--n-trials 60] [--top-k 10]

Requirements
------------
    pip install stable-baselines3 gymnasium optuna
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import traceback
from typing import Any

import numpy as np
import optuna
from tqdm.auto import tqdm
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch import nn

from indicators import CONTINUOUS_FEATURES, load_and_compute
from tradingenv import TradingEnv

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "BTCUSDT_1m.csv")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "optuna")
LOGS_DIR      = os.path.join(RESULTS_DIR, "logs")
BEST_DIR      = os.path.join(RESULTS_DIR, "best_model")
POS_LOGS_DIR  = os.path.join(RESULTS_DIR, "position_logs")
DB_PATH       = f"sqlite:///{os.path.join(RESULTS_DIR, 'study.db')}"

for _d in (RESULTS_DIR, LOGS_DIR, BEST_DIR, POS_LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Stage configs
# ─────────────────────────────────────────────────────────────────────────────

STAGE_STEPS = {1: 23_000, 2: 100_000, 3: 200_000}
N_EVAL_EPISODES = 5

# Stage 2: specific trial indexes to refine from stage 1 results
STAGE2_TRIAL_INDEXES = [28, 4, 37, 32, 3]

# Stage 3: number of eval episodes per checkpoint and window size for
# randomised evaluation.  More episodes → tighter reward estimate across
# diverse market regimes.
N_EVAL_EPISODES_S3  = 15
EVAL_WINDOW_SIZE_S3 = 5_000   # rows per randomised eval episode
EVAL_FREQ_S3        = 20_000  # steps between evaluation checkpoints

# ─────────────────────────────────────────────────────────────────────────────
#  Indicator / strategy lists
# ─────────────────────────────────────────────────────────────────────────────

T_INDICATORS  = ["MACD", "MA", "OBV", "HA"]
MR_INDICATORS = ["RSI", "BBANDS", "STOCH", "CCI"]

CLOSE_STRATEGIES = [
    "fifo",
    "max_profit",
    "least_loss",
    "age_weighted",
    "most_risk",
    "pnl_balanced",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Activation map
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATIONS: dict[str, type] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_run_logger(name: str) -> tuple[logging.Logger, str]:
    log_path = os.path.join(LOGS_DIR, f"{name}.log")
    logger   = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger, log_path


def load_data(max_rows: int | None = None) -> pd.DataFrame:
    print(f"[optuna] Loading data from {DATA_PATH} …")
    df = load_and_compute(DATA_PATH)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    print(f"[optuna] Data shape: {df.shape}")
    return df


def make_env(df: pd.DataFrame, strategy: str) -> Monitor:
    env = TradingEnv(
        df=df,
        T_indicators=T_INDICATORS,
        MR_indicators=MR_INDICATORS,
        continuous_features=CONTINUOUS_FEATURES,
        initial_balance=10_000.0,
        close_strategy=strategy,
        trail_pct=0.05,
    )
    return Monitor(env)


def make_random_window_eval_env(df: pd.DataFrame, strategy: str) -> Monitor:
    """
    Eval env for stage 3: uses the full dataset but picks a fresh random
    window of EVAL_WINDOW_SIZE_S3 rows on every reset().  Training env
    (make_env) is unchanged — it always sees the full df.
    """
    env = TradingEnv(
        df=df,
        T_indicators=T_INDICATORS,
        MR_indicators=MR_INDICATORS,
        continuous_features=CONTINUOUS_FEATURES,
        initial_balance=10_000.0,
        close_strategy=strategy,
        trail_pct=0.05,
        random_start=True,
        window_size=EVAL_WINDOW_SIZE_S3,
    )
    return Monitor(env)

# ─────────────────────────────────────────────────────────────────────────────
#  Position-change logger (stages 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────

_POS_CSV_FIELDS = [
    "step",
    "action_0", "action_1", "action_2", "action_3", "action_4",
    "action_vol",
    "change_type",
    "pos_order",
    "pos_price",
    "pos_volume",
    "pos_signal",
]


class PositionLogCallback(BaseCallback):
    """
    Writes a CSV row for every open position whenever the position list
    changes between two consecutive steps.  Used in stages 2 and 3.
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self._csv_path = csv_path
        self._prev_positions: list[dict] | None = None
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._file   = open(self._csv_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=_POS_CSV_FIELDS)
        self._writer.writeheader()
        self._prev_positions = None

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()
            self._file   = None
            self._writer = None

    def _on_step(self) -> bool:
        try:
            inner_env: TradingEnv = self.training_env.envs[0].unwrapped
        except (AttributeError, IndexError):
            return True

        current_step = getattr(inner_env, "current_step", None)
        positions    = getattr(inner_env, "positions", [])
        current_snapshot = [(p.order, p.volume) for p in positions]

        if self._prev_positions is not None and current_snapshot != self._prev_positions:
            prev_orders    = {o for o, _ in self._prev_positions}
            current_orders = {p.order for p in positions}
            new_orders     = current_orders - prev_orders
            removed_orders = prev_orders    - current_orders
            prev_vol       = {o: v for o, v in self._prev_positions}
            volume_changed = any(
                p.volume != prev_vol.get(p.order, p.volume)
                for p in positions
                if p.order in prev_orders
            )

            if new_orders and not removed_orders and not volume_changed:
                change_type = "opened"
            elif removed_orders and not new_orders and not volume_changed:
                change_type = "closed"
            else:
                change_type = "modified"

            action = self.locals.get("actions")
            if action is not None:
                action = action[0]
            else:
                action = []

            self._write_rows(
                step=current_step,
                positions=positions,
                action=action,
                change_type=change_type,
            )

        self._prev_positions = current_snapshot
        return True

    def _write_rows(self, step, positions, action, change_type) -> None:
        act = list(action) if action is not None and len(action) else []
        def _a(i):
            return round(float(act[i]), 6) if i < len(act) else None

        for p in positions:
            row = {
                "step":        step,
                "action_0":    _a(0),
                "action_1":    _a(1),
                "action_2":    _a(2),
                "action_3":    _a(3),
                "action_4":    _a(4),
                "action_vol":  _a(4),
                "change_type": change_type,
                "pos_order":   p.order,
                "pos_price":   round(p.price,  6),
                "pos_volume":  round(p.volume, 8),
                "pos_signal":  p.signal,
            }
            self._writer.writerow(row)

# ─────────────────────────────────────────────────────────────────────────────
#  Metric-collecting callback
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """
    Collects per-update training metrics from SB3's logger.
    Prints a one-line summary to console after every rollout.
    Optionally reports intermediate rewards to Optuna for pruning.
    """
    def __init__(
        self,
        trial: optuna.Trial | None = None,
        eval_env=None,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 3,
        total_steps: int = 0,
        trial_label: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial           = trial
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.total_steps     = total_steps
        self.trial_label     = trial_label

        self.policy_losses:  list[float] = []
        self.value_losses:   list[float] = []
        self.entropies:      list[float] = []
        self.approx_kls:     list[float] = []
        self.clip_fracs:     list[float] = []
        self._last_eval_step = 0
        self._rollout_count  = 0
        self._train_start    = time.time()

    def _on_step(self) -> bool:
        if (
            self.trial is not None
            and self.eval_env is not None
            and (self.num_timesteps - self._last_eval_step) >= self.eval_freq
        ):
            self._last_eval_step = self.num_timesteps
            mean_r, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                warn=False,
            )
            self.trial.report(float(mean_r), step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return True

    def _on_rollout_end(self) -> None:
        logs = self.model.logger.name_to_value
        for key, store in [
            ("train/policy_gradient_loss", self.policy_losses),
            ("train/value_loss",           self.value_losses),
            ("train/entropy_loss",         self.entropies),
            ("train/approx_kl",            self.approx_kls),
            ("train/clip_fraction",        self.clip_fracs),
        ]:
            val = logs.get(key)
            if val is not None:
                store.append(float(val))

        self._rollout_count += 1
        elapsed   = time.time() - self._train_start
        pct       = (self.num_timesteps / self.total_steps * 100) if self.total_steps else 0
        steps_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
        eta_s     = (self.total_steps - self.num_timesteps) / steps_sec if steps_sec > 0 else 0

        pol  = f"{self.policy_losses[-1]:.4f}"  if self.policy_losses  else "n/a"
        val  = f"{self.value_losses[-1]:.4f}"   if self.value_losses   else "n/a"
        ent  = f"{self.entropies[-1]:.4f}"      if self.entropies      else "n/a"
        kl   = f"{self.approx_kls[-1]:.5f}"     if self.approx_kls     else "n/a"

        print(
            f"  [{self.trial_label}] "
            f"step {self.num_timesteps:>7,}/{self.total_steps:,} ({pct:5.1f}%)  "
            f"pol={pol}  val={val}  ent={ent}  kl={kl}  "
            f"elapsed={elapsed:.0f}s  ETA={eta_s:.0f}s  "
            f"({steps_sec:.0f} steps/s)",
            flush=True,
        )

    def summary(self) -> dict[str, float | None]:
        def _mean(lst):
            return round(float(np.mean(lst)), 6) if lst else None

        return {
            "mean_policy_loss":  _mean(self.policy_losses),
            "mean_value_loss":   _mean(self.value_losses),
            "mean_entropy":      _mean(self.entropies),
            "mean_approx_kl":    _mean(self.approx_kls),
            "mean_clip_frac":    _mean(self.clip_fracs),
            "final_policy_loss": round(self.policy_losses[-1],  6) if self.policy_losses  else None,
            "final_value_loss":  round(self.value_losses[-1],   6) if self.value_losses   else None,
            "final_entropy":     round(self.entropies[-1],      6) if self.entropies      else None,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  Stage-3 multi-window evaluation callback
# ─────────────────────────────────────────────────────────────────────────────

class RandomWindowEvalCallback(BaseCallback):
    """
    Periodic evaluation callback for stage 3.

    Every `eval_freq` steps it runs `n_eval_episodes` full episodes on the
    supplied eval env (which must be a random-window env so each episode
    sees a different market segment).  Each episode is seeded from
    `num_timesteps + episode_index` so windows are:
      - varied     : different episodes within one checkpoint see different data
      - reproducible: re-running the same checkpoint always picks the same windows
      - non-repeating across checkpoints: the seed shifts with num_timesteps

    Intermediate mean rewards are printed to console alongside the training
    rollout summaries produced by MetricsCallback.  The full history of
    checkpoint evaluations is stored in `eval_history` for post-run
    analysis.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = EVAL_FREQ_S3,
        n_eval_episodes: int = N_EVAL_EPISODES_S3,
        trial_label: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.trial_label     = trial_label
        self._last_eval_step = 0
        self.eval_history: list[dict] = []

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self) -> None:
        rewards   = []
        base_seed = self.num_timesteps  # shifts every checkpoint

        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=base_seed + ep)
            done      = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)

        mean_r = float(np.mean(rewards))
        std_r  = float(np.std(rewards))

        self.eval_history.append({
            "timestep":    self.num_timesteps,
            "mean_reward": round(mean_r, 4),
            "std_reward":  round(std_r,  4),
            "n_episodes":  self.n_eval_episodes,
        })

        print(
            f"  [{self.trial_label}] eval @ {self.num_timesteps:,} steps  "
            f"reward={mean_r:.4f} ± {std_r:.4f}"
            f"  ({self.n_eval_episodes} random windows)",
            flush=True,
        )

    def best_mean_reward(self) -> float | None:
        if not self.eval_history:
            return None
        return max(r["mean_reward"] for r in self.eval_history)

# ─────────────────────────────────────────────────────────────────────────────
#  Build PPO model from trial / fixed params
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    env,
    params: dict[str, Any],
    seed: int = 42,
) -> PPO:
    net_arch_key = params["net_arch"]
    arch_map = {
        "small":  [128, 128],
        "medium": [128, 128, 128],
        "large":  [256, 256],
        "xlarge": [512, 512],
    }
    net_arch = arch_map[net_arch_key]
    activation_fn = ACTIVATIONS[params["activation_fn"]]

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
    )

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [5e-5, 1e-4, 3e-4, 1e-3]
        ),
        "n_steps":       trial.suggest_categorical("n_steps", [1024]),
        "batch_size":    trial.suggest_categorical("batch_size", [256]),
        "n_epochs":      trial.suggest_categorical("n_epochs", [10]),
        "gamma":         trial.suggest_categorical(
            "gamma", [0.95, 0.97, 0.99, 0.995, 0.999]
        ),
        "gae_lambda":    trial.suggest_categorical(
            "gae_lambda", [0.85, 0.90, 0.95, 0.97, 0.99, 0.995]
        ),
        "clip_range":    trial.suggest_categorical(
            "clip_range", [0.1, 0.15, 0.2, 0.3, 0.4]
        ),
        "ent_coef":      trial.suggest_categorical(
            "ent_coef", [0.0, 0.001, 0.005, 0.01, 0.02]
        ),
        "vf_coef":       trial.suggest_categorical(
            "vf_coef", [0.25, 0.5, 0.75, 1.0]
        ),
        "max_grad_norm": trial.suggest_categorical(
            "max_grad_norm", [0.5, 0.7, 1.0]
        ),
        "net_arch":      trial.suggest_categorical(
            "net_arch", ["small", "medium", "large", "xlarge"]
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", ["relu", "tanh", "elu"]
        ),
        "close_strategy": trial.suggest_categorical(
            "close_strategy", CLOSE_STRATEGIES
        ),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Global best tracker (persisted to disk)
# ─────────────────────────────────────────────────────────────────────────────

_BEST_METRICS_PATH = os.path.join(BEST_DIR, "best_metrics.json")
_best_reward: float = float("-inf")


def _load_best_reward() -> float:
    if os.path.exists(_BEST_METRICS_PATH):
        with open(_BEST_METRICS_PATH) as f:
            data = json.load(f)
        return float(data.get("mean_reward", float("-inf")))
    return float("-inf")


def maybe_save_best(
    model: PPO,
    mean_reward: float,
    metrics: dict,
    params: dict,
    stage: int,
    trial_id: int,
) -> bool:
    global _best_reward
    if mean_reward <= _best_reward:
        return False

    prev_best    = _best_reward
    _best_reward = mean_reward
    model.save(os.path.join(BEST_DIR, "model"))

    full_metrics = {
        "mean_reward":   round(mean_reward, 4),
        "stage":         stage,
        "trial_id":      trial_id,
        "params":        params,
        **metrics,
    }
    with open(_BEST_METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)

    print(f"\n  ★ NEW BEST  reward={mean_reward:.4f}  (prev best={prev_best:.4f})")
    print(f"    Saved to {BEST_DIR}/")
    return True

# ─────────────────────────────────────────────────────────────────────────────
#  Single trial
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    trial: optuna.Trial | None,
    params: dict[str, Any],
    df: pd.DataFrame,
    train_steps: int,
    stage: int,
    trial_id: int,
    enable_pruning: bool = False,
    pos_log_path: str | None = None,
    # Stage-3 specific: supply a random-window eval env to activate the
    # richer multi-window evaluation path; None falls back to the standard path.
    rw_eval_env=None,
) -> dict:
    """
    Train one PPO config and return a result dict with all metrics.

    When `rw_eval_env` is provided (stage 3), a RandomWindowEvalCallback
    drives periodic evaluation over randomised data windows in addition to
    the MetricsCallback rollout summaries.  The final reward reported is
    the mean over N_EVAL_EPISODES_S3 random windows.

    When `pos_log_path` is provided, a PositionLogCallback writes a CSV
    for every position-list change during training.
    """
    strategy  = params["close_strategy"]
    run_name  = f"s{stage}_t{trial_id:04d}_{strategy}"
    logger, log_path = make_run_logger(run_name)

    logger.info(f"Stage {stage} | trial {trial_id} | strategy={strategy}")
    logger.info(f"Params: {params}")

    _sep = "─" * 80
    print(f"\n{_sep}")
    print(
        f"  Stage {stage} | Trial {trial_id}"
        + (f"  [Optuna #{trial.number}]" if trial is not None else "")
    )
    print(f"  Strategy : {strategy}")
    arch_str = params.get("net_arch", "?")
    act_str  = params.get("activation_fn", "?")
    print(
        f"  lr={params['learning_rate']}  gamma={params['gamma']}  "
        f"gae={params['gae_lambda']}  clip={params['clip_range']}  "
        f"ent={params['ent_coef']}  vf={params['vf_coef']}  "
        f"grad={params['max_grad_norm']}  arch={arch_str}  act={act_str}"
    )
    print(f"  Training for {train_steps:,} steps …")
    if pos_log_path:
        print(f"  Position log → {pos_log_path}")
    if rw_eval_env is not None:
        print(
            f"  Eval: {N_EVAL_EPISODES_S3} random-window episodes "
            f"(window={EVAL_WINDOW_SIZE_S3:,} rows)  every {EVAL_FREQ_S3:,} steps"
        )
    print(_sep, flush=True)

    result: dict[str, Any] = {
        "trial_id":      trial_id,
        "stage":         stage,
        "mean_reward":   None,
        "std_reward":    None,
        "train_time_s":  None,
        "log_file":      log_path,
        "error":         None,
        **params,
    }

    try:
        env = make_env(df, strategy)

        # Standard fixed-window eval env (used by MetricsCallback for
        # rollout-level pruning in stage 1; unused in stage 3 where
        # rw_eval_env takes over).
        eval_df  = df.tail(min(5000, len(df))).reset_index(drop=True)
        eval_env = make_env(eval_df, strategy)

        model = build_model(env, params)

        trial_label = f"s{stage}/t{trial_id}/{strategy}"

        # MetricsCallback: rollout summaries + optional Optuna pruning.
        # In stage 3 we don't use it for pruning (no trial object), but
        # we keep it for the console rollout-level summaries.
        _eval_eps = 1 if stage == 1 else 3
        metrics_cb = MetricsCallback(
            trial=trial if enable_pruning else None,
            eval_env=eval_env,
            eval_freq=10_000,
            n_eval_episodes=_eval_eps,
            total_steps=train_steps,
            trial_label=trial_label,
        )

        callbacks = [metrics_cb]

        # Stage 3: add the random-window eval callback
        rw_eval_cb: RandomWindowEvalCallback | None = None
        if rw_eval_env is not None:
            rw_eval_cb = RandomWindowEvalCallback(
                eval_env=rw_eval_env,
                eval_freq=EVAL_FREQ_S3,
                n_eval_episodes=N_EVAL_EPISODES_S3,
                trial_label=trial_label,
            )
            callbacks.append(rw_eval_cb)

        # Position logger (stages 2 & 3)
        if pos_log_path is not None:
            callbacks.append(PositionLogCallback(csv_path=pos_log_path))

        t0 = time.time()
        model.learn(
            total_timesteps=train_steps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        result["train_time_s"] = round(time.time() - t0, 2)

        # ── Final evaluation ─────────────────────────────────────────────
        if rw_eval_cb is not None:
            # Stage 3: one final multi-window eval pass after training
            # completes (same mechanism as the periodic checkpoints).
            print(
                f"  Final eval ({N_EVAL_EPISODES_S3} random-window episodes) …",
                flush=True,
            )
            rewards   = []
            base_seed = train_steps  # distinct from any checkpoint seed
            for ep in range(N_EVAL_EPISODES_S3):
                obs, _ = rw_eval_env.reset(seed=base_seed + ep)
                done      = False
                ep_reward = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = rw_eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                rewards.append(ep_reward)
            mean_r = float(np.mean(rewards))
            std_r  = float(np.std(rewards))

            # Also store the full checkpoint history in the result for
            # post-run analysis (JSON-serialisable list of dicts).
            result["eval_history"] = rw_eval_cb.eval_history
        else:
            # Stages 1 & 2: standard evaluate_policy on the fixed tail slice
            _n_eval = 1 if stage == 1 else N_EVAL_EPISODES
            print(
                f"  Evaluating ({_n_eval} episode{'s' if _n_eval > 1 else ''}) …",
                flush=True,
            )
            mean_r, std_r = evaluate_policy(
                model, eval_env,
                n_eval_episodes=_n_eval,
                deterministic=True,
                warn=False,
            )
            mean_r = float(mean_r)
            std_r  = float(std_r)

        result["mean_reward"] = round(mean_r, 4)
        result["std_reward"]  = round(std_r,  4)

        metric_summary = metrics_cb.summary()
        result.update(metric_summary)

        logger.info(
            f"mean_reward={result['mean_reward']}  "
            f"std_reward={result['std_reward']}  "
            f"train_time={result['train_time_s']}s"
        )
        logger.info(f"Training metrics: {metric_summary}")

        print(
            f"  ✓ done  reward={result['mean_reward']:.4f} ± {result['std_reward']:.4f}"
            f"  |  pol={metric_summary.get('final_policy_loss')}"
            f"  val={metric_summary.get('final_value_loss')}"
            f"  ent={metric_summary.get('final_entropy')}"
            f"  |  {result['train_time_s']}s total",
            flush=True,
        )

        maybe_save_best(model, mean_r, metric_summary, params, stage, trial_id)

    except optuna.exceptions.TrialPruned:
        result["error"] = "PRUNED"
        logger.info("Trial pruned by Optuna")
        print("  ✗ PRUNED by Optuna (underperforming at intermediate check)", flush=True)
        raise

    except Exception as e:
        tb  = traceback.format_exc()
        result["error"] = f"{type(e).__name__}: {e}\n{tb}"
        logger.error(result["error"])
        print(f"  [ERROR] {str(e)[:200]}")

    return result


def _load_existing_results(stage: int) -> list[dict]:
    csv_path = os.path.join(RESULTS_DIR, f"stage{stage}_results.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        df_existing = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []
    df_existing = df_existing.where(pd.notnull(df_existing), None)
    return df_existing.to_dict(orient="records")


def _merge_results(existing: list[dict], new_row: dict) -> list[dict]:
    by_id = {r["trial_id"]: r for r in existing}
    by_id[new_row["trial_id"]] = new_row
    ordered_ids = [r["trial_id"] for r in existing]
    for tid in by_id:
        if tid not in ordered_ids:
            ordered_ids.append(tid)
    return [by_id[tid] for tid in ordered_ids]

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 – Optuna exploration
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(df: pd.DataFrame, n_trials: int = 60) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"STAGE 1  –  Optuna exploration  ({STAGE_STEPS[1]:,} steps × {n_trials} trials)")
    print(f"{'='*80}\n")

    global _best_reward
    _best_reward = _load_best_reward()

    study = optuna.create_study(
        study_name="ppo_stage1",
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=DB_PATH,
        load_if_exists=True,
    )

    results: list[dict] = _load_existing_results(stage=1)
    if results:
        print(f"[optuna] Resuming stage 1 — loaded {len(results)} previous result(s)")

    stage1_start = time.time()

    def objective(trial: optuna.Trial) -> float:
        completed = len([t for t in study.trials if t.state.is_finished()])
        elapsed   = time.time() - stage1_start
        avg_s     = elapsed / completed if completed else 0
        eta_s     = avg_s * (n_trials - completed) if completed else 0
        print(
            f"\n{'='*80}\n"
            f"  STAGE 1 — Trial {trial.number + 1}/{n_trials}"
            f"  |  completed={completed}"
            f"  |  elapsed={elapsed:.0f}s"
            + (f"  |  avg/trial={avg_s:.0f}s  ETA={eta_s:.0f}s" if completed else "")
            + (
                f"  |  best so far={max((t.value for t in study.trials if t.value is not None), default=float('nan')):.4f}"
                if any(t.value is not None for t in study.trials) else ""
            )
            + f"\n{'='*80}",
            flush=True,
        )
        params = suggest_params(trial)
        res    = run_trial(
            trial=trial,
            params=params,
            df=df,
            train_steps=STAGE_STEPS[1],
            stage=1,
            trial_id=trial.number,
            enable_pruning=True,
        )
        nonlocal results
        results = _merge_results(results, res)
        _save_stage_results(results, stage=1)

        reward = res.get("mean_reward")
        if reward is None:
            raise optuna.exceptions.TrialPruned()
        return reward

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    df_results = _save_stage_results(results, stage=1)
    _print_top(df_results, n=10, stage=1)
    return df_results

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 – Refinement
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Runs refinement on the specific trial indexes defined in STAGE2_TRIAL_INDEXES.
    """
    selected_indexes = STAGE2_TRIAL_INDEXES
    n_configs = len(selected_indexes)

    print(f"\n{'='*80}")
    print(f"STAGE 2  –  Refinement  ({STAGE_STEPS[2]:,} steps × {n_configs} configs)")
    print(f"  Trial indexes: {selected_indexes}")
    print(f"  Position logs → {POS_LOGS_DIR}/")
    print(f"{'='*80}\n")

    global _best_reward
    _best_reward = _load_best_reward()

    csv_path = os.path.join(RESULTS_DIR, "stage1_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Stage 1 results not found at {csv_path}. Run stage 1 first."
        )

    s1 = pd.read_csv(csv_path)
    selected = s1[s1["trial_id"].isin(selected_indexes)].copy()

    missing = set(selected_indexes) - set(selected["trial_id"].tolist())
    if missing:
        raise ValueError(
            f"Trial indexes not found in stage1_results.csv: {sorted(missing)}\n"
            f"Available trial_ids: {sorted(s1['trial_id'].tolist())}"
        )

    selected["_sort_order"] = selected["trial_id"].map(
        {tid: i for i, tid in enumerate(selected_indexes)}
    )
    selected = selected.sort_values("_sort_order").drop(columns="_sort_order")

    param_cols = [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
        "max_grad_norm", "net_arch", "activation_fn", "close_strategy",
    ]

    results: list[dict] = _load_existing_results(stage=2)
    if results:
        print(f"[optuna] Resuming stage 2 — loaded {len(results)} previous result(s)")

    completed_ids = {r["trial_id"] for r in results if r.get("mean_reward") is not None}

    stage2_start = time.time()
    for i, row in enumerate(selected.itertuples(), 1):
        if row.trial_id in completed_ids:
            print(
                f"\n  STAGE 2 — Config {i}/{n_configs}"
                f"  |  stage1_trial_id={row.trial_id}  [already completed, skipping]",
                flush=True,
            )
            continue

        params   = {c: getattr(row, c) for c in param_cols}
        strategy = params["close_strategy"]
        elapsed  = time.time() - stage2_start
        done     = i - 1
        avg_s    = elapsed / done if done else 0
        eta_s    = avg_s * (n_configs - done) if done else 0
        print(
            f"\n{'='*80}\n"
            f"  STAGE 2 — Config {i}/{n_configs}"
            f"  |  stage1_trial_id={row.trial_id}"
            f"  |  stage1_reward={row.mean_reward:.4f}"
            f"  |  elapsed={elapsed:.0f}s"
            + (f"  |  avg/config={avg_s:.0f}s  ETA={eta_s:.0f}s" if done else "")
            + f"\n{'='*80}",
            flush=True,
        )

        pos_log_path = os.path.join(
            POS_LOGS_DIR, f"s2_t{row.trial_id:04d}_{strategy}.csv"
        )

        res = run_trial(
            trial=None,
            params=params,
            df=df,
            train_steps=STAGE_STEPS[2],
            stage=2,
            trial_id=row.trial_id,
            enable_pruning=False,
            pos_log_path=pos_log_path,
        )
        results = _merge_results(results, res)
        _save_stage_results(results, stage=2)

    df_results = _save_stage_results(results, stage=2)
    _print_top(df_results, n=5, stage=2)
    return df_results

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 – Deep refinement (positive stage-2 performers only)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-trains only the stage-2 configs whose mean_reward > 0, using:
      - 200 k training steps (double stage 2)
      - 15 evaluation episodes per checkpoint, each on a fresh random
        5 000-row window drawn from the full dataset
      - Periodic checkpoints every 20 k steps so you can watch
        generalisation improve (or diverge) over time
      - Position-change CSVs (same format as stage 2)
      - Resume/skip logic: already-completed trial_ids are skipped

    Outputs
    -------
    results/optuna/
        stage3_results.csv              – per-trial summary
        position_logs/s3_t*_*.csv       – position-change logs
        logs/s3_t*_*.log                – per-trial log files
        best_model/                     – updated if a new all-time best is set
    """
    stage = 3

    # ── Load stage 2, keep only positive performers ───────────────────────
    s2_csv = os.path.join(RESULTS_DIR, "stage2_results.csv")
    if not os.path.exists(s2_csv):
        raise FileNotFoundError(
            f"Stage 2 results not found at {s2_csv}. Run stage 2 first."
        )

    s2       = pd.read_csv(s2_csv)
    positive = s2[s2["mean_reward"] > 0].copy()

    if positive.empty:
        print(
            "[stage 3] No positive performers found in stage2_results.csv. "
            "Nothing to do."
        )
        return positive

    # Sort descending so the most promising config runs first.
    positive = positive.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    n_configs = len(positive)

    print(f"\n{'='*80}")
    print(
        f"STAGE 3  –  Deep refinement  "
        f"({STAGE_STEPS[3]:,} steps × {n_configs} config{'s' if n_configs != 1 else ''})"
    )
    print(f"  Positive stage-2 performers (mean_reward > 0):")
    for _, r in positive.iterrows():
        print(
            f"    trial_id={int(r['trial_id'])}  "
            f"reward={r['mean_reward']:.4f}  "
            f"strategy={r['close_strategy']}"
        )
    print(
        f"  Eval: {N_EVAL_EPISODES_S3} random-window episodes "
        f"(window={EVAL_WINDOW_SIZE_S3:,} rows)  every {EVAL_FREQ_S3:,} steps"
    )
    print(f"  Position logs → {POS_LOGS_DIR}/")
    print(f"{'='*80}\n")

    global _best_reward
    _best_reward = _load_best_reward()

    param_cols = [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
        "max_grad_norm", "net_arch", "activation_fn", "close_strategy",
    ]

    results: list[dict] = _load_existing_results(stage=stage)
    if results:
        print(
            f"[optuna] Resuming stage 3 — loaded {len(results)} previous result(s) "
            f"from {os.path.join(RESULTS_DIR, 'stage3_results.csv')}"
        )

    # trial_ids whose result row already has a valid reward → skip them
    completed_ids = {
        r["trial_id"] for r in results if r.get("mean_reward") is not None
    }

    stage3_start = time.time()

    for i, row in enumerate(positive.itertuples(), 1):
        trial_id = int(row.trial_id)
        strategy = row.close_strategy

        if trial_id in completed_ids:
            print(
                f"\n  STAGE 3 — Config {i}/{n_configs}"
                f"  |  trial_id={trial_id}  [already completed, skipping]",
                flush=True,
            )
            continue

        params  = {c: getattr(row, c) for c in param_cols}
        elapsed = time.time() - stage3_start
        done    = i - 1
        avg_s   = elapsed / done if done else 0
        eta_s   = avg_s * (n_configs - done) if done else 0

        print(
            f"\n{'='*80}\n"
            f"  STAGE 3 — Config {i}/{n_configs}"
            f"  |  stage2_trial_id={trial_id}"
            f"  |  stage2_reward={row.mean_reward:.4f}"
            f"  |  strategy={strategy}"
            f"  |  elapsed={elapsed:.0f}s"
            + (f"  |  avg/config={avg_s:.0f}s  ETA={eta_s:.0f}s" if done else "")
            + f"\n{'='*80}",
            flush=True,
        )

        # Random-window eval env: full df passed in; TradingEnv picks a
        # fresh random slice of EVAL_WINDOW_SIZE_S3 rows each reset().
        rw_eval_env = make_random_window_eval_env(df, strategy)

        pos_log_path = os.path.join(
            POS_LOGS_DIR, f"s3_t{trial_id:04d}_{strategy}.csv"
        )

        res = run_trial(
            trial=None,
            params=params,
            df=df,
            train_steps=STAGE_STEPS[3],
            stage=stage,
            trial_id=trial_id,
            enable_pruning=False,
            pos_log_path=pos_log_path,
            rw_eval_env=rw_eval_env,
        )

        results = _merge_results(results, res)
        _save_stage_results(results, stage=stage)

    df_results = _save_stage_results(results, stage=stage)
    _print_top(df_results, n=n_configs, stage=stage)
    return df_results

# ─────────────────────────────────────────────────────────────────────────────
#  Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_stage_results(results: list[dict], stage: int) -> pd.DataFrame:
    csv_path = os.path.join(RESULTS_DIR, f"stage{stage}_results.csv")
    df_out   = pd.DataFrame(results)
    df_out.to_csv(csv_path, index=False)
    return df_out


def _print_top(df: pd.DataFrame, n: int, stage: int) -> None:
    valid = df[df["mean_reward"].notna()].sort_values("mean_reward", ascending=False)
    print(f"\nTop {n} configs – Stage {stage}:")
    for rank, row in enumerate(valid.head(n).itertuples(), 1):
        print(
            f"  {rank:>2}.  reward={row.mean_reward:>9.4f}  "
            f"strategy={row.close_strategy:<14}  "
            f"lr={row.learning_rate}  gamma={row.gamma}  "
            f"gae={row.gae_lambda}  clip={row.clip_range}  "
            f"arch={row.net_arch}  act={row.activation_fn}"
        )

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna PPO hyperparam search")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "both", "all"],
        default="1",
        help=(
            "Which stage to run. "
            "'both' runs 1→2, 'all' runs 1→2→3 sequentially."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=60, help="Stage 1 trials.")
    parser.add_argument("--top-k",    type=int, default=10, help="Unused in stage 2 (indexes are hardcoded).")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows loaded from dataset. 0 = full dataset.",
    )
    args = parser.parse_args()

    df = load_data(max_rows=args.max_rows if args.max_rows > 0 else None)

    if args.stage in ("1", "both", "all"):
        run_stage1(df, n_trials=args.n_trials)
    if args.stage in ("2", "both", "all"):
        run_stage2(df, top_k=args.top_k)
    if args.stage in ("3", "all"):
        run_stage3(df)