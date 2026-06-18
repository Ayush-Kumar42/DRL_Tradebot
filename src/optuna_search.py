"""
optuna_search.py
----------------
Optuna-based hyperparameter optimisation for PPO × closing strategies.

Two-stage pipeline
-------------------
Stage 1  –  40 k steps  –  up to N_TRIALS trials  (Optuna exploration / pruning)
Stage 2  – 400 k steps  –  top-K configs from stage 1 (refinement)

Results are persisted after every trial:
  results/optuna/
    study.db                  – Optuna SQLite journal (resume-safe)
    best_model/               – SB3 model checkpoint of all-time best
    best_metrics.json         – detailed metrics for that model
    stage{1,2}_results.csv    – per-trial summaries per stage
    logs/                     – per-trial log files

Usage
-----
    # Stage 1 (exploration)
    python optuna_search.py --stage 1 [--n-trials 60] [--max-rows 0]

    # Stage 2 (refinement, reads stage-1 results automatically)
    python optuna_search.py --stage 2 [--top-k 10]

    # Run both stages sequentially
    python optuna_search.py --stage both [--n-trials 60] [--top-k 10]

Requirements
------------
    pip install stable-baselines3 gymnasium optuna
"""

from __future__ import annotations

import argparse
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

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "BTCUSDT_1m.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "optuna")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
BEST_DIR    = os.path.join(RESULTS_DIR, "best_model")
DB_PATH     = f"sqlite:///{os.path.join(RESULTS_DIR, 'study.db')}"

for _d in (RESULTS_DIR, LOGS_DIR, BEST_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Stage configs
# ─────────────────────────────────────────────────────────────────────────────

STAGE_STEPS = {1: 23_000, 2: 400_000}
N_EVAL_EPISODES = 5

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

# ─────────────────────────────────────────────────────────────────────────────
#  Metric-collecting callback (policy loss, value loss, entropy, approx KL)
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

        # Accumulated metric lists
        self.policy_losses:  list[float] = []
        self.value_losses:   list[float] = []
        self.entropies:      list[float] = []
        self.approx_kls:     list[float] = []
        self.clip_fracs:     list[float] = []
        self._last_eval_step = 0
        self._rollout_count  = 0
        self._train_start    = time.time()

    def _on_step(self) -> bool:
        # Intermediate eval for Optuna pruning
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
        """Pull scalar metrics and print a live console summary."""
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
    """Ask Optuna to suggest a full hyperparameter set."""
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
    """Save model + metrics if this is the best reward seen so far."""
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
#  Single trial (used by Optuna objective)
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    trial: optuna.Trial | None,
    params: dict[str, Any],
    df: pd.DataFrame,
    train_steps: int,
    stage: int,
    trial_id: int,
    enable_pruning: bool = False,
) -> dict:
    """
    Train one PPO config and return a result dict with all metrics.
    `trial` may be None when running fixed configs in stage 2.
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
    arch_str = params.get('net_arch', '?')
    act_str  = params.get('activation_fn', '?')
    print(
        f"  lr={params['learning_rate']}  gamma={params['gamma']}  "
        f"gae={params['gae_lambda']}  clip={params['clip_range']}  "
        f"ent={params['ent_coef']}  vf={params['vf_coef']}  "
        f"grad={params['max_grad_norm']}  arch={arch_str}  act={act_str}"
    )
    print(f"  Training for {train_steps:,} steps …")
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
        env      = make_env(df, strategy)
        eval_df  = df.tail(min(5000, len(df))).reset_index(drop=True)
        eval_env = make_env(eval_df, strategy)

        model = build_model(env, params)

        trial_label = f"s{stage}/t{trial_id}" + (f"/{strategy}" if strategy else "")
        # Stage 1: 1 eval episode per check (full df per episode, keep it fast)
        _eval_eps = 1 if stage == 1 else 3
        metrics_cb = MetricsCallback(
            trial=trial if enable_pruning else None,
            eval_env=eval_env,
            eval_freq=10_000,
            n_eval_episodes=_eval_eps,
            total_steps=train_steps,
            trial_label=trial_label,
        )

        t0 = time.time()
        model.learn(
            total_timesteps=train_steps,
            callback=metrics_cb,
            progress_bar=False,  # we print our own rollout summaries
            reset_num_timesteps=True,
        )
        result["train_time_s"] = round(time.time() - t0, 2)

        # Stage 1: single episode eval to keep it fast (full df per episode)
        _n_eval = 1 if stage == 1 else N_EVAL_EPISODES
        print(f"  Evaluating ({_n_eval} episode{'s' if _n_eval > 1 else ''}) …", flush=True)
        mean_r, std_r = evaluate_policy(
            model, eval_env,
            n_eval_episodes=_n_eval,
            deterministic=True,
            warn=False,
        )
        result["mean_reward"] = round(float(mean_r), 4)
        result["std_reward"]  = round(float(std_r),  4)

        # Collect training metric summaries
        metric_summary = metrics_cb.summary()
        result.update(metric_summary)

        logger.info(
            f"mean_reward={result['mean_reward']}  "
            f"std_reward={result['std_reward']}  "
            f"train_time={result['train_time_s']}s"
        )
        logger.info(f"Training metrics: {metric_summary}")

        elapsed_total = result["train_time_s"]
        print(
            f"  ✓ done  reward={result['mean_reward']:.4f} ± {result['std_reward']:.4f}"
            f"  |  pol={metric_summary.get('final_policy_loss')}"
            f"  val={metric_summary.get('final_value_loss')}"
            f"  ent={metric_summary.get('final_entropy')}"
            f"  |  {elapsed_total}s total",
            flush=True,
        )

        # Save if best
        maybe_save_best(model, float(mean_r), metric_summary, params, stage, trial_id)

    except optuna.exceptions.TrialPruned:
        result["error"] = "PRUNED"
        logger.info("Trial pruned by Optuna")
        print("  ✗ PRUNED by Optuna (underperforming at intermediate check)", flush=True)
        raise  # re-raise so Optuna records it correctly

    except Exception as e:
        tb  = traceback.format_exc()
        result["error"] = f"{type(e).__name__}: {e}\n{tb}"
        logger.error(result["error"])
        print(f"  [ERROR] {str(e)[:200]}")

    return result

def _load_existing_results(stage: int) -> list[dict]:
    """
    Load any previously-saved results for this stage from disk, so that
    resuming a run (e.g. after a crash, or because the Optuna study already
    has completed trials) doesn't wipe out prior CSV rows.

    Returns a list of row-dicts (NaN converted to None) ordered by trial_id.
    Returns an empty list if no CSV exists yet.
    """
    csv_path = os.path.join(RESULTS_DIR, f"stage{stage}_results.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        df_existing = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []

    # Convert NaN -> None so JSON/CSV round-tripping behaves like the
    # in-memory dicts produced by run_trial() (which use None for missing).
    df_existing = df_existing.where(pd.notnull(df_existing), None)
    return df_existing.to_dict(orient="records")


def _merge_results(existing: list[dict], new_row: dict) -> list[dict]:
    """
    Insert/replace `new_row` into `existing` keyed by trial_id, preserving
    order (existing rows keep their position, new trial_ids are appended).
    """
    by_id = {r["trial_id"]: r for r in existing}
    by_id[new_row["trial_id"]] = new_row

    # Preserve original order, then append any genuinely new ids at the end
    ordered_ids = [r["trial_id"] for r in existing]
    for tid in by_id:
        if tid not in ordered_ids:
            ordered_ids.append(tid)
    return [by_id[tid] for tid in ordered_ids]



# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 – Optuna exploration (40 k steps)
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
        print(f"[optuna] Resuming stage 1 — loaded {len(results)} previous result(s) from {os.path.join(RESULTS_DIR, 'stage1_results.csv')}")

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
#  Stage 2 – Refinement (400 k steps, top-K from stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"STAGE 2  –  Refinement  ({STAGE_STEPS[2]:,} steps × top-{top_k} configs)")
    print(f"{'='*80}\n")

    global _best_reward
    _best_reward = _load_best_reward()

    csv_path = os.path.join(RESULTS_DIR, "stage1_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Stage 1 results not found at {csv_path}. Run stage 1 first."
        )

    s1 = pd.read_csv(csv_path)
    s1 = s1[s1["mean_reward"].notna()].sort_values("mean_reward", ascending=False)
    top = s1.head(top_k)

    param_cols = [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
        "max_grad_norm", "net_arch", "activation_fn", "close_strategy",
    ]

    results: list[dict] = _load_existing_results(stage=2)
    if results:
        print(f"[optuna] Resuming stage 2 — loaded {len(results)} previous result(s) from {os.path.join(RESULTS_DIR, 'stage2_results.csv')}")

    stage2_start = time.time()
    for i, row in enumerate(top.itertuples(), 1):
        params  = {c: getattr(row, c) for c in param_cols}
        elapsed = time.time() - stage2_start
        done    = i - 1
        avg_s   = elapsed / done if done else 0
        eta_s   = avg_s * (top_k - done) if done else 0
        print(
            f"\n{'='*80}\n"
            f"  STAGE 2 — Config {i}/{top_k}"
            f"  |  stage1_reward={row.mean_reward:.4f}"
            f"  |  elapsed={elapsed:.0f}s"
            + (f"  |  avg/config={avg_s:.0f}s  ETA={eta_s:.0f}s" if done else "")
            + f"\n{'='*80}",
            flush=True,
        )
        res = run_trial(
            trial=None,
            params=params,
            df=df,
            train_steps=STAGE_STEPS[2],
            stage=2,
            trial_id=i,
            enable_pruning=False,
        )
        results = _merge_results(results, res)
        _save_stage_results(results, stage=2)

    df_results = _save_stage_results(results, stage=2)
    _print_top(df_results, n=5, stage=2)
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
        choices=["1", "2", "both"],
        default="1",
        help="Which stage to run. 'both' runs 1→2 sequentially.",
    )
    parser.add_argument("--n-trials", type=int, default=60, help="Stage 1 trials.")
    parser.add_argument("--top-k",    type=int, default=10, help="Stage 2: top-K configs from stage 1.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows loaded from dataset. 0 = full dataset.",
    )
    args = parser.parse_args()

    df = load_data(max_rows=args.max_rows if args.max_rows > 0 else None)

    if args.stage in ("1", "both"):
        run_stage1(df, n_trials=args.n_trials)
    if args.stage in ("2", "both"):
        run_stage2(df, top_k=args.top_k)