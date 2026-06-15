import os
import numpy as np
 
from indicators import load_and_compute, SIGNAL_INDICATORS, CONTINUOUS_FEATURES
from tradingenv import TradingEnv
 
# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
 
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "BTCUSDT_1m.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
#  Indicator routing
#  TradingEnv expects two separate lists:
#    T_indicators  — used in "trending"      regime  (Hurst > 0.5)
#    MR_indicators — used in "mean_reverting" regime (Hurst <= 0.5)
#
#  The env reads *_signal columns from the DataFrame.
# ─────────────────────────────────────────────────────────────────────────────
 
# Signal columns produced by indicators.py  →  e.g. "RSI_signal"
SIGNAL_COLS = [f"{ind}_signal" for ind in SIGNAL_INDICATORS]
 
# Trending regime: momentum / trend-following indicators
# Trending regime
T_INDICATOR_NAMES = ["MACD", "MA", "OBV", "HA"]
T_INDICATORS      = T_INDICATOR_NAMES          # ← bare names, NOT f"{ind}_signal"

# Mean-reverting regime  
MR_INDICATOR_NAMES = ["RSI", "BBANDS", "STOCH", "CCI"]
MR_INDICATORS      = MR_INDICATOR_NAMES        # ← bare names



# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
 
INITIAL_BALANCE = 10_000.0
TRAIL_PCT       = 0.05
CLOSE_STRATEGY  = "fifo"
N_EPISODES      = 5          # adjust as needed
 
# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────
 
def prepare_data(csv_path: str):
    """
    Run indicators.py pipeline and return a clean DataFrame ready for TradingEnv.
 
    Columns guaranteed to be present after this call
    -------------------------------------------------
    Open, High, Low, Close, Volume
    RSI_signal, MACD_signal, MA_signal, HA_signal,
    STOCH_signal, BBANDS_signal, CCI_signal, OBV_signal
    RSI_strength, MACD_strength, MA_strength, HA_strength,
    STOCH_strength, BBANDS_strength, CCI_strength, OBV_strength
    CMF, VWAP, ATR, VOLATILITY, PARKINSON, PRICE_ACTION
    dist_from_high, dist_from_low, RSI
    """
    print(f"[train] Loading data from: {csv_path}")
    df = load_and_compute(csv_path)

    # Sanity-check: signal columns exist (indicators produce *_signal columns)
    missing = [f"{ind}_signal" for ind in T_INDICATORS + MR_INDICATORS
               if f"{ind}_signal" not in df.columns]
    if missing:
        raise ValueError(f"[train] Missing indicator columns: {missing}")

    missing_cont = [c for c in CONTINUOUS_FEATURES if c not in df.columns]
    if missing_cont:
        raise ValueError(f"[train] Missing continuous-feature columns: {missing_cont}")

    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────────────
 
def make_env(df):
    return TradingEnv(
        df=df,
        T_indicators=T_INDICATORS,
        MR_indicators=MR_INDICATORS,
        continuous_features=CONTINUOUS_FEATURES,
        initial_balance=INITIAL_BALANCE,
        close_strategy=CLOSE_STRATEGY,
        trail_pct=TRAIL_PCT,
    )

def run_random_baseline(env: TradingEnv, n_episodes: int = N_EPISODES):
    """
    Drives the env with random actions to verify the pipeline end-to-end.
    Replace this loop with your RL agent's .learn() call.
    """
    for ep in range(n_episodes):
        obs, info = env.reset()
        done       = False
        total_reward = 0.0
        step_count   = 0
 
        print(f"\n[episode {ep + 1}/{n_episodes}] start  "
              f"portfolio={info['portfolio_value']:.2f}")
 
        while not done:
            action = env.action_space.sample()          # ← replace with agent
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count   += 1
            done = terminated or truncated
 
        print(f"[episode {ep + 1}/{n_episodes}] end    "
              f"steps={step_count}  total_reward={total_reward:.4f}  "
              f"portfolio={info['portfolio_value']:.2f}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    df = prepare_data(DATA_PATH)

    print("\n===== DATA CHECK =====")
    print("Shape:", df.shape)
    print("\nSignal columns:")
    signal_cols = [f"{ind}_signal" for ind in T_INDICATORS + MR_INDICATORS]
    print(df[signal_cols].head())

    # And the distribution loop:
    for col in [f"{ind}_signal" for ind in T_INDICATORS + MR_INDICATORS]:
        print(f"{col}:")
        print(df[col].value_counts().sort_index())
        print()

    print("\nContinuous features:")
    print(df[CONTINUOUS_FEATURES].head())


    env = make_env(df)

    obs, info = env.reset()

    print("\n===== ENV CHECK =====")
    print("Observation shape:", obs.shape)
    print("Observation length:", len(obs))
    print("Initial portfolio:", info["portfolio_value"])

    print("\nFirst observation:")
    print(obs)

    print("\n===== RANDOM STEPS =====")
    for i in range(10):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i+1:2d} "
            f"action={action} "
            f"reward={reward:8.4f} "
            f"portfolio={info['portfolio_value']:10.2f}"
        )

        if terminated or truncated:
            print("Episode ended early.")
            break

    print("\n===== FULL RANDOM RUN =====")
    run_random_baseline(env, n_episodes=N_EPISODES)