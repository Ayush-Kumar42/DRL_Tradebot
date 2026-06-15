"""
plot_results.py
---------------
Reads results/search_results.csv (written by hyperparam_search.py) and
produces a set of charts saved to results/plots/.

Charts produced
---------------
1. mean_reward_by_algo_strategy.png
   — Grouped bar chart: mean reward per (algo, strategy) combination,
     error bars = std across hyperparameter variants.

2. top10_runs.png
   — Horizontal bar chart of the 10 best runs (mean_reward), labelled
     with algo + strategy + key hyperparams.

3. heatmap_<algo>.png  (one per algorithm)
   — Heatmap of mean_reward averaged over strategies, with the two most
     impactful hyperparams on each axis (learning_rate × the next best).

4. strategy_comparison_boxplot.png
   — Box plot: distribution of mean_reward per closing strategy,
     across all algos and hyperparameter settings.

5. final_portfolio_by_algo.png
   — Box plot: final portfolio value per algorithm.

6. parallel_coords_<algo>.png  (one per algorithm)
   — Parallel coordinates plot of hyperparams coloured by mean_reward.

Usage
-----
    python plot_results.py [--results-dir src/results]
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS   = {"PPO": "#4C72B0", "A2C": "#DD8452", "DQN": "#55A868"}
STRATEGY_ORDER = [
    "fifo", "max_profit", "least_loss",
    "age_weighted", "most_risk", "pnl_balanced",
]

ALGO_HPARAM_AXES: dict[str, tuple[str, str]] = {
    "PPO": ("learning_rate", "n_steps"),
    "A2C": ("learning_rate", "n_steps"),
    "DQN": ("learning_rate", "exploration_fraction"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "search_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Run hyperparam_search.py first."
        )
    df = pd.read_csv(path)
    df = df[df["error"].isna()].copy()   # drop failed runs
    df["mean_reward"]      = pd.to_numeric(df["mean_reward"],      errors="coerce")
    df["std_reward"]       = pd.to_numeric(df["std_reward"],       errors="coerce")
    df["final_portfolio"]  = pd.to_numeric(df["final_portfolio"],  errors="coerce")
    print(f"[plot] Loaded {len(df)} successful runs from {path}")
    return df


def savefig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 1 — Grouped bar: mean reward by (algo, strategy)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_by_algo_strategy(df: pd.DataFrame, out_dir: str):
    agg = (
        df.groupby(["algo", "strategy"])["mean_reward"]
        .agg(["mean", "std"])
        .reset_index()
    )

    strategies = [s for s in STRATEGY_ORDER if s in agg["strategy"].unique()]
    algos      = list(ALGO_COLORS.keys())
    x          = np.arange(len(strategies))
    width      = 0.25
    offsets    = np.linspace(-width, width, len(algos))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, algo in enumerate(algos):
        sub = agg[agg["algo"] == algo].set_index("strategy")
        means = [sub.loc[s, "mean"] if s in sub.index else 0 for s in strategies]
        stds  = [sub.loc[s, "std"]  if s in sub.index else 0 for s in strategies]
        ax.bar(
            x + offsets[i], means, width * 0.9,
            yerr=stds, capsize=4,
            label=algo, color=ALGO_COLORS[algo], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_xlabel("Closing Strategy")
    ax.set_ylabel("Mean Reward (avg over hyperparams)")
    ax.set_title("Mean Reward by Algorithm × Closing Strategy")
    ax.legend(title="Algorithm")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "mean_reward_by_algo_strategy.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 2 — Top-10 runs
# ─────────────────────────────────────────────────────────────────────────────

def plot_top10(df: pd.DataFrame, out_dir: str):
    top = df.nlargest(10, "mean_reward").copy()

    # Build a readable label from algo + strategy + key params
    param_cols = [c for c in df.columns if c not in
                  {"algo", "strategy", "mean_reward", "std_reward",
                   "final_portfolio", "train_time_s", "error"}]

    def label(row):
        parts = [f"{row['algo']} | {row['strategy']}"]
        for col in param_cols[:3]:                      # up to 3 params in label
            if pd.notna(row.get(col)):
                parts.append(f"{col}={row[col]:.2g}")
        return "\n".join(parts)

    top["label"] = top.apply(label, axis=1)
    colors = [ALGO_COLORS.get(a, "#888") for a in top["algo"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(len(top)), top["mean_reward"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Reward")
    ax.set_title("Top 10 Runs")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", alpha=0.3)

    legend_els = [
        Line2D([0], [0], color=c, linewidth=8, label=a)
        for a, c in ALGO_COLORS.items()
    ]
    ax.legend(handles=legend_els, title="Algorithm", loc="lower right")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "top10_runs.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 3 — Heatmap per algo (lr × n_steps / exploration_fraction)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(df: pd.DataFrame, out_dir: str):
    for algo, (x_col, y_col) in ALGO_HPARAM_AXES.items():
        sub = df[(df["algo"] == algo) & df[x_col].notna() & df[y_col].notna()].copy()
        if sub.empty:
            print(f"[plot] Skipping heatmap for {algo} — no data with {x_col}/{y_col}")
            continue

        pivot = (
            sub.groupby([y_col, x_col])["mean_reward"]
            .mean()
            .unstack(x_col)
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="Mean Reward")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns], rotation=30)
        ax.set_yticklabels([f"{v:.2g}" for v in pivot.index])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{algo} — Mean Reward Heatmap ({y_col} × {x_col})")

        # Annotate cells
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, f"heatmap_{algo}.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 4 — Strategy comparison boxplot
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_boxplot(df: pd.DataFrame, out_dir: str):
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    data = [df[df["strategy"] == s]["mean_reward"].dropna().values for s in strategies]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(
        data, labels=strategies, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Closing Strategy")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Distribution per Closing Strategy (all algos & hyperparams)")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="break-even")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "strategy_comparison_boxplot.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 5 — Final portfolio by algo
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_by_algo(df: pd.DataFrame, out_dir: str):
    algos = list(ALGO_COLORS.keys())
    data  = [df[df["algo"] == a]["final_portfolio"].dropna().values for a in algos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data, labels=algos, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, (algo, color) in zip(bp["boxes"], ALGO_COLORS.items()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(10_000, color="grey", linewidth=1, linestyle="--", label="initial balance")
    ax.set_ylabel("Final Portfolio Value ($)")
    ax.set_title("Final Portfolio Distribution by Algorithm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "final_portfolio_by_algo.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 6 — Parallel coordinates per algo
# ─────────────────────────────────────────────────────────────────────────────

def plot_parallel_coords(df: pd.DataFrame, out_dir: str):
    for algo in ALGO_COLORS:
        sub = df[df["algo"] == algo].copy()
        if sub.empty:
            continue

        param_cols = [c for c in ALGO_HPARAM_AXES.get(algo, ())]+ \
                     [c for c in df.columns if c not in
                      {"algo", "strategy", "mean_reward", "std_reward",
                       "final_portfolio", "train_time_s", "error"}
                      and c not in list(ALGO_HPARAM_AXES.get(algo, ()))
                      and sub[c].notna().any()]

        use_cols = [c for c in param_cols if sub[c].notna().all()]
        if not use_cols:
            continue

        # Normalise each param column to [0,1] for display
        plot_df = sub[use_cols + ["mean_reward"]].dropna().copy()
        for col in use_cols:
            lo, hi = plot_df[col].min(), plot_df[col].max()
            if hi > lo:
                plot_df[col] = (plot_df[col] - lo) / (hi - lo)

        # Colour by mean_reward
        norm  = mcolors.Normalize(
            vmin=plot_df["mean_reward"].min(),
            vmax=plot_df["mean_reward"].max(),
        )
        cmap  = cm.RdYlGn
        colors = [cmap(norm(v)) for v in plot_df["mean_reward"]]

        fig, ax = plt.subplots(figsize=(12, 5))
        xs = range(len(use_cols))
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            ax.plot(xs, row[use_cols].values, color=colors[idx], alpha=0.4, linewidth=0.8)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(use_cols, rotation=20, ha="right")
        ax.set_ylabel("Normalised value")
        ax.set_title(f"{algo} — Parallel Coordinates (coloured by mean_reward)")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Mean Reward")
        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, f"parallel_coords_{algo}.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n===== BEST RUN PER ALGO =====")
    for algo in ALGO_COLORS:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        best = sub.loc[sub["mean_reward"].idxmax()]
        print(f"\n{algo}")
        print(f"  Strategy      : {best['strategy']}")
        print(f"  Mean reward   : {best['mean_reward']:.4f}")
        print(f"  Final portfolio: ${best['final_portfolio']:,.2f}")
        param_cols = [c for c in df.columns if c not in
                      {"algo", "strategy", "mean_reward", "std_reward",
                       "final_portfolio", "train_time_s", "error"}]
        for col in param_cols:
            if pd.notna(best.get(col)):
                print(f"  {col:30s}: {best[col]}")

    print("\n===== BEST STRATEGY OVERALL =====")
    by_strat = df.groupby("strategy")["mean_reward"].mean().sort_values(ascending=False)
    print(by_strat.to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(results_dir: str):
    out_dir = os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load(results_dir)
    print_summary(df)

    plot_reward_by_algo_strategy(df, out_dir)
    plot_top10(df, out_dir)
    plot_heatmaps(df, out_dir)
    plot_strategy_boxplot(df, out_dir)
    plot_portfolio_by_algo(df, out_dir)
    plot_parallel_coords(df, out_dir)

    print(f"\n[plot] All charts saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    args = parser.parse_args()
    main(args.results_dir)