"""
  Key features vs target scatter plot
  Correlation of Features with Close / Adj Close (heatmap)

Usage:
  python 2_plots.py
"""

import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration 
DATA_PATH = "./Dataset/DJIA/processed_data.csv"
OUTPUT_DIR = "./results/figures"

FEATURES = ["Open", "High", "Low", "Volume", "pos_mean",
            "open_today", "prev_adj_Close", "ma_5", "ma_20"]
TARGETS = ["Close", "Adj Close"]

# Scatter plot: key features to plot against the target
SCATTER_FEATURES = ["pos_mean", "Volume", "open_today",
                    "prev_adj_Close", "ma_5", "ma_20"]
SCATTER_COLOR = "red"
SCATTER_N_COLS = 3

# Heatmap: seaborn's default annotation size is 10, so +2 means 12
HEATMAP_FONT_SIZE = 12


def plot_scatter(df):
    """Draw each key feature against Adj Close, save as a PNG."""
    plot_feats = [c for c in SCATTER_FEATURES if c in df.columns]
    if not plot_feats:
        print("No scatter features found in the data; skipping scatter plot.")
        return

    n_rows = -(-len(plot_feats) // SCATTER_N_COLS)  # ceiling division
    fig, axes = plt.subplots(n_rows, SCATTER_N_COLS, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, plot_feats):
        ax.scatter(df[col], df["Adj Close"], alpha=0.4, s=8, color=SCATTER_COLOR)
        ax.set_xlabel(col)
        ax.set_ylabel("Adj Close")
        ax.set_title(f"{col} vs Adj Close")

    # Hide any unused subplot slots
    for ax in axes[len(plot_feats):]:
        ax.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "key_features_vs_target.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to: {save_path}")


def plot_correlation_heatmap(df):
    """Draw the feature-vs-target correlation heatmap, save as a PNG."""
    features = [c for c in FEATURES if c in df.columns]
    targets = [c for c in TARGETS if c in df.columns]

    # Keep only the feature rows and target columns
    corr_sub = df[features + targets].corr().loc[features, targets]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_sub, annot=True, cmap="coolwarm", center=0,
                linewidths=0.5, fmt=".2f",
                annot_kws={"size": HEATMAP_FONT_SIZE},
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation of Features with Close / Adj Close",
              fontsize=HEATMAP_FONT_SIZE + 2)
    plt.xticks(fontsize=HEATMAP_FONT_SIZE)
    plt.yticks(fontsize=HEATMAP_FONT_SIZE)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "correlation_features_targets.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    plot_scatter(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()
