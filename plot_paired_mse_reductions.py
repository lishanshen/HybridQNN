"""
 the four-run paired MSE reduction figure.

Usage:
    python plot_paired_mse_reductions.py
    python plot_paired_mse_reductions.py --show
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
ASSETS = ("AAWW", "ACRX", "CAC", "DJP", "EWV")
SEEDS = (42, 43, 44, 45)


def load_results(metrics_dir):
    results = {}
    for asset in ASSETS:
        values = []
        for seed in SEEDS:
            path = metrics_dir / asset / f"seed_{seed}" / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            hybrid = float(metrics["hybrid"]["MSE"])
            classical = float(metrics["classical"]["MSE"])
            if not (math.isfinite(hybrid) and hybrid >= 0
                    and math.isfinite(classical) and classical > 0):
                raise ValueError(f"Invalid MSE values: {path}")
            values.append(100 * (1 - hybrid / classical))
        results[asset] = {"seeds": list(SEEDS), "values": values, "mean": mean(values)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path,
                        default=ROOT / "results/single_stock")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output/paired_mse_reductions")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(args.metrics_dir)
    averages = [results[a]["mean"] for a in ASSETS]
    grand_mean = mean(averages)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
    ax.bar(ASSETS, averages, color="#356c91", alpha=.82, width=.6,
           label="Mean of four paired reductions")
    for i, asset in enumerate(ASSETS):
        ax.scatter([i + offset for offset in (-.18, -.06, .06, .18)],
                   results[asset]["values"], c="#bf4a24", s=20, zorder=4)
        ax.text(i, averages[i] + 3, f"{averages[i]:.2f}%", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=.6)
    ax.axhline(grand_mean, color="#555555", ls="--", lw=.9)
    ax.set_ylabel("Paired MSE reduction (%)")
    ax.set_ylim(-45, 112)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(.99, .04, f"Five-asset mean: {grand_mean:.2f}%\nDots: seeds 42, 43, 44, 45",
            transform=ax.transAxes, ha="right", fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"paired_mse_reductions.{extension}", dpi=300)
    summary = {"assets": results, "five_asset_mean_percent": grand_mean}
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    caption = ("Four-run paired MSE reductions: bars show asset means, dots show individual "
               f"runs, and the dashed line marks the five-asset mean of {grand_mean:.2f}%.")
    (args.output_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    for asset in ASSETS:
        print(f"{asset}: {results[asset]['mean']:.10f}%")
    print(f"Five-asset mean before rounding: {grand_mean:.15g}%")
    print(f"Saved to: {args.output_dir.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
