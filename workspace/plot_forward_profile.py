#!/usr/bin/env python3
"""
Scatter plot: x = block_size * batch_size, y = forward_time_avg_ms
from a block_size x batch_size profile CSV.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_profile(csv_path: Path) -> list[dict]:
    """Load profile CSV; return rows with block_size, batch_size, forward_time_avg_ms."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                block_size = int(row["block_size"])
                batch_size = int(row["batch_size"])
                raw = row.get("forward_time_avg_ms", "").strip()
                if raw.upper() == "N/A" or raw == "":
                    continue
                forward_time_avg_ms = float(raw)
                rows.append({
                    "block_size": block_size,
                    "batch_size": batch_size,
                    "forward_time_avg_ms": forward_time_avg_ms,
                })
            except (ValueError, KeyError):
                continue
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scatter plot: block_size*batch_size vs forward_time_avg_ms")
    parser.add_argument(
        "csv",
        nargs="?",
        default="SDAR_block_size_batch_size_full.csv",
        help="Profile CSV path",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("forward_profile.png"),
        help="Save figure to path (default: forward_profile.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Error: file not found: {csv_path}")

    rows = load_profile(csv_path)
    if not rows:
        raise SystemExit("Error: no valid rows with forward_time_avg_ms found.")

    # Linear regression: forward_time_avg_ms ~ (block_size * batch_size) + batch_size + block_size
    bs_batch = np.array([r["block_size"] * r["batch_size"] for r in rows], dtype=float)
    batch = np.array([r["batch_size"] for r in rows], dtype=float)
    block = np.array([r["block_size"] for r in rows], dtype=float)
    y = np.array([r["forward_time_avg_ms"] for r in rows], dtype=float)
    X = np.column_stack([np.ones(len(rows)), bs_batch, batch, block])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / len(y))

    print("Linear regression: forward_time_avg_ms ~ intercept + (block_size×batch_size) + batch_size + block_size")
    print(f"  Coefficients: intercept={coeffs[0]:.4f}, block_size×batch_size={coeffs[1]:.6f}, batch_size={coeffs[2]:.4f}, block_size={coeffs[3]:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  RMSE = {rmse:.4f} ms")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))
    unique_bs = sorted({r["block_size"] for r in rows})
    colors = plt.cm.tab10.colors[: len(unique_bs)]

    for bs, color in zip(unique_bs, colors):
        pts = [r for r in rows if r["block_size"] == bs]
        x = [r["block_size"] * r["batch_size"] for r in pts]
        y_left = [r["forward_time_avg_ms"] for r in pts]
        y_right = [r["forward_time_avg_ms"] / (r["block_size"] * r["batch_size"]) for r in pts]
        ax0.scatter(x, y_left, c=[color], alpha=0.7, s=20, label=str(bs), edgecolors="none")
        ax1.scatter(x, y_right, c=[color], alpha=0.7, s=20, label=str(bs), edgecolors="none")

    for ax, ylabel, title in [
        (ax0, "forward_time_avg_ms", "Forward time vs block_size × batch_size"),
        (ax1, "forward_time_avg_ms / (block_size × batch_size)", "Forward time per token vs block_size × batch_size"),
    ]:
        ax.legend(title="block_size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("block_size × batch_size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Third subplot: actual vs predicted (linear fit quality)
    for bs, color in zip(unique_bs, colors):
        mask = np.array([r["block_size"] == bs for r in rows])
        ax2.scatter(y[mask], y_pred[mask], c=[color], alpha=0.7, s=20, label=str(bs), edgecolors="none")
    lim_lo = min(y.min(), y_pred.min())
    lim_hi = max(y.max(), y_pred.max())
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.6, label="y = x")
    ax2.set_xlabel("Actual forward_time_avg_ms")
    ax2.set_ylabel("Predicted forward_time_avg_ms")
    ax2.set_title(f"Linear fit: actual vs predicted (R² = {r_squared:.4f})")
    ax2.legend(title="block_size")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
