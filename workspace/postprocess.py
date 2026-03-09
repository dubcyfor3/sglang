#!/usr/bin/env python3
"""
Postprocess block_size x batch_size profile CSV to compute end-to-end inference
latency and tokens/sec using the DLLM forward model:

  E2E time = num_blocks * (small_fwd_num * small_fwd_time + big_fwd_num * big_fwd_time)

Where per block:
  - small_fwd_num = block_size / num_transfer_token - 1
  - big_fwd_num = 1
  - small_fwd_time = forward_time_avg_ms(block_size, batch_size)
  - big_fwd_time = forward_time_avg_ms(2 * block_size, batch_size)

num_blocks = seq_len / block_size. We report best block size per batch size.
"""

import argparse
import csv
import sys
from pathlib import Path

# Constants from your setup
SEQ_LEN = 256

# Default path for num_transfer_tokens (same dir as this script)
SCRIPT_DIR = Path(__file__).resolve().parent
NUM_TRANSFER_TOKENS_CSV = SCRIPT_DIR / "num_transfer_tokens.csv"


def load_num_transfer_tokens(csv_path: Path, model: str) -> dict[int, float]:
    """Load num_transfer_tokens.csv: block_size -> num_transfer_tokens (float) for the given model column."""
    out: dict[int, float] = {}
    if not csv_path.is_file():
        return out
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if model not in (reader.fieldnames or []):
            print(
                f"Warning: model '{model}' not in CSV columns {list(reader.fieldnames or [])}; using 1.0 for all block sizes.",
                file=sys.stderr,
            )
            return out
        for row in reader:
            try:
                blk = int(row["block_size"])
                out[blk] = float(row[model])
            except (ValueError, KeyError):
                continue
    return out


def load_profile(csv_path: str) -> list[dict]:
    """Load profile CSV; return list of rows with numeric forward_time_avg_ms."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["block_size"] = int(row["block_size"])
                row["batch_size"] = int(row["batch_size"])
                raw = row.get("forward_time_avg_ms", "").strip()
                if raw.upper() == "N/A" or raw == "":
                    continue
                row["forward_time_avg_ms"] = float(raw)
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def build_lookup(rows: list[dict]) -> dict[tuple[int, int], float]:
    """(block_size, batch_size) -> forward_time_avg_ms."""
    return {(r["block_size"], r["batch_size"]): r["forward_time_avg_ms"] for r in rows}


def e2e_time_ms(
    block_size: int,
    batch_size: int,
    lookup: dict[tuple[int, int], float],
    num_transfer_by_block: dict[int, float],
    seq_len: int = SEQ_LEN,
) -> float | None:
    """
    E2E time in ms for this (block_size, batch_size).
    Returns None if we don't have small_fwd or big_fwd (2*block_size) data.
    num_transfer_by_block[block_size] used for small_fwd_num; fallback 1.0 if missing.
    """
    small_fwd_time = lookup.get((block_size, batch_size))
    big_block = 2 * block_size
    big_fwd_time = lookup.get((big_block, batch_size))
    if small_fwd_time is None or big_fwd_time is None:
        return None
    num_transfer = num_transfer_by_block.get(block_size, 1.0)
    small_fwd_num = block_size / num_transfer - 1
    big_fwd_num = 1
    per_block_ms = small_fwd_num * small_fwd_time + big_fwd_num * big_fwd_time
    num_blocks = seq_len // block_size
    return num_blocks * per_block_ms


def main():
    parser = argparse.ArgumentParser(description="Compute E2E latency and best block size per batch size.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="block_size_batch_size_profile_full.csv",
        help="Profile CSV path (default: block_size_batch_size_profile_full.csv)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length (default: {SEQ_LEN})",
    )
    parser.add_argument(
        "--num-transfer-tokens-csv",
        type=Path,
        default=NUM_TRANSFER_TOKENS_CSV,
        help="CSV with block_size and one column per model (default: num_transfer_tokens.csv in script dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SDAR-8B",
        help="Model name: column in num_transfer_tokens CSV to use (default: SDAR-8B)",
    )
    args = parser.parse_args()
    seq_len = args.seq_len

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    num_transfer_by_block = load_num_transfer_tokens(args.num_transfer_tokens_csv, args.model)
    if not num_transfer_by_block:
        print("Warning: no num_transfer_tokens loaded; using 1.0 for all block sizes.", file=sys.stderr)

    rows = load_profile(csv_path)
    if not rows:
        print("Error: no valid rows with forward_time_avg_ms found.", file=sys.stderr)
        sys.exit(1)

    lookup = build_lookup(rows)
    block_sizes = sorted({r["block_size"] for r in rows})
    batch_sizes = sorted({r["batch_size"] for r in rows})

    # AR (autoregressive) E2E time per batch: block_size=1 fwd time * seq_len
    ar_e2e_ms: dict[int, float | None] = {}
    for bs in batch_sizes:
        fwd_1 = lookup.get((1, bs))
        ar_e2e_ms[bs] = (fwd_1 * seq_len) if fwd_1 is not None else None

    # Collect (batch_size, block_size) -> e2e_ms, tokens_per_sec
    results = []
    for bs in batch_sizes:
        for blk in block_sizes:
            if seq_len % blk != 0:
                continue
            t_ms = e2e_time_ms(blk, bs, lookup, num_transfer_by_block, seq_len)
            if t_ms is None:
                continue
            t_sec = t_ms / 1000.0
            # Total tokens per second for the batch (batch_size * seq_len tokens in t_sec)
            total_tokens = bs * seq_len
            tps = total_tokens / t_sec if t_sec > 0 else 0.0
            ar_e2e = ar_e2e_ms.get(bs)
            speedup_vs_ar = (ar_e2e / t_ms) if (ar_e2e is not None and t_ms > 0) else None
            results.append(
                {
                    "batch_size": bs,
                    "block_size": blk,
                    "e2e_time_ms": t_ms,
                    "tokens_per_sec": tps,
                    "ar_e2e_time_ms": ar_e2e,
                    "speedup_vs_AR": speedup_vs_ar,
                }
            )

    # Sort by batch_size then by block_size for display
    results.sort(key=lambda x: (x["batch_size"], x["block_size"]))

    # Baseline: tokens_per_sec for block_size=32 per batch_size
    tps_b32: dict[int, float] = {}
    for r in results:
        if r["block_size"] == 32:
            tps_b32[r["batch_size"]] = r["tokens_per_sec"]

    # Add speedup over block size 32 for each result
    for r in results:
        base = tps_b32.get(r["batch_size"])
        if base is not None and base > 0:
            r["speedup_vs_block32"] = r["tokens_per_sec"] / base
        else:
            r["speedup_vs_block32"] = None

    # Print table
    print("batch_size,block_size,e2e_time_ms,tokens_per_sec,speedup_vs_block32,ar_e2e_time_ms,speedup_vs_AR")
    for r in results:
        speedup_str = f"{r['speedup_vs_block32']:.3f}" if r["speedup_vs_block32"] is not None else "N/A"
        ar_str = f"{r['ar_e2e_time_ms']:.2f}" if r["ar_e2e_time_ms"] is not None else "N/A"
        ar_speedup_str = f"{r['speedup_vs_AR']:.3f}" if r["speedup_vs_AR"] is not None else "N/A"
        print(f"{r['batch_size']},{r['block_size']},{r['e2e_time_ms']:.2f},{r['tokens_per_sec']:.2f},{speedup_str},{ar_str},{ar_speedup_str}")

    # Best block size per batch size (max tokens_per_sec)
    print("\n--- Best block size per batch size (by tokens/sec) ---")
    by_batch = {}
    for r in results:
        b = r["batch_size"]
        if b not in by_batch or r["tokens_per_sec"] > by_batch[b]["tokens_per_sec"]:
            by_batch[b] = r

    for bs in sorted(by_batch.keys()):
        r = by_batch[bs]
        speedup_str = f", {r['speedup_vs_block32']:.3f}x vs block32" if r.get("speedup_vs_block32") is not None else ""
        ar_str = f", {r['speedup_vs_AR']:.3f}x vs AR" if r.get("speedup_vs_AR") is not None else ""
        print(
            f"  batch_size={bs}: best block_size={r['block_size']} "
            f"(e2e={r['e2e_time_ms']:.2f} ms, {r['tokens_per_sec']:.2f} tok/s{speedup_str}{ar_str})"
        )

    # Best block size per batch size (by AR speedup) — same as by tokens/sec when AR time is fixed per batch
    print("\n--- Best block size per batch size (by AR speedup) ---")
    by_batch_ar = {}
    for r in results:
        if r.get("speedup_vs_AR") is None:
            continue
        b = r["batch_size"]
        if b not in by_batch_ar or r["speedup_vs_AR"] > by_batch_ar[b]["speedup_vs_AR"]:
            by_batch_ar[b] = r

    for bs in sorted(by_batch_ar.keys()):
        r = by_batch_ar[bs]
        print(
            f"  batch_size={bs}: best block_size={r['block_size']} "
            f"(speedup_vs_AR={r['speedup_vs_AR']:.3f}x, e2e={r['e2e_time_ms']:.2f} ms, AR_e2e={r['ar_e2e_time_ms']:.2f} ms)"
        )


if __name__ == "__main__":
    main()
