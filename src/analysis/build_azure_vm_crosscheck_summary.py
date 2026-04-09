"""
Build a small, package-friendly telemetry cross-check summary from the Azure VM Traces (2019, Public Dataset V2).

Why this exists:
- The canonical Figure 3 calibration in the paper is proxy-based (public incident metadata).
- Reviewers may reasonably ask for a direct-telemetry cross-check.
- Shipping the full Azure traces is infeasible (very large).

This script ingests one (or more) Azure VM CPU shard(s) and produces a compact, reproducible, binned summary:
    - total samples per load bin
    - "burst to saturation" event count per load bin, defined via max_cpu threshold

The resulting binned CSV can be committed into the repo and used to render a telemetry panel in Figure 3
without requiring the raw Azure dataset at reproduction time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinSpec:
    edges: np.ndarray  # shape (n_bins+1,)

    @property
    def n_bins(self) -> int:
        return int(len(self.edges) - 1)

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cpu-shards",
        nargs="+",
        required=True,
        help="One or more Azure VM CPU shard .csv.gz files (vm_cpu_readings-file-*-of-195.csv.gz).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (will write azure_vm_crosscheck_binned.csv and azure_vm_crosscheck_summary.json).",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of load bins over rho in [0,1].",
    )
    p.add_argument(
        "--burst-threshold",
        type=float,
        default=95.0,
        help="Define a burst event as max_cpu >= this threshold (in percent, 0-100).",
    )
    p.add_argument(
        "--min-count-per-bin",
        type=int,
        default=5000,
        help="Bins with fewer samples are retained in CSV but can be excluded during fitting/plotting.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size for streaming CSV reads.",
    )
    return p.parse_args()


def get_bin_spec(n_bins: int) -> BinSpec:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return BinSpec(edges=edges)


def update_counts(
    total_counts: np.ndarray,
    event_counts: np.ndarray,
    bins: BinSpec,
    avg_cpu_pct: np.ndarray,
    max_cpu_pct: np.ndarray,
    burst_threshold: float,
) -> None:
    rho = np.clip(avg_cpu_pct / 100.0, 0.0, 1.0)
    event = max_cpu_pct >= burst_threshold

    # Map rho to bins.
    # np.digitize returns 1..n_bins; we convert to 0..n_bins-1
    idx = np.digitize(rho, bins.edges, right=False) - 1
    valid = (idx >= 0) & (idx < bins.n_bins) & np.isfinite(rho)
    idx = idx[valid]
    event = event[valid]

    if idx.size == 0:
        return

    # Total counts per bin.
    binc = np.bincount(idx, minlength=bins.n_bins)
    total_counts[: bins.n_bins] += binc

    # Event counts per bin.
    idx_event = idx[event]
    if idx_event.size:
        bine = np.bincount(idx_event, minlength=bins.n_bins)
        event_counts[: bins.n_bins] += bine


def build_binned_summary(
    cpu_shards: list[Path],
    bins: BinSpec,
    burst_threshold: float,
    chunksize: int,
) -> tuple[pd.DataFrame, dict]:
    total_counts = np.zeros(bins.n_bins, dtype=np.int64)
    event_counts = np.zeros(bins.n_bins, dtype=np.int64)

    # Azure schema for shards:
    # timestamp, vm_id, min_cpu, max_cpu, avg_cpu
    usecols = [0, 3, 4]
    colnames = ["timestamp", "max_cpu", "avg_cpu"]

    n_rows = 0
    for shard in cpu_shards:
        for chunk in pd.read_csv(
            shard,
            header=None,
            usecols=usecols,
            names=colnames,
            chunksize=chunksize,
        ):
            # Cast to numeric, drop bad rows.
            avg_cpu = pd.to_numeric(chunk["avg_cpu"], errors="coerce").to_numpy()
            max_cpu = pd.to_numeric(chunk["max_cpu"], errors="coerce").to_numpy()
            good = np.isfinite(avg_cpu) & np.isfinite(max_cpu)
            avg_cpu = avg_cpu[good]
            max_cpu = max_cpu[good]

            n_rows += int(avg_cpu.size)
            update_counts(total_counts, event_counts, bins, avg_cpu, max_cpu, burst_threshold)

    df = pd.DataFrame(
        {
            "rho_center": bins.centers,
            "rho_left": bins.edges[:-1],
            "rho_right": bins.edges[1:],
            "n_total": total_counts,
            "n_burst": event_counts,
        }
    )
    df["p_burst"] = np.where(df["n_total"] > 0, df["n_burst"] / df["n_total"], np.nan)
    meta = {
        "cpu_shards": [str(p) for p in cpu_shards],
        "n_rows_processed": int(n_rows),
        "n_bins": int(bins.n_bins),
        "burst_threshold_pct": float(burst_threshold),
    }
    return df, meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = [Path(p) for p in args.cpu_shards]
    bins = get_bin_spec(args.n_bins)

    df, meta = build_binned_summary(
        cpu_shards=shards,
        bins=bins,
        burst_threshold=float(args.burst_threshold),
        chunksize=int(args.chunksize),
    )

    out_csv = out_dir / "azure_vm_crosscheck_binned.csv"
    out_json = out_dir / "azure_vm_crosscheck_summary.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                **meta,
                "min_count_per_bin": int(args.min_count_per_bin),
                "notes": (
                    "This binned summary is intended for a telemetry cross-check panel: "
                    "we examine how the probability of a burst-to-saturation event (max_cpu >= threshold) "
                    "varies with average load (avg_cpu)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

