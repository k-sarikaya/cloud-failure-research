"""
Cloud Uptime Archive (CUA) cross-check.

Goal:
- Provide an external validation touchpoint for the "incident-frequency proxy" idea used in the proxy-based calibration.
- CUA provides outage intervals (start/end) + severity scores for many services and sources.

What this script does:
- Loads selected CUA CSV traces (service-source pairs).
- Computes per-trace summary statistics:
  - number of outages, duration distribution, inter-arrival distribution, severity distribution
- Writes a compact CSV + JSON summary that can be shipped in the repo/supplementary without bundling the full dataset.

Input CSV schema (from Zenodo record 14712442):
  Start time, End time, Severity
  timestamp, timestamp, value between 0-1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TraceStats:
    trace_name: str
    n_outages: int
    start_min: str
    start_max: str
    dur_median_min: float
    dur_p90_min: float
    ia_median_min: float
    ia_p90_min: float
    severity_mean: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cua-csvs",
        nargs="+",
        required=True,
        help="One or more Cloud Uptime Archive CSV trace files.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (writes cloud_uptime_archive_crosscheck.csv/json).",
    )
    p.add_argument(
        "--timezone",
        default="UTC",
        help="Assume timestamps are in this timezone when parsing (default UTC).",
    )
    return p.parse_args()


def read_cua_csv(path: Path, timezone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) < 3:
        raise ValueError(f"Unexpected CUA schema in {path}: columns={list(df.columns)}")

    # Two common schemas observed in this record:
    # 1) start_time,end_time,status,service  (start/end are numeric seconds from 0; status is severity in [0,1])
    # 2) Start time, End time, Severity     (start/end are timestamps)
    cols = {c.strip().lower(): c for c in df.columns}
    start_col = cols.get("start_time") or cols.get("start time") or cols.get("start")
    end_col = cols.get("end_time") or cols.get("end time") or cols.get("end")
    sev_col = cols.get("status") or cols.get("severity") or cols.get("sev")
    if not (start_col and end_col and sev_col):
        raise ValueError(f"Missing expected columns in {path}: {list(df.columns)}")

    start_raw = df[start_col]
    end_raw = df[end_col]
    sev_raw = df[sev_col]

    # Prefer numeric seconds if the column is overwhelmingly numeric. This avoids the pitfall where
    # pd.to_datetime(0.0) is interpreted as an epoch timestamp, producing misleading 1970-era datetimes.
    start_num = pd.to_numeric(start_raw, errors="coerce")
    end_num = pd.to_numeric(end_raw, errors="coerce")
    num_good = start_num.notna().mean() > 0.95 and end_num.notna().mean() > 0.95
    if num_good:
        out = pd.DataFrame(
            {
                "start_s": start_num,
                "end_s": end_num,
                "severity": pd.to_numeric(sev_raw, errors="coerce"),
            }
        )
        out = out.dropna(subset=["start_s", "end_s"]).copy()
        out = out[out["end_s"] >= out["start_s"]].copy()
        out["duration_min"] = (out["end_s"] - out["start_s"]) / 60.0
        out = out.sort_values("start_s").reset_index(drop=True)
        out["time_axis"] = "seconds_from_zero"
        return out

    # Otherwise fall back to timestamp parsing.
    start_dt = pd.to_datetime(start_raw, errors="coerce", utc=True)
    end_dt = pd.to_datetime(end_raw, errors="coerce", utc=True)
    dt_good = start_dt.notna().mean() > 0.80 and end_dt.notna().mean() > 0.80

    if dt_good:
        out = pd.DataFrame(
            {
                "start_time": start_dt,
                "end_time": end_dt,
                "severity": pd.to_numeric(sev_raw, errors="coerce"),
            }
        )
        out = out.dropna(subset=["start_time", "end_time"]).copy()
        out = out[out["end_time"] >= out["start_time"]].copy()
        out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
        out = out.sort_values("start_time").reset_index(drop=True)
        out["time_axis"] = "timestamp_utc"
        return out

    start_s = start_num
    end_s = end_num
    out = pd.DataFrame(
        {
            "start_s": start_s,
            "end_s": end_s,
            "severity": pd.to_numeric(sev_raw, errors="coerce"),
        }
    )
    out = out.dropna(subset=["start_s", "end_s"]).copy()
    out = out[out["end_s"] >= out["start_s"]].copy()
    out["duration_min"] = (out["end_s"] - out["start_s"]) / 60.0
    out = out.sort_values("start_s").reset_index(drop=True)
    out["time_axis"] = "seconds_from_zero"
    return out


def compute_stats(name: str, df: pd.DataFrame) -> TraceStats:
    n = int(len(df))
    if "start_time" in df.columns:
        start_min = df["start_time"].min().isoformat()
        start_max = df["start_time"].max().isoformat()
    else:
        start_min = f"{float(df['start_s'].min()):.1f}s"
        start_max = f"{float(df['start_s'].max()):.1f}s"

    dur = df["duration_min"].to_numpy(dtype=float)
    dur_median = float(np.median(dur)) if n else float("nan")
    dur_p90 = float(np.percentile(dur, 90)) if n else float("nan")

    # Inter-arrival in minutes between consecutive outage start times.
    if n >= 2:
        if "start_time" in df.columns:
            ia = (
                np.diff(df["start_time"].to_numpy(dtype="datetime64[ns]"))
                .astype("timedelta64[s]")
                .astype(float)
                / 60.0
            )
        else:
            ia = np.diff(df["start_s"].to_numpy(dtype=float)) / 60.0
        ia_median = float(np.median(ia))
        ia_p90 = float(np.percentile(ia, 90))
    else:
        ia_median = float("nan")
        ia_p90 = float("nan")

    sev = df["severity"].to_numpy(dtype=float)
    severity_mean = float(np.nanmean(sev)) if n else float("nan")

    return TraceStats(
        trace_name=name,
        n_outages=n,
        start_min=start_min,
        start_max=start_max,
        dur_median_min=dur_median,
        dur_p90_min=dur_p90,
        ia_median_min=ia_median,
        ia_p90_min=ia_p90,
        severity_mean=severity_mean,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: list[TraceStats] = []
    for p in [Path(x) for x in args.cua_csvs]:
        df = read_cua_csv(p, timezone=args.timezone)
        traces.append(compute_stats(p.name, df))

    out_csv = out_dir / "cloud_uptime_archive_crosscheck.csv"
    out_json = out_dir / "cloud_uptime_archive_crosscheck_summary.json"

    df_out = pd.DataFrame([t.__dict__ for t in traces]).sort_values("trace_name")
    df_out.to_csv(out_csv, index=False)

    out_json.write_text(
        json.dumps(
            {
                "inputs": [str(Path(x)) for x in args.cua_csvs],
                "n_traces": int(len(traces)),
                "notes": (
                    "Cloud Uptime Archive is used here as an external outage-trace cross-check. "
                    "These summaries are intended to support discussion of incident-frequency proxies "
                    "and their high variance / burstiness characteristics."
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
