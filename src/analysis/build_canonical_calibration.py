"""
Canonical calibration pipeline for Figure 3.

Builds a single, auditable calibration artifact chain from the processed
incident-proxy table:

    processed_with_load_proxies.csv
        -> calibration_dataset.csv
        -> calibration_summary.json
        -> calibration_report.txt
        -> fig3_calibration.pdf / .png

The model is intentionally simple and transparent:

    lambda_proxy(rho) = lambda_0 * exp(alpha * rho)

where rho is the proxy load ratio (`severity_load`) and lambda_proxy is the
incident-frequency proxy (`failure_rate_proxy`).
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings


SEED = 42
BOOTSTRAP_N = 1000
INPUT_FILE = settings.PROCESSED_DATA_DIR / "processed_with_load_proxies.csv"
OUTPUT_DIR = settings.PROCESSED_DATA_DIR / "canonical_calibration"
AZURE_BINNED_FILE = OUTPUT_DIR / "azure_vm_crosscheck_binned.csv"


def load_canonical_table() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    required = ["severity_load", "failure_rate_proxy", "source", "impact", "incident_time"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["severity_load", "failure_rate_proxy", "source"]).copy()
    df = df[df["failure_rate_proxy"] > 0].copy()
    df["incident_time"] = pd.to_datetime(df["incident_time"], errors="coerce", utc=True)
    df["log_failure_rate_proxy"] = np.log(df["failure_rate_proxy"])
    return df


def fit_log_linear(df: pd.DataFrame) -> dict:
    x = df["severity_load"].to_numpy()
    y = df["log_failure_rate_proxy"].to_numpy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "alpha": float(slope),
        "log_lambda_0": float(intercept),
        "lambda_0": float(np.exp(intercept)),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_rows": int(len(df)),
    }


def fit_log_linear_xy(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "alpha": float(slope),
        "log_lambda_0": float(intercept),
        "lambda_0": float(np.exp(intercept)),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_rows": int(len(x)),
    }


def bootstrap_alpha_xy(x: np.ndarray, y: np.ndarray, n_bootstrap: int = BOOTSTRAP_N) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(x)
    alphas = []
    intercepts = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        try:
            slope, intercept, _, _, _ = stats.linregress(xb, yb)
        except ValueError:
            continue
        alphas.append(slope)
        intercepts.append(intercept)

    alphas = np.asarray(alphas, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    return {
        "n_bootstrap": int(len(alphas)),
        "alpha_median": float(np.median(alphas)),
        "alpha_ci_lower": float(np.percentile(alphas, 2.5)),
        "alpha_ci_upper": float(np.percentile(alphas, 97.5)),
        "alpha_std": float(np.std(alphas)),
        "intercept_median": float(np.median(intercepts)),
        "bootstrap_alphas": alphas,
        "bootstrap_intercepts": intercepts,
    }


def try_load_azure_binned() -> pd.DataFrame | None:
    if not AZURE_BINNED_FILE.exists():
        return None
    df = pd.read_csv(AZURE_BINNED_FILE)
    required = ["rho_center", "n_total", "n_burst", "p_burst"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Azure binned file missing columns: {missing}")
    return df


def bootstrap_alpha(df: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N) -> dict:
    rng = np.random.default_rng(SEED)
    x = df["severity_load"].to_numpy()
    y = df["log_failure_rate_proxy"].to_numpy()
    n = len(df)
    alphas = []
    intercepts = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        try:
            slope, intercept, _, _, _ = stats.linregress(xb, yb)
        except ValueError:
            continue
        alphas.append(slope)
        intercepts.append(intercept)

    alphas = np.asarray(alphas, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    return {
        "n_bootstrap": int(len(alphas)),
        "alpha_median": float(np.median(alphas)),
        "alpha_ci_lower": float(np.percentile(alphas, 2.5)),
        "alpha_ci_upper": float(np.percentile(alphas, 97.5)),
        "alpha_std": float(np.std(alphas)),
        "intercept_median": float(np.median(intercepts)),
        "bootstrap_alphas": alphas,
        "bootstrap_intercepts": intercepts,
    }


def leave_one_source_out(df: pd.DataFrame) -> list[dict]:
    rows = []
    for source in sorted(df["source"].dropna().unique()):
        subset = df[df["source"] != source].copy()
        if len(subset) < 10:
            continue
        fit = fit_log_linear(subset)
        rows.append(
            {
                "held_out_source": source,
                "alpha": fit["alpha"],
                "lambda_0": fit["lambda_0"],
                "r_squared": fit["r_squared"],
                "p_value": fit["p_value"],
                "n_rows": fit["n_rows"],
            }
        )
    return rows


def summarize_levels(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["severity_load", "impact"], dropna=False)
        .agg(
            n=("failure_rate_proxy", "size"),
            mean_failure_rate_proxy=("failure_rate_proxy", "mean"),
            median_failure_rate_proxy=("failure_rate_proxy", "median"),
        )
        .reset_index()
        .sort_values(["severity_load", "impact"])
    )
    return grouped


def save_outputs(
    df: pd.DataFrame,
    fit: dict,
    bootstrap: dict,
    loocv_rows: list[dict],
    level_summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_dataset = OUTPUT_DIR / "calibration_dataset.csv"
    calibration_summary = OUTPUT_DIR / "calibration_summary.json"
    calibration_report = OUTPUT_DIR / "calibration_report.txt"
    loocv_csv = OUTPUT_DIR / "calibration_loocv.csv"
    level_csv = OUTPUT_DIR / "calibration_level_summary.csv"

    df.to_csv(calibration_dataset, index=False)
    pd.DataFrame(loocv_rows).to_csv(loocv_csv, index=False)
    level_summary.to_csv(level_csv, index=False)

    summary = {
        "input_file": str(INPUT_FILE),
        "dataset_rows": int(len(df)),
        "sources": df["source"].value_counts().to_dict(),
        "impacts": df["impact"].fillna("missing").value_counts().to_dict(),
        "model": "log(failure_rate_proxy) = log(lambda_0) + alpha * severity_load",
        "seed": SEED,
        "primary_fit": fit,
        "bootstrap": {
            "n_bootstrap": bootstrap["n_bootstrap"],
            "alpha_median": bootstrap["alpha_median"],
            "alpha_ci_lower": bootstrap["alpha_ci_lower"],
            "alpha_ci_upper": bootstrap["alpha_ci_upper"],
            "alpha_std": bootstrap["alpha_std"],
        },
        "loocv": loocv_rows,
    }
    calibration_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "CANONICAL CALIBRATION REPORT",
        "=" * 70,
        f"Input table: {INPUT_FILE}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "Inclusion rules:",
        "- required non-missing columns: severity_load, failure_rate_proxy, source",
        "- failure_rate_proxy > 0",
        "",
        f"Rows retained: {len(df)}",
        f"Sources: {df['source'].value_counts().to_dict()}",
        f"Impacts: {df['impact'].fillna('missing').value_counts().to_dict()}",
        "",
        "Primary fit:",
        f"- alpha: {fit['alpha']:.6f}",
        f"- lambda_0: {fit['lambda_0']:.6f}",
        f"- R^2: {fit['r_squared']:.6f}",
        f"- p-value: {fit['p_value']:.6f}",
        "",
        "Bootstrap (row resampling):",
        f"- n: {bootstrap['n_bootstrap']}",
        f"- alpha median: {bootstrap['alpha_median']:.6f}",
        f"- 95% CI: [{bootstrap['alpha_ci_lower']:.6f}, {bootstrap['alpha_ci_upper']:.6f}]",
        f"- alpha std: {bootstrap['alpha_std']:.6f}",
        "",
        "LOOCV by source:",
    ]
    for row in loocv_rows:
        lines.append(
            f"- hold out {row['held_out_source']}: alpha={row['alpha']:.6f}, "
            f"R^2={row['r_squared']:.6f}, n={row['n_rows']}"
        )
    calibration_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_proxy_panel(ax: plt.Axes, df: pd.DataFrame, fit: dict, bootstrap: dict) -> None:
    color_map = {
        "none": "#9AA0A6",
        "minor": "#4C9F70",
        "major": "#F4A261",
        "critical": "#D1495B",
    }

    rng = np.random.default_rng(SEED)
    jitter = rng.normal(0.0, 0.012, len(df))
    x = np.clip(df["severity_load"].to_numpy() + jitter, 0.0, 1.0)
    y = df["failure_rate_proxy"].to_numpy()
    colors = [color_map.get(str(v), "#457B9D") for v in df["impact"].fillna("missing")]
    ax.scatter(x, y, s=20, c=colors, alpha=0.30, edgecolors="none", label="Incident records")

    xs = np.linspace(df["severity_load"].min(), df["severity_load"].max(), 200)
    ys = fit["lambda_0"] * np.exp(fit["alpha"] * xs)
    ax.plot(xs, ys, color="#B22222", linewidth=2.2, label=f"Fitted curve ($\\alpha={fit['alpha']:.2f}$)")

    alpha_samples = bootstrap["bootstrap_alphas"]
    intercept_samples = bootstrap["bootstrap_intercepts"]
    if len(alpha_samples) and len(intercept_samples):
        sample_idx = np.linspace(0, len(alpha_samples) - 1, min(250, len(alpha_samples))).astype(int)
        curves = np.array(
            [
                np.exp(intercept_samples[i] + alpha_samples[i] * xs)
                for i in sample_idx
            ]
        )
        lower = np.percentile(curves, 2.5, axis=0)
        upper = np.percentile(curves, 97.5, axis=0)
        ax.fill_between(xs, lower, upper, color="#B22222", alpha=0.16, label="95% bootstrap band")

    level_means = df.groupby("severity_load", as_index=False)["failure_rate_proxy"].mean()
    ax.scatter(
        level_means["severity_load"],
        level_means["failure_rate_proxy"],
        s=90,
        marker="D",
        color="#1D3557",
        label="Mean by load level",
        zorder=4,
    )

    ax.set_xlabel("Load proxy $\\rho$")
    ax.set_ylabel("Failure-rate proxy $\\lambda$")
    ax.set_title("Incident proxy calibration")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")


def plot_azure_panel(ax: plt.Axes, azure_df: pd.DataFrame) -> tuple[dict, dict]:
    # Fit on bins with sufficient support and non-zero probability.
    # p_burst is a direct telemetry cross-check: probability that max_cpu spikes above a threshold,
    # conditioned on average load (avg_cpu). We treat this as a "risk proxy" curve.
    d = azure_df.copy()
    d = d[(d["n_total"] >= 5000) & (d["p_burst"] > 0)].copy()
    x = d["rho_center"].to_numpy(dtype=float)
    y = np.log(d["p_burst"].to_numpy(dtype=float))

    fit = fit_log_linear_xy(x, y)
    bootstrap = bootstrap_alpha_xy(x, y)

    ax.scatter(
        d["rho_center"],
        d["p_burst"],
        s=35,
        color="#2A9D8F",
        alpha=0.65,
        edgecolors="none",
        label="Binned burst probability",
    )

    xs = np.linspace(float(d["rho_center"].min()), float(d["rho_center"].max()), 200)
    ys = fit["lambda_0"] * np.exp(fit["alpha"] * xs)
    ax.plot(xs, ys, color="#264653", linewidth=2.2, label=f"Fitted exp. ($\\alpha={fit['alpha']:.2f}$)")

    alpha_samples = bootstrap["bootstrap_alphas"]
    intercept_samples = bootstrap["bootstrap_intercepts"]
    if len(alpha_samples) and len(intercept_samples):
        sample_idx = np.linspace(0, len(alpha_samples) - 1, min(250, len(alpha_samples))).astype(int)
        curves = np.array([np.exp(intercept_samples[i] + alpha_samples[i] * xs) for i in sample_idx])
        lower = np.percentile(curves, 2.5, axis=0)
        upper = np.percentile(curves, 97.5, axis=0)
        ax.fill_between(xs, lower, upper, color="#264653", alpha=0.14, label="95% bootstrap band")

    ax.set_xlabel("Average CPU load $\\rho$ (Azure VM trace)")
    ax.set_ylabel(r"Burst risk proxy $P(\max\mathrm{CPU}\geq 95\%)$")
    ax.set_title("Direct telemetry cross-check")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    return fit, bootstrap


def plot_figure(df: pd.DataFrame, fit: dict, bootstrap: dict) -> None:
    azure = try_load_azure_binned()

    if azure is None:
        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        plot_proxy_panel(ax, df, fit, bootstrap)
        fig.suptitle("Proxy-based calibration of the load-dependent failure model", y=1.02)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
        plot_proxy_panel(axes[0], df, fit, bootstrap)
        plot_azure_panel(axes[1], azure)
        fig.suptitle("Load-dependent risk calibration: incident proxies + telemetry cross-check", y=1.02)

    pdf_path = OUTPUT_DIR / "fig3_calibration.pdf"
    png_path = OUTPUT_DIR / "fig3_calibration.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_canonical_table()
    fit = fit_log_linear(df)
    bootstrap = bootstrap_alpha(df)
    loocv_rows = leave_one_source_out(df)
    level_summary = summarize_levels(df)
    save_outputs(df, fit, bootstrap, loocv_rows, level_summary)
    plot_figure(df, fit, bootstrap)
    print(f"Canonical calibration artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
