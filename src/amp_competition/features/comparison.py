"""Statistical comparison of AMP vs non-AMP descriptor distributions."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance

from amp_competition.features.descriptors import descriptor_columns
from amp_competition.io import FastaRecord


def robust_scale(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = q75 - q25
    return iqr if iqr > 0 else np.nan


def compare_descriptor(
    amp_values: np.ndarray | list[float],
    nonamp_values: np.ndarray | list[float],
) -> dict[str, Any]:
    amp_values = np.asarray(amp_values, dtype=float)
    nonamp_values = np.asarray(nonamp_values, dtype=float)

    amp_values = amp_values[np.isfinite(amp_values)]
    nonamp_values = nonamp_values[np.isfinite(nonamp_values)]

    u_result = mannwhitneyu(
        amp_values,
        nonamp_values,
        alternative="two-sided",
        method="auto",
    )

    cliff_delta = (2.0 * u_result.statistic) / (
        len(amp_values) * len(nonamp_values)
    ) - 1.0

    ks_result = ks_2samp(
        amp_values,
        nonamp_values,
        alternative="two-sided",
        method="auto",
    )

    wasserstein = wasserstein_distance(amp_values, nonamp_values)
    pooled_values = np.concatenate([amp_values, nonamp_values])
    pooled_iqr = robust_scale(pooled_values)
    normalized_wasserstein = (
        wasserstein / pooled_iqr if np.isfinite(pooled_iqr) else np.nan
    )

    return {
        "n_amp": len(amp_values),
        "n_nonamp": len(nonamp_values),
        "amp_mean": np.mean(amp_values),
        "nonamp_mean": np.mean(nonamp_values),
        "amp_median": np.median(amp_values),
        "nonamp_median": np.median(nonamp_values),
        "amp_q25": np.quantile(amp_values, 0.25),
        "amp_q75": np.quantile(amp_values, 0.75),
        "nonamp_q25": np.quantile(nonamp_values, 0.25),
        "nonamp_q75": np.quantile(nonamp_values, 0.75),
        "cliffs_delta": cliff_delta,
        "ks_statistic": ks_result.statistic,
        "ks_pvalue": ks_result.pvalue,
        "wasserstein_distance": wasserstein,
        "wasserstein_pooled_iqr": normalized_wasserstein,
        "mannwhitney_u": u_result.statistic,
        "mannwhitney_pvalue": u_result.pvalue,
    }


def run_comparison(
    amp_df: pd.DataFrame,
    nonamp_df: pd.DataFrame,
    cohort_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = descriptor_columns(amp_df)

    results = []
    for descriptor in descriptors:
        row = {"cohort": cohort_name, "descriptor": descriptor}
        row.update(
            compare_descriptor(
                amp_df[descriptor].values,
                nonamp_df[descriptor].values,
            )
        )
        results.append(row)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(
        output_dir / f"descriptor_comparison_{cohort_name}.csv",
        index=False,
    )

    # Quantiles describe working ranges without declaring an arbitrary
    # "optimal" interval.
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    distribution_rows = []

    for class_name, frame in [("AMP", amp_df), ("nonAMP", nonamp_df)]:
        for descriptor in descriptors:
            values = frame[descriptor].dropna().to_numpy()
            row = {
                "cohort": cohort_name,
                "class": class_name,
                "descriptor": descriptor,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "min": np.min(values),
                "max": np.max(values),
            }
            for quantile in quantiles:
                row[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile)
            distribution_rows.append(row)

    pd.DataFrame(distribution_rows).to_csv(
        output_dir / f"distribution_summary_{cohort_name}.csv",
        index=False,
    )

    plot_dir = output_dir / f"distributions_{cohort_name}"
    plot_dir.mkdir(exist_ok=True)

    for descriptor in descriptors:
        amp_values = amp_df[descriptor].dropna().to_numpy()
        nonamp_values = nonamp_df[descriptor].dropna().to_numpy()

        combined = np.concatenate([amp_values, nonamp_values])
        low, high = np.quantile(combined, [0.005, 0.995])

        if math.isclose(low, high):
            low, high = combined.min(), combined.max()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            amp_values,
            bins=50,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="AMP",
        )
        ax.hist(
            nonamp_values,
            bins=50,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="non-AMP",
        )
        ax.set_xlabel(descriptor)
        ax.set_ylabel("Density")
        ax.set_title(f"{descriptor}: {cohort_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{descriptor}.png", dpi=180)
        plt.close(fig)

    # Correlations are used only to inspect descriptor redundancy.
    amp_corr = amp_df[descriptors].corr(method="spearman")
    nonamp_corr = nonamp_df[descriptors].corr(method="spearman")

    amp_corr.to_csv(output_dir / f"spearman_amp_{cohort_name}.csv")
    nonamp_corr.to_csv(output_dir / f"spearman_nonamp_{cohort_name}.csv")

    return comparison_df


def save_length_distribution(
    amp_records: list[FastaRecord],
    unmatched_records: list[FastaRecord],
    matched_records: list[FastaRecord],
    output_dir: Path,
) -> None:
    lengths = []

    for name, records in [
        ("AMP", amp_records),
        ("nonAMP_unmatched", unmatched_records),
        ("nonAMP_length_matched", matched_records),
    ]:
        counts = Counter(len(sequence) for _, sequence in records)
        for length, count in sorted(counts.items()):
            lengths.append(
                {
                    "class": name,
                    "length": length,
                    "count": count,
                    "fraction": count / len(records),
                }
            )

    pd.DataFrame(lengths).to_csv(
        output_dir / "length_distributions.csv",
        index=False,
    )
