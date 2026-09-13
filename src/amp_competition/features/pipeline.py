"""Orchestrate AMP vs putative non-AMP physicochemical analysis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from amp_competition.features.cohorts import (
    clean_nonamp_records,
    load_amp_records,
    sample_unmatched,
)
from amp_competition.features.comparison import run_comparison, save_length_distribution
from amp_competition.features.descriptors import compute_descriptors
from amp_competition.features.physchem_constants import HYDROPHOBICITY_SCALES, UNIPROT_QUERY
from amp_competition.features.uniprot import download_uniprot_fasta
from amp_competition.io import write_fasta_records


def _sample_length_matched_cohorts(
    amp_records: list[tuple[str, str]],
    nonamp_records: list[tuple[str, str]],
    *,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Build AMP and non-AMP control cohorts with exactly identical length distributions.

    For each peptide length, sample without replacement
    min(n_amp, n_nonamp) records from each class.
    """
    amp_by_length: dict[int, list[tuple[str, str]]] = {}
    nonamp_by_length: dict[int, list[tuple[str, str]]] = {}

    for record in amp_records:
        amp_by_length.setdefault(len(record[1]), []).append(record)

    for record in nonamp_records:
        nonamp_by_length.setdefault(len(record[1]), []).append(record)

    matched_amp: list[tuple[str, str]] = []
    matched_nonamp: list[tuple[str, str]] = []

    for length in sorted(set(amp_by_length) & set(nonamp_by_length)):
        amp_group = amp_by_length[length]
        nonamp_group = nonamp_by_length[length]
        n = min(len(amp_group), len(nonamp_group))

        if n == 0:
            continue

        amp_idx = rng.choice(len(amp_group), size=n, replace=False)
        nonamp_idx = rng.choice(len(nonamp_group), size=n, replace=False)

        matched_amp.extend(amp_group[i] for i in amp_idx)
        matched_nonamp.extend(nonamp_group[i] for i in nonamp_idx)

    return matched_amp, matched_nonamp


def run_physchem_analysis(
    *,
    amp_fasta: Path,
    outdir: Path,
    downloaded_nonamp_fasta: Path | None = None,
    unmatched_size: str = "amp",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run the full physicochemical comparison pipeline.

    Returns a summary dict with cohort sizes and output paths.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    amp_records = load_amp_records(amp_fasta)
    amp_sequences = {sequence for _, sequence in amp_records}

    cleaned_nonamp_fasta = outdir / "uniprot_putative_nonamp_clean.fasta"

    if cleaned_nonamp_fasta.exists():
        print(f"Using existing cleaned non-AMP dataset: {cleaned_nonamp_fasta}")
        nonamp_records = load_amp_records(cleaned_nonamp_fasta)
        cleaning_stats = {
            "reused_cleaned_dataset": True,
            "cleaned_sequences": len(nonamp_records),
        }
    else:
        if downloaded_nonamp_fasta is None:
            raw_nonamp_fasta = outdir / "uniprot_putative_nonamp_raw.fasta.gz"
            if not raw_nonamp_fasta.exists():
                print("Downloading putative non-AMP sequences from UniProtKB...")
                download_uniprot_fasta(raw_nonamp_fasta)
            else:
                print(f"Using existing download: {raw_nonamp_fasta}")
        else:
            raw_nonamp_fasta = downloaded_nonamp_fasta

        print("Cleaning non-AMP dataset...")
        nonamp_records, cleaning_stats = clean_nonamp_records(
            raw_nonamp_fasta,
            amp_sequences,
            cleaned_nonamp_fasta,
        )

    unmatched_records = sample_unmatched(
        nonamp_records,
        amp_size=len(amp_records),
        size_spec=unmatched_size,
        rng=rng,
    )

    matched_amp_records, matched_records = _sample_length_matched_cohorts(
        amp_records,
        nonamp_records,
        rng=rng,
    )

    write_fasta_records(
        unmatched_records,
        outdir / "nonamp_unmatched.fasta",
    )
    write_fasta_records(
        matched_records,
        outdir / "nonamp_length_matched.fasta",
    )
    write_fasta_records(
        matched_amp_records,
        outdir / "amp_length_matched.fasta",
    )

    save_length_distribution(
        amp_records,
        unmatched_records,
        matched_records,
        outdir,
    )

    metadata = {
        "seed": seed,
        "uniprot_query": UNIPROT_QUERY,
        "amp_input": str(amp_fasta),
        "amp_unique_valid": len(amp_records),
        "nonamp_cleaning": dict(cleaning_stats),
        "nonamp_unmatched": len(unmatched_records),
        "amp_length_matched": len(matched_amp_records),
        "nonamp_length_matched": len(matched_records),
        "hydrophobicity_scales": HYDROPHOBICITY_SCALES,
        "notes": {
            "negative_class": (
                "Putative non-AMP UniProtKB sequences without the excluded "
                "annotation terms; not experimentally validated inactive peptides."
            ),
            "hydrophobic_moment": (
                "Exploratory structural descriptor. Alpha uses 100 degrees and "
                "beta uses 160 degrees. Window is 11 residues, or the complete "
                "sequence for peptides shorter than 11 aa."
            ),
            "correlation": (
                "Spearman correlation is used only for redundancy analysis, "
                "not as a descriptor quality score."
            ),
            "pvalues": (
                "Mann-Whitney and KS p-values are auxiliary. Descriptor selection "
                "should emphasize physical interpretation and effect size/"
                "distribution separation rather than significance alone."
            ),
        },
    }

    with open(outdir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("Computing AMP descriptors...")
    amp_df = compute_descriptors(amp_records, "AMP")
    amp_df.to_csv(outdir / "amp_descriptors.csv", index=False)

    print("Computing unmatched non-AMP descriptors...")
    unmatched_df = compute_descriptors(unmatched_records, "nonAMP")
    unmatched_df.to_csv(
        outdir / "nonamp_unmatched_descriptors.csv",
        index=False,
    )

    print("Computing length-matched AMP descriptors...")
    matched_amp_df = compute_descriptors(matched_amp_records, "AMP")
    matched_amp_df.to_csv(
        outdir / "amp_length_matched_descriptors.csv",
        index=False,
    )

    print("Computing length-matched non-AMP descriptors...")
    matched_df = compute_descriptors(matched_records, "nonAMP")
    matched_df.to_csv(
        outdir / "nonamp_length_matched_descriptors.csv",
        index=False,
    )

    print("Running unmatched comparison...")
    unmatched_comparison = run_comparison(
        amp_df,
        unmatched_df,
        cohort_name="unmatched",
        output_dir=outdir,
    )

    print("Running length-matched comparison...")
    matched_comparison = run_comparison(
        matched_amp_df,
        matched_df,
        cohort_name="length_matched",
        output_dir=outdir,
    )

    combined = unmatched_comparison.merge(
        matched_comparison,
        on="descriptor",
        how="outer",
        suffixes=("_unmatched", "_length_matched"),
    )
    combined.to_csv(
        outdir / "descriptor_comparison_combined.csv",
        index=False,
    )

    print("\nDone.")
    print(f"AMP sequences: {len(amp_records):,}")
    print(f"Clean unique non-AMP sequences: {len(nonamp_records):,}")
    print(f"Unmatched non-AMP cohort: {len(unmatched_records):,}")
    print(f"Length-matched AMP cohort: {len(matched_amp_records):,}")
    print(f"Length-matched non-AMP cohort: {len(matched_records):,}")
    print(f"Results: {outdir.resolve()}")

    return {
        "amp_unique_valid": len(amp_records),
        "nonamp_cleaning": dict(cleaning_stats) if isinstance(cleaning_stats, Counter) else cleaning_stats,
        "nonamp_unmatched": len(unmatched_records),
        "amp_length_matched": len(matched_amp_records),
        "nonamp_length_matched": len(matched_records),
        "outdir": str(outdir.resolve()),
    }
