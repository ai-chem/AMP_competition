#!/usr/bin/env python
"""Build leakage-controlled cluster-disjoint splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.features.clustering import (  # noqa: E402
    assign_locked_split,
    greedy_cluster,
    max_train_identity,
)
from psp.paths import HC50_SAFE_THRESHOLD_UM, PROCESSED, REPORTS, SPLITS, ensure_dirs  # noqa: E402


def _hc50_band(row) -> str:
    if row["censor_type"] == "right":
        return "right_censored"
    v = row.get("hc50_value")
    if pd.isna(v):
        return "unknown"
    if v <= 10:
        return "le10"
    if v <= 50:
        return "10_50"
    if v <= HC50_SAFE_THRESHOLD_UM:
        return "50_128"
    return "gt128"


def main() -> None:
    ensure_dirs()
    obs_path = PROCESSED / "hc50_observations.parquet"
    if not obs_path.exists():
        raise SystemExit(f"Missing {obs_path}; run prepare_data.py first")
    df = pd.read_parquet(obs_path)
    print(f"Loaded {len(df)} observations, {df['sequence'].nunique()} unique sequences")

    # One row per unique sequence for clustering — keep the "primary" observation
    # Prefer exact over censored; among exact prefer median-closest later.
    prim = (
        df.assign(_exact_first=(df["censor_type"] != "exact").astype(int))
        .sort_values(["sequence", "_exact_first"], kind="mergesort")
        .drop_duplicates("sequence", keep="first")
        .drop(columns="_exact_first")
        .reset_index(drop=True)
    )
    prim["hc50_band"] = prim.apply(_hc50_band, axis=1)
    prim["censor_flag"] = (prim["censor_type"] != "exact").astype(int)

    # Cluster at 70%
    assign70 = greedy_cluster(prim["sequence"].tolist(), identity_threshold=0.70, coverage_threshold=0.80)
    prim["cluster_id70"] = prim["sequence"].map(assign70)
    print(f"Clusters @70%: {prim['cluster_id70'].nunique()}")

    # Cluster at 80% (sensitivity)
    assign80 = greedy_cluster(prim["sequence"].tolist(), identity_threshold=0.80, coverage_threshold=0.80)
    prim["cluster_id80"] = prim["sequence"].map(assign80)
    print(f"Clusters @80%: {prim['cluster_id80'].nunique()}")

    # Locked split on 70% clusters
    prim["cluster_id"] = prim["cluster_id70"]
    prim["split"] = assign_locked_split(
        prim,
        cluster_col="cluster_id",
        test_fraction=0.18,
        seed=42,
        stratify_cols=["censor_flag", "hc50_band"],
    )

    # PubMed-disjoint split: hold out publications
    if "pubmed_id" in prim.columns and prim["pubmed_id"].notna().any():
        pubs = prim["pubmed_id"].fillna("UNKNOWN").astype(str)
        rng = np.random.default_rng(42)
        uniq_pubs = [p for p in pubs.unique() if p != "UNKNOWN"]
        rng.shuffle(uniq_pubs)
        n_hold = max(1, int(round(len(uniq_pubs) * 0.15)))
        hold_pubs = set(uniq_pubs[:n_hold])
        prim["split_pubmed"] = np.where(pubs.isin(hold_pubs), "test", "train")
        # UNKNOWN pubmed stays in train
        prim.loc[pubs == "UNKNOWN", "split_pubmed"] = "train"
    else:
        prim["split_pubmed"] = "train"

    # Propagate split back to all observations
    split_map = prim.set_index("sequence")["split"].to_dict()
    cluster_map = prim.set_index("sequence")["cluster_id70"].to_dict()
    cluster80_map = prim.set_index("sequence")["cluster_id80"].to_dict()
    pubmed_map = prim.set_index("sequence")["split_pubmed"].to_dict()
    df["split"] = df["sequence"].map(split_map)
    df["cluster_id70"] = df["sequence"].map(cluster_map)
    df["cluster_id80"] = df["sequence"].map(cluster80_map)
    df["split_pubmed"] = df["sequence"].map(pubmed_map)
    df["hc50_band"] = df.apply(_hc50_band, axis=1)

    # Save
    prim.to_csv(SPLITS / "sequence_primary_id70.csv", index=False)
    df.to_parquet(PROCESSED / "hc50_observations_with_splits.parquet", index=False)
    df.to_csv(SPLITS / "split_locked.csv", index=False)

    # ---- split audit: max train→test identity ----
    train_seqs = prim.loc[prim["split"] == "train", "sequence"].tolist()
    test_prim = prim[prim["split"] == "test"].copy()
    print(f"Auditing {len(test_prim)} test sequences against {len(train_seqs)} train...")
    nearest = []
    violations = 0
    for _, row in test_prim.iterrows():
        ident, cov, neigh = max_train_identity(row["sequence"], train_seqs)
        same_source = False
        same_pub = False
        if neigh:
            nsrc = prim.loc[prim["sequence"] == neigh, "source"]
            same_source = bool(len(nsrc) and nsrc.iloc[0] == row.get("source"))
            if "pubmed_id" in prim.columns:
                npub = prim.loc[prim["sequence"] == neigh, "pubmed_id"]
                same_pub = bool(
                    len(npub)
                    and pd.notna(npub.iloc[0])
                    and pd.notna(row.get("pubmed_id"))
                    and str(npub.iloc[0]) == str(row.get("pubmed_id"))
                )
        flag = ident >= 0.70
        if flag:
            violations += 1
        nearest.append(
            {
                "test_sequence": row["sequence"],
                "nearest_train_sequence": neigh,
                "sequence_identity": ident,
                "alignment_coverage": cov,
                "same_source": same_source,
                "same_publication": same_pub,
                "violation_id70": flag,
            }
        )
    audit = pd.DataFrame(nearest)
    audit.to_csv(REPORTS / "split_audit.csv", index=False)

    summary = {
        "n_observations": int(len(df)),
        "n_unique_sequences": int(prim["sequence"].nunique()),
        "n_clusters_70": int(prim["cluster_id70"].nunique()),
        "n_clusters_80": int(prim["cluster_id80"].nunique()),
        "n_train_seq": int((prim["split"] == "train").sum()),
        "n_test_seq": int((prim["split"] == "test").sum()),
        "n_train_obs": int((df["split"] == "train").sum()),
        "n_test_obs": int((df["split"] == "test").sum()),
        "max_train_test_identity": float(audit["sequence_identity"].max()) if len(audit) else None,
        "n_violations_id70": int(violations),
        "censor_train": df.loc[df["split"] == "train", "censor_type"].value_counts().to_dict(),
        "censor_test": df.loc[df["split"] == "test", "censor_type"].value_counts().to_dict(),
    }
    (REPORTS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if violations:
        raise SystemExit(
            f"PIPELINE ERROR: {violations} test sequences have ≥70% identity to train. "
            "See reports/split_audit.csv"
        )
    print("Split audit passed.")


if __name__ == "__main__":
    main()
