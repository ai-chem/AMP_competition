#!/usr/bin/env python
"""Score external predictors on the locked test + contamination audit."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.evaluation import classification_metrics, regression_metrics  # noqa: E402
from psp.paths import EXTERNAL, PROCESSED, REPORTS, ensure_dirs  # noqa: E402


def load_locked_test():
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    test = df[df["split"] == "test"].copy()
    test["_pref"] = (test["censor_type"] != "exact").astype(int)
    return test.sort_values(["sequence", "_pref"]).groupby("sequence", as_index=False).first()


def run_hemopi2(sequences: list[str]) -> pd.DataFrame:
    hemo_dir = EXTERNAL / "hemopi2"
    script = hemo_dir / "Model" / "composition_calculate_hemopi2_2.py"
    sav = hemo_dir / "Model" / "HemoPI2_reg.sav"
    if not script.exists() or not sav.exists():
        raise FileNotFoundError("HemoPI2 model files missing")
    scored = [s[:40] for s in sequences]
    with tempfile.TemporaryDirectory(dir=hemo_dir, prefix="_ext_") as tmp:
        work = Path(tmp)
        seq_file = work / "sequences.txt"
        seq_file.write_text("\n".join(scored) + "\n", encoding="utf-8")
        out_csv = work / "features.csv"
        proc = subprocess.run(
            [sys.executable, str(script), str(seq_file), str(work), str(out_csv)],
            cwd=str(hemo_dir),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_csv.exists():
            raise RuntimeError(proc.stderr[-2000:] if proc.stderr else "hemopi2 failed")
        features = pd.read_csv(out_csv).apply(pd.to_numeric, errors="coerce")
    with sav.open("rb") as f:
        model = pickle.load(f)
    for est in getattr(model, "estimators_", []):
        if not hasattr(est, "monotonic_cst"):
            est.monotonic_cst = None
    # HemoPI2 predicts -log(HC50)
    pred = np.exp(-model.predict(features))
    return pd.DataFrame({"hemopi2_hc50_uM": pred, "hemopi2_log": np.log(np.clip(pred, 1e-9, None))})


def contamination_audit(test: pd.DataFrame) -> dict:
    """Flag overlap with HemoPI2 curated training data (DBAASP+Hemolytik lineage)."""
    frames = []
    for name in ("cross_val_dataset.csv", "independent_dataset.csv"):
        p = EXTERNAL / "hemopi2" / "Dataset" / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return {"status": "hemopi2_datasets_missing"}
    hp = pd.concat(frames, ignore_index=True)
    seq_col = "SEQUENCE" if "SEQUENCE" in hp.columns else hp.columns[0]
    hp_seqs = set(hp[seq_col].astype(str).str.upper().str.strip())
    test_seqs = set(test["sequence"])
    exact = test_seqs & hp_seqs
    return {
        "predictor": "HemoPI2",
        "potentially_contaminated": True,
        "reason": "HemoPI2 was trained on DBAASP+Hemolytik; our test is also DBAASP-derived",
        "exact_sequence_overlap_with_hemopi2_curated": len(exact),
        "exact_overlap_fraction": len(exact) / max(len(test_seqs), 1),
        "n_test": len(test_seqs),
        "note": "Do NOT interpret HemoPI2 locked-test metrics as independent generalization.",
    }, hp_seqs


def hemopi2_seen_vs_unseen(test, preds, hp_seqs) -> dict:
    """Score HemoPI2 separately on sequences it did and did not train on.

    The gap between the two is a direct measurement of how much of HemoPI2's
    apparent accuracy is memorisation rather than generalization. Only the
    unseen subset is a fair external comparison against our own model, and even
    that is optimistic because near-duplicates are not excluded here.
    """
    exact = (test["censor_type"] == "exact").to_numpy()
    seen_mask = test["sequence"].isin(hp_seqs).to_numpy()
    y = test["hc50_log_value"].to_numpy(dtype=float)
    p = preds["hemopi2_log"].to_numpy(dtype=float)

    out = {}
    for label, mask in (("seen_in_hemopi2_training", exact & seen_mask),
                        ("unseen_by_hemopi2", exact & ~seen_mask)):
        if mask.sum() >= 5:
            met = regression_metrics(y[mask], p[mask])
            met["n"] = int(mask.sum())
            out[label] = met
        else:
            out[label] = {"n": int(mask.sum()), "note": "too few observations to score"}

    a = out["seen_in_hemopi2_training"].get("pearson_r")
    b = out["unseen_by_hemopi2"].get("pearson_r")
    if a is not None and b is not None:
        out["memorisation_gap_pearson"] = float(a - b)
    return out


def main():
    ensure_dirs()
    test = load_locked_test()
    print(f"Locked test sequences: {len(test)}")
    results = []
    audit, hp_seqs = contamination_audit(test)

    # HemoPI2
    try:
        preds = run_hemopi2(test["sequence"].tolist())
        exact = test["censor_type"] == "exact"
        met = regression_metrics(
            test.loc[exact, "hc50_log_value"].to_numpy(dtype=float),
            preds.loc[exact.to_numpy(), "hemopi2_log"].to_numpy(dtype=float),
        )
        row = {
            "predictor": "HemoPI2",
            "family": "external",
            "potentially_contaminated": True,
            **met,
        }
        # P(HC50>128) from point estimate
        psafe = (preds["hemopi2_hc50_uM"] > 128).astype(float)
        known = test["y_safe_gt_128"].notna()
        if known.sum() and known.sum() == len(test):
            from psp.evaluation import classification_metrics

            clf = classification_metrics(
                test.loc[known, "y_safe_gt_128"].to_numpy(dtype=float),
                psafe[known.to_numpy()].to_numpy(),
            )
            row.update({f"psafe_{k}": v for k, v in clf.items()})
        results.append(row)
        print(f"HemoPI2 pearson={met.get('pearson_r')} (CONTAMINATED)")
        preds.to_csv(REPORTS / "external_hemopi2_test_preds.csv", index=False)
        audit["seen_vs_unseen"] = hemopi2_seen_vs_unseen(test, preds, hp_seqs)
        print("HemoPI2 seen vs unseen:", json.dumps(audit["seen_vs_unseen"], indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        results.append({"predictor": "HemoPI2", "error": str(exc), "potentially_contaminated": True})
        print(f"HemoPI2 failed: {exc}")

    # ConsAMPHemo / others — attempt if cloned
    clones = EXTERNAL / "_clones"
    for name, note in [
        ("ConsAMPHemo", "quantitative HC50; check trained-model/"),
        ("PeptideBERT", "hemolysis head if checkpoint present"),
        ("LysePred", "retraining-oriented; no released checkpoint expected"),
        ("HemoNet", "weights.hdf if present"),
        ("ML-guided-discovery-and-design-of-non-hemolytic-peptides", "binary .pkl classifiers"),
    ]:
        path = clones / name
        status = {
            "predictor": name,
            "family": "external",
            "cloned": path.exists(),
            "note": note,
            "potentially_contaminated": True,
        }
        if path.exists():
            ckpts = list(path.rglob("*.pkl")) + list(path.rglob("*.pt")) + list(path.rglob("*.h5")) + list(path.rglob("*.hdf*")) + list(path.rglob("*.sav"))
            status["checkpoint_files_found"] = [str(p.relative_to(path)) for p in ckpts[:20]]
            status["n_checkpoints"] = len(ckpts)
            if not ckpts:
                status["inference_reproducible"] = False
                status["exclusion_reason"] = "no checkpoint artifacts in cloned repo"
            else:
                status["inference_reproducible"] = False
                status["exclusion_reason"] = (
                    "checkpoints present but environment/API not wired for automated "
                    "locked-test scoring in this run; see external_audit.md"
                )
        else:
            status["inference_reproducible"] = False
            status["exclusion_reason"] = "repo not cloned"
        results.append(status)

    (REPORTS / "contamination_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    out = pd.DataFrame(results)
    out.to_csv(REPORTS / "external_baseline_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
