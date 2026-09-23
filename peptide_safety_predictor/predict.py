"""Local inference for hemolysis (HemoPI2 HC50) and SIF stability (ML_Peptide)."""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
HEMO_DIR = EXTERNAL / "hemopi2"
HEMO_SAV = HEMO_DIR / "Model" / "HemoPI2_reg.sav"
HEMO_SCRIPT = HEMO_DIR / "Model" / "composition_calculate_hemopi2_2.py"
SIF_PATH = EXTERNAL / "ml_peptide" / "SIF_model"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HEMO_MAX_LEN = 40
SIF_FEATURES = [
    "MinAbsEStateIndex",
    "qed",
    "MinPartialCharge",
    "Chi1v",
    "PEOE_VSA8",
    "SMR_VSA10",
    "SMR_VSA4",
    "SMR_VSA6",
    "SlogP_VSA3",
    "EState_VSA10",
    "EState_VSA2",
    "EState_VSA6",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState4",
    "VSA_EState8",
]
SIF_LABELS = ("Not Stable", "Partly Stable", "Stable")


def _require_files() -> None:
    missing = [p for p in (HEMO_SAV, HEMO_SCRIPT, SIF_PATH) if not p.exists()]
    if missing:
        lines = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(
            "Model files are missing. Download them as described in README.md:\n" + lines
        )


def _patch_forest(model) -> None:
    """sklearn>=1.4 expects monotonic_cst on trees pickled by sklearn 1.3."""
    for estimator in getattr(model, "estimators_", []):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise SystemExit(
            f"{path} has no 'sequence' column. Columns: {list(df.columns)}"
        )
    return df


def normalize_sequence(raw: object) -> tuple[str | None, str | None]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, "empty sequence"
    text = str(raw).strip().upper()
    if not text or text == "NAN":
        return None, "empty sequence"
    bad = sorted(set(text) - AA20)
    if bad:
        return None, "nonstandard residues: " + "".join(bad)
    return text, None


def hemopi2_features(sequences: list[str]) -> pd.DataFrame:
    """Run the authors' composition script. Sequences must already be <= 40 aa."""
    with tempfile.TemporaryDirectory(dir=HEMO_DIR, prefix="_infer_") as tmp:
        work = Path(tmp)
        seq_file = work / "sequences.txt"
        seq_file.write_text("\n".join(sequences) + "\n", encoding="utf-8")
        out_csv = work / "features.csv"
        proc = subprocess.run(
            [sys.executable, str(HEMO_SCRIPT), str(seq_file), str(work), str(out_csv)],
            cwd=str(HEMO_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_csv.exists():
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"HemoPI2 feature script failed:\n{tail}")
        features = pd.read_csv(out_csv)
    features = features.apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise RuntimeError("HemoPI2 feature matrix contains non-numeric values.")
    return features


def predict_hemopi2(sequences: list[str]) -> pd.DataFrame:
    scored = [seq[:HEMO_MAX_LEN] for seq in sequences]
    truncated = [len(seq) > HEMO_MAX_LEN for seq in sequences]
    print(f"HemoPI2: features for {len(scored)} sequences (official truncation at {HEMO_MAX_LEN} aa)")
    features = hemopi2_features(scored)
    print(f"HemoPI2: loading regressor ({features.shape[1]} features)")
    with HEMO_SAV.open("rb") as handle:
        model = pickle.load(handle)
    _patch_forest(model)
    if features.shape[1] != model.n_features_in_:
        raise RuntimeError(
            f"Feature width {features.shape[1]} != model.n_features_in_ {model.n_features_in_}"
        )
    x = features
    hc50 = np.exp(-model.predict(x))
    tree_pred = np.vstack([est.predict(x) for est in model.estimators_])
    tree_hc50 = np.exp(-tree_pred)
    fraction = (tree_hc50 > 128.0).mean(axis=0)
    labels = np.where(hc50 < 100.0, "Hemolytic", "Non-hemolytic")
    return pd.DataFrame(
        {
            "hemopi2_sequence_scored": scored,
            "hemopi2_truncated_to_40": truncated,
            "hemopi2_hc50_uM": np.round(hc50, 4),
            "hemopi2_class_if_hc50_lt_100": labels,
            "hemopi2_tree_fraction_hc50_gt_128": np.round(fraction, 4),
        }
    )


def predict_sif(sequences: list[str]) -> pd.DataFrame:
    import xgboost as xgb
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    print("ML_Peptide: SIF descriptors")
    names = [name for name, _ in Descriptors._descList]
    missing = [name for name in SIF_FEATURES if name not in names]
    if missing:
        raise RuntimeError(f"RDKit is missing descriptors required by ML_Peptide: {missing}")
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(SIF_FEATURES)
    rows = []
    for seq in sequences:
        mol = Chem.MolFromSequence(seq)
        if mol is None:
            raise ValueError(f"RDKit could not build a peptide from {seq}")
        rows.append(calculator.CalcDescriptors(mol))
    raw = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError("RDKit produced non-finite SIF descriptors.")

    with SIF_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    scaler = bundle["feature_scaler"]
    # Fitted MinMaxScaler from sklearn 0.24: X_scaled = X * scale_ + min_.
    # transform() itself is not callable across this sklearn jump.
    scaled = raw * np.asarray(scaler.scale_, dtype=np.float64) + np.asarray(scaler.min_, dtype=np.float64)
    categories = list(bundle["GI_encoder"].categories_[0])
    if categories != ["Intestinal"]:
        raise RuntimeError(f"Unexpected SIF environment encoder categories: {categories}")
    env = np.zeros((len(sequences), 1), dtype=np.float64)
    matrix = np.concatenate([env, scaled], axis=1)
    booster = bundle["clf"].get_booster()
    proba = booster.predict(xgb.DMatrix(matrix))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    labels = list(bundle["Label_encoder"].categories_[0])
    if tuple(labels) != SIF_LABELS:
        raise RuntimeError(f"Unexpected SIF labels: {labels}")
    pred_idx = proba.argmax(axis=1)
    return pd.DataFrame(
        {
            "ml_peptide_sif_class": [labels[i] for i in pred_idx],
            "ml_peptide_sif_p_not_stable": np.round(proba[:, 0], 4),
            "ml_peptide_sif_p_partly_stable": np.round(proba[:, 1], 4),
            "ml_peptide_sif_p_stable": np.round(proba[:, 2], 4),
        }
    )


def predict_frame(sequences: list[str]) -> pd.DataFrame:
    parts = [
        predict_hemopi2(sequences),
        predict_sif(sequences),
    ]
    return pd.concat(parts, axis=1)


def run(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    _require_files()
    table = load_table(input_csv)
    cleaned: list[str] = []
    errors: list[str | None] = []
    for raw in table["sequence"]:
        seq, err = normalize_sequence(raw)
        if err:
            cleaned.append("")
            errors.append(err)
        else:
            cleaned.append(seq)
            errors.append(None)
    if any(errors):
        bad = [f"row {i}: {err}" for i, err in enumerate(errors) if err]
        raise SystemExit("Invalid sequences:\n" + "\n".join(bad[:20]))

    preds = predict_frame(cleaned)
    out = pd.concat([table.reset_index(drop=True), preds], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} rows to {output_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict peptide HC50 and SIF stability.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with a sequence column")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
