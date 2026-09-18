import argparse
import logging
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_inference(data_path: Path, emb_path: Path, models_dir: Path, output_dir: Path):
    logging.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["sequence", "bacterium"]).reset_index(drop=True)
    
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    logging.info(f"Loading embeddings: {emb_path}")
    E = np.load(emb_path).astype(np.float32)
    assert len(E) == len(df), f"Mismatch: E={len(E)}, df={len(df)}"

    logging.info("Loading clean CV artifacts...")
    booster = lgb.Booster(model_file=str(models_dir / "lgbm_cv_ranker.txt"))
    scaler = joblib.load(models_dir / "robust_scaler_cv.pkl")
    pca = joblib.load(models_dir / "pca_cv.pkl")
    physchem_cols = joblib.load(models_dir / "physchem_cols.pkl")

    missing_cols = set(physchem_cols) - set(df.columns)
    for mc in missing_cols:
        df[mc] = 0.0

    P = df[physchem_cols].fillna(0).values.astype(np.float32)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    X = np.hstack([pca.transform(E_norm), scaler.transform(P)])
    assert X.shape[1] == booster.num_feature(), f"Shape mismatch: {X.shape[1]} vs {booster.num_feature()}"

    df["predicted_score"] = booster.predict(X)
    df = df.sort_values(by=["bacterium", "predicted_score"], ascending=[True, False]).copy()
    df["rank_in_strain"] = df.groupby("bacterium")["predicted_score"].rank(ascending=False, method="min").astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "ranked_predictions.csv"
    df.to_csv(out_csv, index=False)

    print(f"RANKING COMPLETE. Saved {len(df)} rows across {df['bacterium'].nunique()} strains to {out_csv}")
    print("Top 5 sample:")
    print(df[["bacterium", "sequence", "predicted_score", "rank_in_strain"]].head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/enriched_ranked_dataset.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/features_esm2.npy"))
    parser.add_argument("--models_dir", type=Path, default=Path("models"))
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    run_inference(args.data, args.embeddings, args.models_dir, args.output_dir)